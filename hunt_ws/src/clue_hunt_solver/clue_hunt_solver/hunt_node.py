"""
Clue Chain Hunt - LEADER node. Your solution goes here.

Run (with the simulation + Nav2 already running):
  ros2 run clue_hunt_solver hunt_node

Topics the referee listens to (you MUST publish these):
  /hunt/clues     std_msgs/String            full QR text of every VALID clue you read, in chain order,
                                             e.g. "HUNT:2:7F49:PILLAR RED"
  /hunt/boards    std_msgs/String            "<id> <x> <y>" - your estimate of each board's ArUco centre
                                             in the map frame, e.g. "2 1.65 -3.88"
  /hunt/treasure  geometry_msgs/PoseStamped  (frame "map") the treasure position, once, after the
                                             leader has driven onto it
Topic the follower may listen to (optional):
  /leader/status  std_msgs/String            MOVING / SEARCHING / READING / DONE

Le
  /camera/image_raw, /camera/camera_info   RGB camera (frame cam_optical_link, tilted 5 deg down)
  /scan                                    360 deg LiDAR (frame lidar_link)
  /odom, /imu, /tf                         odometry, IMU, TF (map -> odom -> base_footprint -> ...)
  Nav2 action /navigate_to_pose            (or nav2_simple_commander.BasicNavigator)

Clue text:  HUNT:<id>:<token>:[TREASURE ]<command>
  token = first 4 hex characters (upper case) of SHA-1 of the PREVIOUS valid clue text
          (for board 1: SHA-1 of "START")  ->  python: hashlib.sha1(prev.encode()).hexdigest()[:4].upper()
  A board whose token does not match is a LOOK-ALIKE: ignore it and keep searching.

Not allowed: Gazebo ground truth (/model/...), hard-coded board / pillar / treasure positions.
"""
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import String

import cv2
import numpy as np
from cv_bridge import CvBridge
import tf2_ros
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from tf_transformations import quaternion_from_euler

# Import your pure Python logic modules
from clue_hunt_solver.perception import detect_boards
from clue_hunt_solver.clue_logic import (
    ChainValidator, 
    transform_to_matrix, 
    board_xy_and_axes, 
    rel_target, 
    between_point,
    approach_goal
)

class HuntNode(Node):
    def __init__(self):
        super().__init__('hunt_node')
        
        # Core Utilities
        self.bridge = CvBridge()
        self.validator = ChainValidator()
        self.navigator = BasicNavigator()
        
        self.K = None
        self.dist = None
        self.current_state = "SEARCHING"
        self.latest_scan = None
        
        # TF2 Setup
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Subscribers
        self.create_subscription(Image, '/camera/image_raw', self.on_image, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, '/camera/camera_info', self.on_info, qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        
        # Publishers
        self.clue_pub = self.create_publisher(String, '/hunt/clues', 10)
        self.board_pub = self.create_publisher(String, '/hunt/boards', 10)
        self.treasure_pub = self.create_publisher(PoseStamped, '/hunt/treasure', 10)
        self.status_pub = self.create_publisher(String, '/leader/status', 10)
        
        # State Machine Timer (Runs at 2Hz)
        self.timer = self.create_timer(0.5, self.control_loop)
        self.get_logger().info('hunt_node initialized and waiting for Nav2.')
        
        # Wait for Nav2 to be fully active before starting
        self.navigator.waitUntilNav2Active()
        self.set_status("SEARCHING")

    def set_status(self, status):
        self.current_state = status
        self.status_pub.publish(String(data=status))
        self.get_logger().info(f"Leader Status: {status}")

    def on_info(self, msg):
        # Extract intrinsic matrix K and distortion coefficients for solvePnP
        if self.K is None:
            self.K = np.array(msg.k).reshape((3, 3))
            self.dist = np.array(msg.d)

    def on_scan(self, msg):
        # Cache the latest LiDAR scan for obstacle avoidance or pillar detection
        self.latest_scan = msg

    def get_map_to_cam_transform(self):
        """Retrieve the latest TF transform from map to the camera optical frame."""
        try:
            # Query the transform tree for the math bridging the map and camera frames
            t = self.tf_buffer.lookup_transform(
                'map', 'camera_link_optical', rclpy.time.Time()
            )
            # Convert the geometry_msgs transform into a 4x4 numpy matrix
            T_map_cam = transform_to_matrix(
                t.transform.translation.x, t.transform.translation.y, t.transform.translation.z,
                t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w
            )
            return T_map_cam
        except Exception as e:
            self.get_logger().warn(f"TF Lookup failed: {e}")
            return None

    def on_image(self, msg):
        """Process the RGB camera feed to detect and validate clues."""
        if self.K is None or self.current_state == "MOVING":
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # 1. Detect ArUco markers and decode QR codes (OpenCV)
        boards = detect_boards(cv_image, self.K, self.dist)
        
        for board in boards:
            if board.qr_text and board.T_cam_board is not None:
                # 2. Validate the Clue Token (SHA-1)
                clue, status = self.validator.check(board.qr_text)
                
                if status == 'ok':
                    self.get_logger().info(f"Valid Clue Found: {clue.text}")
                    self.set_status("READING")
                    self.process_valid_clue(board, clue)
                elif "decoy" not in status:
                    self.get_logger().debug(f"Ignored: {status}")

    def process_valid_clue(self, board, clue):
        """Calculate global positions, publish results, and command Nav2."""
        # 3. Transform Local Camera Pose to Global Map Pose (TF2)
        T_map_cam = self.get_map_to_cam_transform()
        if T_map_cam is None:
            return
            
        T_map_board = T_map_cam @ board.T_cam_board
        
        # 4. Extract global X and Y and publish to referee topics
        map_x, map_y = board_xy_and_axes(T_map_board)[:2]
        
        self.clue_pub.publish(String(data=clue.text))
        self.board_pub.publish(String(data=f"{clue.id} {map_x:.2f} {map_y:.2f}"))
        self.validator.accept(clue)
        
        # Check if this board reveals the TREASURE
        if clue.treasure:
            self.handle_treasure(clue, T_map_board)
            return

        # 5. Parse command logic and calculate next goal
        next_x, next_y = self.calculate_next_target(clue, T_map_board)
        if next_x is not None:
            self.send_nav_goal(next_x, next_y)

    def calculate_next_target(self, clue, T_map_board):
        """Use clue_logic geometry to determine the next (x, y) map coordinate."""
        if clue.cmd == 'GOTO':
            return clue.args[0], clue.args[1]
            
        elif clue.cmd == 'REL':
            target_pt = rel_target(T_map_board, clue.args[0], clue.args[1])
            return target_pt[0], target_pt[1]
            
        elif clue.cmd == 'PILLAR':
            # TODO: Implement vision/LiDAR pillar localization dynamically.
            # Using placeholder coordinates for demonstration.
            pillar_color = clue.args[0]
            self.get_logger().info(f"Searching for {pillar_color} PILLAR...")
            pillar_pos = self.find_pillar(pillar_color)
            return pillar_pos[0], pillar_pos[1]
            
        elif clue.cmd == 'BETWEEN':
            # TODO: Implement dynamic localization of both pillars.
            pA = self.find_pillar(clue.args[0])
            pB = self.find_pillar(clue.args[1])
            target_pt = between_point(pA, pB, clue.args[2])
            return target_pt[0], target_pt[1]
            
        return None, None

    def find_pillar(self, color):
        """
        Placeholder: Must be replaced with dynamic RGB color thresholding (cv2.inRange) 
        combined with LiDAR depth mapping to find global pillar coordinates.
        """
        # Return a dummy coordinate for now to prevent breaking the state machine
        return (0.0, 0.0)

    def send_nav_goal(self, x, y):
        """Dispatch the calculated coordinate to the Nav2 Action Server."""
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.pose.position.x = float(x)
        goal_pose.pose.position.y = float(y)
        
        # Simple default orientation facing forward along the X axis
        goal_pose.pose.orientation.w = 1.0
        
        self.navigator.goToPose(goal_pose)
        self.set_status("MOVING")
        self.get_logger().info(f"Navigating to next goal: X={x:.2f}, Y={y:.2f}")

    def handle_treasure(self, clue, T_map_board):
        """Process the final treasure location and terminate the hunt."""
        treasure_x, treasure_y = self.calculate_next_target(clue, T_map_board)
        
        if treasure_x is not None:
            treasure_pose = PoseStamped()
            treasure_pose.header.frame_id = 'map'
            treasure_pose.header.stamp = self.get_clock().now().to_msg()
            treasure_pose.pose.position.x = float(treasure_x)
            treasure_pose.pose.position.y = float(treasure_y)
            treasure_pose.pose.orientation.w = 1.0
            
            # Publish to referee and drive to it
            self.treasure_pub.publish(treasure_pose)
            self.navigator.goToPose(treasure_pose)
            self.set_status("MOVING")
            self.get_logger().info("TREASURE LOCATED. Proceeding to final coordinate.")

    def control_loop(self):
        """Asynchronous timer loop to check Nav2 status without blocking callbacks."""
        if self.current_state == "MOVING":
            if self.navigator.isTaskComplete():
                result = self.navigator.getResult()
                if result == TaskResult.SUCCEEDED:
                    self.get_logger().info('Goal reached! Searching for next board...')
                    self.set_status("SEARCHING")
                    # TODO: Implement a rotation/frontier search behavior here if the 
                    # board isn't immediately visible upon arrival.
                else:
                    self.get_logger().error(f'Navigation failed with status: {result}')
                    self.set_status("SEARCHING")

def main():
    rclpy.init()
    node = HuntNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
