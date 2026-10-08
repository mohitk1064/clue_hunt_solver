#!/usr/bin/env python3
"""
Clue Chain Hunt - FOLLOWER node. Your solution goes here.

Run (with the simulation running):
  ros2 run clue_hunt_solver follower_node

The follower is a second robot with ONLY:
  /follower/camera/image_raw, /follower/camera/camera_info   RGB camera (tilted 8 deg UP,
                                                              frame follower/cam_optical_link)
  /follower/odom                                             wheel odometry
  TF frames that start with "follower/"                      follower/odom -> follower/base_footprint -> ...
  /leader/status (optional)                                  MOVING / SEARCHING / READING / DONE
and drives with:
  /follower/cmd_vel                                          geometry_msgs/Twist

It has NO LiDAR and NO map. It must NOT use the leader's /odom, /tf frames, /cmd_vel, Nav2 topics
or any Gazebo ground truth.

The leader carries an ArUco marker on its back: DICT_4X4_50, id 49, side 0.12 m,
centre 0.30 m above the floor and 0.21 m behind the leader's centre, facing backwards.

Goal: stay 0.6 - 2.0 m from the leader during the whole hunt, never touch a wall or the leader,
and finish within 1.5 m of the treasure.
Hint: aiming straight at the leader cuts corners into walls...
"""
import math
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import tf2_ros

# Import the helper from your perception module
from clue_hunt_solver.perception import detect_leader_tag


class FollowerNode(Node):
    def __init__(self):
        super().__init__('follower_node')
        self.create_subscription(Image, '/follower/camera/image_raw', self.on_image, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, '/follower/camera/camera_info', self.on_info, qos_profile_sensor_data)
        self.create_subscription(String, '/leader/status', self.on_status, 10)
        self.cmd_pub = self.create_publisher(Twist, '/follower/cmd_vel', 10)
        
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.K = None
        self.dist = None
        self.leader_status = "UNKNOWN"
        
        # Breadcrumb trail to prevent corner-cutting
        self.path = [] 
        
        self.create_timer(0.05, self.control)
        self.get_logger().info('follower_node started - find the leader!')

    def on_image(self, msg):
        # TODO 1: detect the leader's tag (id 49) and estimate its pose (solvePnP, 0.12 m)
        if self.K is None:
            return
            
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        detection = detect_leader_tag(cv_img, self.K, self.dist)
        
        if detection is not None:
            T_cam_tag, _ = detection
            
            # TODO 2: express the leader's position in follower/odom (tf2)
            try:
                # Get transform from follower's odom to its camera
                t = self.tf_buffer.lookup_transform(
                    'follower/odom', 'follower/cam_optical_link', rclpy.time.Time()
                )
                
                T_odom_cam = self.transform_to_matrix(
                    t.transform.translation.x, t.transform.translation.y, t.transform.translation.z,
                    t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w
                )
                
                # Transform local tag pose into follower's global odometry frame
                T_odom_tag = T_odom_cam @ T_cam_tag
                leader_x, leader_y = T_odom_tag[0, 3], T_odom_tag[1, 3]
                
                # Drop a breadcrumb if we moved far enough from the last one (0.2m)
                if not self.path:
                    self.path.append((leader_x, leader_y))
                else:
                    last_x, last_y = self.path[-1]
                    if math.hypot(leader_x - last_x, leader_y - last_y) > 0.2:
                        self.path.append((leader_x, leader_y))
            except Exception as e:
                pass

    def on_info(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape((3, 3))
            self.dist = np.array(msg.d)

    def on_status(self, msg):
        self.leader_status = msg.data

    def control(self):
        # TODO 3: follow the leader safely and publish /follower/cmd_vel
        cmd = Twist()
        try:
            # Locate where the follower currently is in its own odometry
            t = self.tf_buffer.lookup_transform('follower/odom', 'follower/base_footprint', rclpy.time.Time())
            bot_x = t.transform.translation.x
            bot_y = t.transform.translation.y
            
            # Convert quaternion to yaw angle
            q = t.transform.rotation
            bot_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            
            # Prune waypoints we have already reached (within 0.4m)
            while self.path:
                wp_x, wp_y = self.path[0]
                if math.hypot(wp_x - bot_x, wp_y - bot_y) < 0.4:
                    self.path.pop(0)
                else:
                    break
                    
            if self.path:
                # Steer towards the *oldest* unreached breadcrumb to trace the path
                target_x, target_y = self.path[0]
                
                # Measure distance to the *newest* breadcrumb (the leader's actual location)
                leader_x, leader_y = self.path[-1]
                dist_to_leader = math.hypot(leader_x - bot_x, leader_y - bot_y)
                
                # Calculate required heading to hit the target breadcrumb
                yaw_error = math.atan2(target_y - bot_y, target_x - bot_x) - bot_yaw
                yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi
                
                # Distance regulation (0.6 - 2.0m goal limit)
                if dist_to_leader < 0.8:
                    cmd.linear.x = 0.0  # Safe zone reached, halt forward motion
                else:
                    cmd.linear.x = min(0.6, 0.5 * math.hypot(target_x - bot_x, target_y - bot_y))
                    
                # Steering regulation
                cmd.angular.z = max(-1.0, min(1.0, 1.5 * yaw_error))
            else:
                # If no path exists, spin slowly to search for the leader tag
                cmd.angular.z = 0.4
                
            self.cmd_pub.publish(cmd)
            
        except Exception:
            pass

    def transform_to_matrix(self, tx, ty, tz, qx, qy, qz, qw):
        """Helper to convert geometry_msgs Transform to 4x4 matrix."""
        n = math.sqrt(qx**2 + qy**2 + qz**2 + qw**2) or 1.0
        x, y, z, w = qx / n, qy / n, qz / n, qw / n
        T = np.eye(4)
        T[:3, :3] = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w),       2*(x*z + y*w)],
            [2*(x*y + z*w),       1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
        ])
        T[:3, 3] = [tx, ty, tz]
        return T

def main():
    rclpy.init()
    node = FollowerNode()
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
