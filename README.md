# Clue Chain Hunt: Autonomous Multi-Robot Navigation & Visual Servoing

An end-to-end autonomous multi-robot system developed in **ROS 2 Humble** and simulated in **Gazebo Fortress**. The project deploys a dual-robot convoy—a **Leader** ("hunter") and a **Follower**—to solve a dynamic, sequential clue chain hunt. The stack integrates real-time computer vision, cryptographic anti-spoofing verification, SLAM, Nav2 trajectory tracking, and Image-Based Visual Servoing (IBVS).

---

## 📂 Repository & Workspace Architecture

```text
hunt_ws/
├── maps/                               # Saved 2D occupancy grid maps
├── src/
│   ├── clue_hunt_description/          # Robot kinematic definitions & URDF models
│   │   ├── launch/
│   │   │   └── rsp.launch.py           # Robot State Publisher node launcher
│   │   ├── meshes/
│   │   │   ├── leader_tag.obj          # 3D ArUco ID 49 visual marker mesh
│   │   │   ├── leader_tag.mtl
│   │   │   └── leader_tag.png
│   │   └── urdf/
│   │       ├── robot.urdf.xacro        # Base differential-drive robot description
│   │       ├── gazebo.xacro            # Sensor plugins (Lidar, RGB Camera, diff_drive)
│   │       └── inertial_macros.xacro   # Standard geometric inertia calculations
│   │
│   ├── clue_hunt_gazebo/               # Simulation environment & world definitions
│   │   ├── launch/
│   │   │   └── sim.launch.py           # World ignition spawner & Gazebo bridge
│   │   ├── models/                     # Custom arena models & clue targets
│   │   │   ├── board_practice_b1..b5/  # Sequential clue boards (ArUco + QR code pairs)
│   │   │   ├── board_practice_d7/      # Decoy board (invalid cryptographic token)
│   │   │   ├── board_practice_x4/      # Look-alike board (out-of-sequence ArUco)
│   │   │   └── roboclub_banner/        # Arena asset meshes
│   │   └── worlds/
│   │       └── practice.sdf            # Gazebo Fortress simulation world file
│   │
│   ├── clue_hunt_navigation/           # Mapping & navigation configurations
│   │   ├── config/
│   │   │   ├── nav2_params.yaml        # Costmaps, DWB controller, and Navfn planner settings
│   │   │   └── slam_params.yaml        # slam_toolbox 2D online asynchronous SLAM settings
│   │   ├── launch/
│   │   │   ├── mapping.launch.py       # SLAM mapping launch file
│   │   │   └── navigation.launch.py    # Nav2 stack + pre-configured RViz2 interface
│   │   └── rviz/
│   │       └── hunt.rviz               # Dual-camera, costmap, and TF visualization layout
│   │
│   └── clue_hunt_solver/               # Primary autonomy, perception & tracking logic
│       ├── clue_hunt_solver/
│       │   ├── hunt_node.py            # Main mission state machine & Nav2 action dispatcher
│       │   ├── perception.py           # CLAHE filtering, ArUco solvePnP, & QR rectification
│       │   ├── clue_logic.py           # SHA-1 anti-spoofing validator & instruction parser
│       │   └── follower_node.py        # Visual servoing P-controller & breadcrumb odometry
│       └── launch/
│           └── hunt.launch.py          # Unified launcher for solver and follower nodes
