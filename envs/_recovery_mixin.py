"""
Recovery Data Generation Mixin

Provides funnel-based subtask starts and via-point trajectory perturbation
for generating robust VLA training data.

The goal is to create data where P(recovery_action | perturbed_state) >> P(perturbation_action),
teaching the model to recover from off-trajectory states while maintaining smooth motion.

Key features:
- Funnel-based random subtask starts (teleportation to diverse starting poses)
- Via-point trajectory perturbation with configurable decay
- TOPP-based smooth trajectory execution
- Command sequence tracking for deterministic replay
- Arm proximity safety checks for teleportation

Usage:
    class MyTask_recovery(RecoveryMixin, MyTask):
        # Define task-specific parts:
        DEFAULT_FUNNEL_CONFIGS = {...}  # Task-specific funnel parameters
        SUBTASK_TEMPLATES = {...}       # Task-specific language templates

        def play_once(self):
            if not self.need_plan:
                return self._replay_stored_trajectories()
            # Task-specific planning logic...
"""

import numpy as np
import transforms3d as t3d
from copy import deepcopy
import os
import pickle


class RecoveryMixin:
    """
    Mixin class providing recovery trajectory generation capabilities.

    Inherit from this mixin along with your base task class to add
    recovery data generation features.

    Required from base class:
        - self.robot (with left/right arm planning and gripper control)
        - self.scene (Sapien scene for simulation stepping)
        - self.need_plan (bool: planning vs replay mode)
        - self.save_data (bool: whether to save data)
        - self.FRAME_IDX (int: current frame counter)
        - self.table_z_bias (float: table height offset)
        - self._take_picture() (method for recording frames)
        - self.take_dense_action() (method for executing trajectories)
    """

    # =========================================================================
    # DEFAULT CONFIGURATIONS (override in subclass for task-specific values)
    # =========================================================================

    # Funnel configuration per subtask type
    # Defines the region of valid starting poses relative to the goal
    DEFAULT_FUNNEL_CONFIGS = {
        "approach": {
            "enabled": True,
            "radius_range": [0.08, 0.20],      # Horizontal distance from goal (m)
            "height_range": [-0.03, 0.12],     # Vertical offset from goal (m)
            "orientation_range": 0.35,          # Max rotation perturbation (rad, ~20 deg)
            "max_attempts": 30,
            "distribution": "uniform",          # "uniform" or "gaussian"
        },
        "lift": {
            "enabled": False,  # Usually start lift from grasp pose
            "radius_range": [0.02, 0.05],
            "height_range": [-0.02, 0.02],
            "orientation_range": 0.1,
            "max_attempts": 20,
            "distribution": "uniform",
        },
        "move_to_target": {
            "enabled": True,
            "radius_range": [0.05, 0.15],
            "height_range": [0.0, 0.08],       # Stay above table when holding object
            "orientation_range": 0.2,
            "max_attempts": 30,
            "distribution": "uniform",
        },
        "place": {
            "enabled": True,
            "radius_range": [0.03, 0.08],      # Tighter funnel for precision
            "height_range": [0.02, 0.06],
            "orientation_range": 0.15,
            "max_attempts": 25,
            "distribution": "uniform",
        },
    }

    # Via-point perturbation defaults
    DEFAULT_VIA_POINT_CONFIG = {
        "enabled": True,
        "num_via_points": 4,                   # Number of intermediate waypoints
        "position_sigma": 0.015,               # Gaussian std for position (m)
        "rotation_sigma": 0.05,                # Gaussian std for rotation (rad)
        "skip_endpoints": True,                # Don't perturb start/end poses
        "decay": "linear",                     # Perturbation decay: "none", "linear", "exponential"
        "decay_rate": 3.0,                     # For exponential decay: exp(-decay_rate * t)
    }

    # Minimum distance between arm end-effectors to allow teleportation
    MIN_ARM_DISTANCE = 0.15  # 15cm - prevents unrealistic arm proximity

    # =========================================================================
    # SETUP AND CONFIGURATION
    # =========================================================================

    def setup_recovery(self, **kwargs):
        """
        Initialize recovery configuration. Call this from setup_demo().

        Args:
            **kwargs: Should contain 'domain_randomization' dict with
                     'recovery_generation' config.
        """
        recovery_config = kwargs.get("domain_randomization", {}).get("recovery_generation", {})

        # Funnel configuration
        self.funnel_enabled = recovery_config.get("funnel_enabled", True)
        self.funnel_probability = recovery_config.get("funnel_probability", 0.7)
        self.funnel_configs = deepcopy(self.DEFAULT_FUNNEL_CONFIGS)

        # Override with user config if provided
        user_funnel_configs = recovery_config.get("funnel_configs", {})
        for subtask_type, config in user_funnel_configs.items():
            if subtask_type in self.funnel_configs:
                self.funnel_configs[subtask_type].update(config)

        # Via-point perturbation configuration
        self.via_point_config = deepcopy(self.DEFAULT_VIA_POINT_CONFIG)
        user_via_config = recovery_config.get("via_point_perturbation", {})
        self.via_point_config.update(user_via_config)

        # Pause frames at subtask boundaries (for clean action chunks)
        self.boundary_pause_frames = recovery_config.get("boundary_pause_frames", 3)

        # Command sequence for replay - tracks order of operations
        self.command_sequence = kwargs.get("command_sequence", [])

    # =========================================================================
    # SUBTASK MARKERS (override base class to track in command_sequence)
    # =========================================================================

    def start_subtask(self, subtask_type: str, obj_name: str = None,
                      base_name: str = None, arm_tag: str = None,
                      metadata: dict = None):
        """
        Override to add subtask markers to command_sequence during planning.
        """
        if self.need_plan:
            self.command_sequence.append(("start_subtask", {
                "type": subtask_type,
                "obj": obj_name,
                "base": base_name,
                "arm": arm_tag,
                "metadata": metadata or {},
            }))
        # Call parent for base functionality
        super().start_subtask(subtask_type, obj_name, base_name, arm_tag, metadata)

    def end_subtask(self, instruction: str = None):
        """
        Override to add subtask markers to command_sequence during planning.
        """
        if self.need_plan:
            self.command_sequence.append(("end_subtask", instruction))
        return super().end_subtask(instruction)

    # =========================================================================
    # TRAJECTORY DATA SERIALIZATION
    # =========================================================================

    def save_traj_data(self, idx):
        """Save trajectory data including command_sequence for replay."""
        file_path = os.path.join(self.save_dir, "_traj_data", f"episode{idx}.pkl")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        traj_data = {
            "left_joint_path": deepcopy(self.left_joint_path),
            "right_joint_path": deepcopy(self.right_joint_path),
            "command_sequence": deepcopy(self.command_sequence),
            # Save random start configuration for replay
            "random_start_used": getattr(self, '_random_start_used', False),
            "random_start_arm": getattr(self, '_random_start_arm', None),
            "random_start_joint_config": getattr(self, '_random_start_joint_config', None),
        }
        with open(file_path, "wb") as f:
            pickle.dump(traj_data, f)

    def load_traj_data(self, idx):
        """Load trajectory data including command_sequence."""
        file_path = os.path.join(self.save_dir, "_traj_data", f"episode{idx}.pkl")
        with open(file_path, "rb") as f:
            traj_data = pickle.load(f)
        return traj_data

    # Alias for compatibility
    def load_tran_data(self, idx):
        """Alias for load_traj_data (compatibility with base class naming)."""
        return self.load_traj_data(idx)

    def set_path_lst(self, args):
        """Override to also load command_sequence from saved data."""
        super().set_path_lst(args)

        # Load command_sequence and random start config from the saved trajectory data
        file_path = os.path.join(self.save_dir, "_traj_data", f"episode{self.ep_num}.pkl")
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                traj_data = pickle.load(f)
            self.command_sequence = traj_data.get("command_sequence", [])
            self._random_start_used = traj_data.get("random_start_used", False)
            self._random_start_arm = traj_data.get("random_start_arm", None)
            self._random_start_joint_config = traj_data.get("random_start_joint_config", None)
        else:
            self.command_sequence = []
            self._random_start_used = False
            self._random_start_arm = None
            self._random_start_joint_config = None

    # =========================================================================
    # FUNNEL SAMPLING
    # =========================================================================

    def _should_use_funnel(self, subtask_type: str) -> bool:
        """Check if funnel start should be used for this subtask."""
        if not self.funnel_enabled:
            return False
        if subtask_type not in self.funnel_configs:
            return False
        if not self.funnel_configs[subtask_type].get("enabled", True):
            return False
        return np.random.random() < self.funnel_probability

    def _sample_funnel_start(self, goal_pose, subtask_type: str, arm_tag):
        """
        Sample a random valid starting pose within the funnel region for a goal.

        Args:
            goal_pose: Target pose [x, y, z, qw, qx, qy, qz]
            subtask_type: Type of subtask (key in funnel_configs)
            arm_tag: Which arm ("left" or "right")

        Returns:
            Valid starting pose if found, None otherwise
        """
        if subtask_type not in self.funnel_configs:
            return None

        config = self.funnel_configs[subtask_type]
        if not config.get("enabled", True):
            return None

        goal_pose = list(goal_pose)
        goal_pos = np.array(goal_pose[:3])
        goal_quat = np.array(goal_pose[3:7])

        # Get planner for this arm
        if arm_tag == "left" or str(arm_tag) == "left":
            plan_func = self.robot.left_plan_path
        else:
            plan_func = self.robot.right_plan_path

        table_z = 0.74 + self.table_z_bias
        distribution = config.get("distribution", "uniform")

        for _ in range(config["max_attempts"]):
            # Sample random offset based on distribution type
            angle = np.random.uniform(0, 2 * np.pi)

            if distribution == "gaussian":
                sigma_r = config["radius_range"][0]
                radius = np.abs(np.random.normal(0, sigma_r))
                radius = np.clip(radius, 0, config["radius_range"][1])

                height_mid = (config["height_range"][0] + config["height_range"][1]) / 2
                sigma_h = (config["height_range"][1] - config["height_range"][0]) / 4
                height_offset = np.random.normal(height_mid, sigma_h)
                height_offset = np.clip(height_offset, config["height_range"][0], config["height_range"][1])
            else:
                # Uniform distribution
                radius = np.random.uniform(*config["radius_range"])
                height_offset = np.random.uniform(*config["height_range"])

            # Compute start position
            start_pos = goal_pos.copy()
            start_pos[0] += radius * np.cos(angle)
            start_pos[1] += radius * np.sin(angle)
            start_pos[2] += height_offset

            # Workspace constraints
            start_pos[0] = np.clip(start_pos[0], -0.35, 0.35)
            start_pos[1] = np.clip(start_pos[1], -0.25, 0.10)
            start_pos[2] = np.clip(start_pos[2], table_z + 0.10, table_z + 0.35)

            # Sample orientation perturbation
            start_quat = self._perturb_orientation(
                goal_quat,
                max_angle=config["orientation_range"]
            )

            start_pose = list(start_pos) + list(start_quat)

            # Verify start pose is reachable
            start_result = plan_func(start_pose)
            if start_result is None or start_result.get("status") != "Success":
                continue

            # Verify we can also reach goal
            goal_result = plan_func(goal_pose)
            if goal_result is not None and goal_result.get("status") == "Success":
                return start_pose

        return None

    # =========================================================================
    # TELEPORTATION
    # =========================================================================

    def _teleport_arm_to_pose(self, arm_tag, target_pose) -> int:
        """
        Teleport arm to a target pose (for funnel starts).

        Plans path to target and directly sets joint positions.
        Records a pause at the teleported position.

        Safety: Rejects teleport if it would place the arm too close to the
        other arm, since teleportation bypasses collision-aware motion planning.

        Args:
            arm_tag: Which arm to teleport
            target_pose: Target pose [x, y, z, qw, qx, qy, qz]

        Returns:
            Number of pause frames recorded (0 if teleport failed)
        """
        # Safety check: ensure teleport won't put arms too close together
        target_pos = np.array(target_pose[:3])
        if arm_tag == "left" or str(arm_tag) == "left":
            other_arm_pose = self.robot.get_right_ee_pose()
        else:
            other_arm_pose = self.robot.get_left_ee_pose()
        other_arm_pos = np.array(other_arm_pose[:3])

        distance = np.linalg.norm(target_pos - other_arm_pos)
        if distance < self.MIN_ARM_DISTANCE:
            return 0

        # Plan to get joint configuration
        if arm_tag == "left" or str(arm_tag) == "left":
            result = self.robot.left_plan_path(target_pose)
            joints = self.robot.left_arm_joints
        else:
            result = self.robot.right_plan_path(target_pose)
            joints = self.robot.right_arm_joints

        if result is None or result.get("status") != "Success":
            return 0

        # Get final joint configuration
        joint_config = result["position"][-1]

        # Store teleport command for replay (during planning phase)
        num_pause_frames = self.boundary_pause_frames * 2
        if self.need_plan:
            arm_str = "left" if arm_tag == "left" or str(arm_tag) == "left" else "right"
            self.command_sequence.append(("teleport", arm_str, list(joint_config), num_pause_frames))

        # Set joints directly
        for i, joint in enumerate(joints):
            joint.set_drive_target(joint_config[i])

        # Settle simulation
        for _ in range(50):
            self.scene.step()

        # Record pause frames at teleported position
        self._create_boundary_pause(num_pause_frames)

        return num_pause_frames

    # =========================================================================
    # VIA-POINT TRAJECTORY GENERATION
    # =========================================================================

    def _generate_via_point_trajectory(self, start_pose, end_pose, arm_tag) -> list:
        """
        Generate a trajectory with perturbed via-points for smooth recovery behavior.

        Creates intermediate waypoints between start and end, perturbs them with
        Gaussian noise that decays toward the goal.

        Args:
            start_pose: Starting pose [x, y, z, qw, qx, qy, qz]
            end_pose: Goal pose
            arm_tag: Which arm to use

        Returns:
            List of waypoint poses forming the perturbed trajectory
        """
        if not self.via_point_config.get("enabled", True):
            return [start_pose, end_pose]

        num_via = self.via_point_config["num_via_points"]
        pos_sigma = self.via_point_config["position_sigma"]
        rot_sigma = self.via_point_config["rotation_sigma"]
        skip_endpoints = self.via_point_config["skip_endpoints"]
        decay_mode = self.via_point_config.get("decay", "linear")
        decay_rate = self.via_point_config.get("decay_rate", 3.0)

        start_pos = np.array(start_pose[:3])
        start_quat = np.array(start_pose[3:7])
        end_pos = np.array(end_pose[:3])
        end_quat = np.array(end_pose[3:7])

        # Create interpolated via-points
        waypoints = []
        for i in range(num_via + 2):  # +2 for start and end
            t = i / (num_via + 1)
            pos = (1 - t) * start_pos + t * end_pos
            quat = self._slerp(start_quat, end_quat, t)
            waypoints.append(list(pos) + list(quat))

        # Apply perturbations to intermediate waypoints
        table_z = 0.74 + self.table_z_bias

        for i in range(len(waypoints)):
            if skip_endpoints and (i == 0 or i == len(waypoints) - 1):
                continue

            t = i / (len(waypoints) - 1) if len(waypoints) > 1 else 0

            # Compute decay factor
            if decay_mode == "linear":
                decay_factor = 1.0 - t
            elif decay_mode == "exponential":
                decay_factor = np.exp(-decay_rate * t)
            else:
                decay_factor = 1.0

            current_pos_sigma = pos_sigma * decay_factor
            current_rot_sigma = rot_sigma * decay_factor

            # Position perturbation
            pos_noise = np.random.normal(0, current_pos_sigma, 3)
            pos_noise[2] *= 0.5  # Reduce Z perturbation

            waypoints[i][0] += pos_noise[0]
            waypoints[i][1] += pos_noise[1]
            waypoints[i][2] += pos_noise[2]
            waypoints[i][2] = max(waypoints[i][2], table_z + 0.03)

            # Orientation perturbation
            waypoints[i][3:7] = self._perturb_orientation(
                waypoints[i][3:7],
                current_rot_sigma
            )

        return waypoints

    def _move_through_via_points(self, arm_tag, waypoints):
        """
        Execute smooth movement through a series of via-points using TOPP.

        In planning mode: Plans trajectory and stores in joint_path
        In replay mode: Executes stored trajectory

        Args:
            arm_tag: Which arm to move
            waypoints: List of poses to move through
        """
        if len(waypoints) < 2:
            return

        # === REPLAY MODE ===
        if not self.need_plan:
            if arm_tag == "left" or str(arm_tag) == "left":
                if self.left_cnt < len(self.left_joint_path):
                    result = deepcopy(self.left_joint_path[self.left_cnt])
                    self.left_cnt += 1
                else:
                    return
            else:
                if self.right_cnt < len(self.right_joint_path):
                    result = deepcopy(self.right_joint_path[self.right_cnt])
                    self.right_cnt += 1
                else:
                    return

            control_seq = {
                "left_arm": result if (arm_tag == "left" or str(arm_tag) == "left") else None,
                "left_gripper": None,
                "right_arm": result if (arm_tag == "right" or str(arm_tag) == "right") else None,
                "right_gripper": None,
            }
            self.take_dense_action(control_seq)
            return

        # === PLANNING MODE ===
        arm_str = "left" if arm_tag == "left" or str(arm_tag) == "left" else "right"

        if arm_str == "left":
            current_qpos = self.robot.get_left_arm_jointState()[:-1]
            plan_func = self.robot.left_plan_path
            mplib_planner = self.robot.left_mplib_planner
        else:
            current_qpos = self.robot.get_right_arm_jointState()[:-1]
            plan_func = self.robot.right_plan_path
            mplib_planner = self.robot.right_mplib_planner

        # Collect joint configurations for each waypoint
        joint_waypoints = [np.array(current_qpos)]
        planning_failed = False

        for waypoint in waypoints:
            result = plan_func(waypoint)
            if result is not None and result.get("status") == "Success":
                joint_config = result["position"][-1]
                joint_waypoints.append(joint_config)
            else:
                planning_failed = True
                break

        if planning_failed or len(joint_waypoints) < 2:
            # Fall back to planning just to endpoint
            result = plan_func(waypoints[-1])
            if result is not None and result.get("status") == "Success":
                if arm_str == "left":
                    path_idx = len(self.left_joint_path)
                    self.left_joint_path.append(deepcopy(result))
                    self.command_sequence.append(("left_arm", path_idx))
                else:
                    path_idx = len(self.right_joint_path)
                    self.right_joint_path.append(deepcopy(result))
                    self.command_sequence.append(("right_arm", path_idx))

                control_seq = {
                    "left_arm": result if arm_str == "left" else None,
                    "left_gripper": None,
                    "right_arm": result if arm_str == "right" else None,
                    "right_gripper": None,
                }
                self.take_dense_action(control_seq)
            else:
                # Complete failure
                empty_result = {
                    "position": np.array([current_qpos]),
                    "velocity": np.array([np.zeros_like(current_qpos)]),
                    "status": "Fail"
                }
                if arm_str == "left":
                    path_idx = len(self.left_joint_path)
                    self.left_joint_path.append(empty_result)
                    self.command_sequence.append(("left_arm", path_idx))
                else:
                    path_idx = len(self.right_joint_path)
                    self.right_joint_path.append(empty_result)
                    self.command_sequence.append(("right_arm", path_idx))
                self.plan_success = False
            return

        # Use TOPP for smooth trajectory
        joint_path = np.array(joint_waypoints)

        try:
            times, positions, velocities, acc, duration = mplib_planner.TOPP(
                joint_path,
                1 / 250,
                verbose=False
            )

            if len(positions) == 0:
                raise ValueError("TOPP returned empty trajectory")

            result = {
                "position": positions,
                "velocity": velocities,
                "status": "Success"
            }

        except Exception as e:
            print(f"TOPP failed: {e}, using waypoint-only fallback")
            result = {
                "position": joint_path,
                "velocity": np.zeros_like(joint_path),
                "status": "Success"
            }

        # Store path for replay
        if arm_str == "left":
            path_idx = len(self.left_joint_path)
            self.left_joint_path.append(deepcopy(result))
            self.command_sequence.append(("left_arm", path_idx))
        else:
            path_idx = len(self.right_joint_path)
            self.right_joint_path.append(deepcopy(result))
            self.command_sequence.append(("right_arm", path_idx))

        # Execute the trajectory
        control_seq = {
            "left_arm": result if arm_str == "left" else None,
            "left_gripper": None,
            "right_arm": result if arm_str == "right" else None,
            "right_gripper": None,
        }
        self.take_dense_action(control_seq)

    # =========================================================================
    # REPLAY INFRASTRUCTURE
    # =========================================================================

    def _replay_command_sequence(self, pre_replay_hook=None, post_replay_hook=None):
        """
        Generic replay of stored command sequence.

        Processes all commands in command_sequence in order, rebuilding
        subtask annotations with correct frame numbers.

        Args:
            pre_replay_hook: Optional callable() to run before replay loop
            post_replay_hook: Optional callable() to run after replay loop

        Returns:
            dict: Task info dictionary
        """
        # Restore random start position if used during planning
        if getattr(self, '_random_start_used', False) and self._random_start_joint_config is not None:
            self._set_arm_to_config(self._random_start_arm, self._random_start_joint_config)

        # Reset subtask annotations for rebuild with correct frame numbers
        self._subtask_annotations = []
        current_subtask_start = None
        current_subtask_info = None

        if pre_replay_hook:
            pre_replay_hook()

        # Execute commands in order
        for cmd in self.command_sequence:
            if cmd[0] == "start_subtask":
                current_subtask_start = self.FRAME_IDX
                current_subtask_info = cmd[1]

            elif cmd[0] == "end_subtask":
                if current_subtask_info is not None:
                    instruction = cmd[1]
                    annotation = {
                        "subtask_id": len(self._subtask_annotations),
                        "type": current_subtask_info["type"],
                        "start_frame": current_subtask_start,
                        "end_frame": self.FRAME_IDX,
                        "num_frames": self.FRAME_IDX - current_subtask_start,
                        "instruction": instruction,
                        "obj": current_subtask_info["obj"],
                        "base": current_subtask_info["base"],
                        "arm": current_subtask_info["arm"],
                        "metadata": current_subtask_info["metadata"],
                    }
                    self._subtask_annotations.append(annotation)
                    current_subtask_info = None

            elif cmd[0] == "gripper":
                _, arm_str, _, target_pos = cmd
                self._execute_gripper_command(arm_str, target_pos)

            elif cmd[0] == "teleport":
                _, arm_str, joint_config, num_pause_frames = cmd
                self._execute_teleport_command(arm_str, joint_config, num_pause_frames)

            elif cmd[0] == "left_arm":
                cmd_idx = cmd[1]
                if cmd_idx < len(self.left_joint_path):
                    result = deepcopy(self.left_joint_path[cmd_idx])
                    if result.get("status") != "Fail":
                        control_seq = {
                            "left_arm": result,
                            "left_gripper": None,
                            "right_arm": None,
                            "right_gripper": None,
                        }
                        self.take_dense_action(control_seq)

            elif cmd[0] == "right_arm":
                cmd_idx = cmd[1]
                if cmd_idx < len(self.right_joint_path):
                    result = deepcopy(self.right_joint_path[cmd_idx])
                    if result.get("status") != "Fail":
                        control_seq = {
                            "left_arm": None,
                            "left_gripper": None,
                            "right_arm": result,
                            "right_gripper": None,
                        }
                        self.take_dense_action(control_seq)

            elif cmd[0] == "together_move":
                # Simultaneous move for both arms
                info = cmd[1]
                left_idx = info["left_path_idx"]
                right_idx = info["right_path_idx"]
                if left_idx < len(self.left_joint_path) and right_idx < len(self.right_joint_path):
                    left_result = deepcopy(self.left_joint_path[left_idx])
                    right_result = deepcopy(self.right_joint_path[right_idx])
                    self._execute_together_move(left_result, right_result)

        if post_replay_hook:
            post_replay_hook()

        # Save subtask annotations
        self.save_subtask_annotations()

        return self.info

    def _execute_teleport_command(self, arm_str: str, joint_config: list, num_pause_frames: int):
        """Execute a teleport command during replay."""
        if arm_str == "left":
            joints = self.robot.left_arm_joints
        else:
            joints = self.robot.right_arm_joints

        for i, joint in enumerate(joints):
            joint.set_drive_target(joint_config[i])

        for _ in range(50):
            self.scene.step()

        self._create_boundary_pause(num_pause_frames)

    def _execute_gripper_command(self, arm_str: str, target_pos: float):
        """Execute a gripper open/close command during replay."""
        if arm_str == "left":
            current_pos = self.robot.get_left_gripper_val()
            gripper_result = self.robot.left_plan_grippers(current_pos, target_pos)
            control_seq = {
                "left_arm": None,
                "left_gripper": gripper_result,
                "right_arm": None,
                "right_gripper": None,
            }
        else:
            current_pos = self.robot.get_right_gripper_val()
            gripper_result = self.robot.right_plan_grippers(current_pos, target_pos)
            control_seq = {
                "left_arm": None,
                "left_gripper": None,
                "right_arm": None,
                "right_gripper": gripper_result,
            }

        self.take_dense_action(control_seq)

    def _execute_together_move(self, left_result, right_result):
        """
        Execute a simultaneous move for both arms during replay.

        This mirrors the execution logic of together_move_to_pose but
        uses pre-planned joint paths from the saved trajectory data.
        """
        left_success = left_result.get("status") == "Success"
        right_success = right_result.get("status") == "Success"

        if not left_success and not right_success:
            return

        left_n_step = left_result["position"].shape[0] if left_success else 0
        right_n_step = right_result["position"].shape[0] if right_success else 0

        now_left_id = 0
        now_right_id = 0
        i = 0

        save_freq = getattr(self, 'save_freq', None)

        if save_freq is not None:
            self._take_picture()

        while now_left_id < left_n_step or now_right_id < right_n_step:
            # Interleave execution to keep arms synchronized by progress percentage
            if (left_success and now_left_id < left_n_step
                    and (not right_success or now_left_id / left_n_step <= now_right_id / right_n_step)):
                self.robot.set_arm_joints(
                    left_result["position"][now_left_id],
                    left_result["velocity"][now_left_id],
                    "left",
                )
                now_left_id += 1

            if (right_success and now_right_id < right_n_step
                    and (not left_success or now_right_id / right_n_step <= now_left_id / left_n_step)):
                self.robot.set_arm_joints(
                    right_result["position"][now_right_id],
                    right_result["velocity"][now_right_id],
                    "right",
                )
                now_right_id += 1

            self.scene.step()

            if self.render_freq and i % self.render_freq == 0:
                self._update_render()
                if hasattr(self, 'viewer') and self.viewer:
                    self.viewer.render()

            if save_freq is not None and i % save_freq == 0:
                self._update_render()
                self._take_picture()
            i += 1

        if save_freq is not None:
            self._take_picture()

    # =========================================================================
    # BOUNDARY PAUSE
    # =========================================================================

    def _create_boundary_pause(self, num_frames: int = None):
        """
        Create a pause at subtask boundary for clean action chunks.
        Records frames while robot is stationary.
        """
        if num_frames is None:
            num_frames = self.boundary_pause_frames

        for _ in range(num_frames):
            self.scene.step()
            if self.save_data:
                self._take_picture()

    # =========================================================================
    # MATH UTILITIES
    # =========================================================================

    def _slerp(self, q1, q2, t) -> list:
        """Spherical linear interpolation between two quaternions."""
        q1 = np.array(q1)
        q2 = np.array(q2)

        dot = np.dot(q1, q2)
        if dot < 0:
            q2 = -q2
            dot = -dot

        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return (result / np.linalg.norm(result)).tolist()

        theta_0 = np.arccos(np.clip(dot, -1, 1))
        theta = theta_0 * t

        q2_perp = q2 - q1 * dot
        q2_perp = q2_perp / np.linalg.norm(q2_perp)

        result = q1 * np.cos(theta) + q2_perp * np.sin(theta)
        return result.tolist()

    def _perturb_orientation(self, quat, max_angle: float) -> list:
        """Apply random rotation perturbation to a quaternion."""
        quat = np.array(quat)

        axis = np.random.randn(3)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-6:
            return quat.tolist()
        axis = axis / axis_norm

        angle = np.clip(np.random.normal(0, max_angle / 2), -max_angle, max_angle)
        delta_quat = t3d.quaternions.axangle2quat(axis, angle)
        perturbed = t3d.quaternions.qmult(delta_quat, quat)
        perturbed = perturbed / np.linalg.norm(perturbed)

        return perturbed.tolist()
