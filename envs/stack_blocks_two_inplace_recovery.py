"""
Stack Blocks Two Inplace - Recovery Data Generation Variant

This task variant generates robust training data for VLA models by combining:
1. Funnel-based subtask starts: Random valid starting positions for each subtask
2. Via-point trajectory perturbation: Smooth trajectories through perturbed waypoints

Uses RecoveryMixin for generic recovery functionality.

Unlike stack_blocks_two_recovery, this task stacks the second block directly
on the first block at its initial position (no center placement).
"""

from .stack_blocks_two_inplace import stack_blocks_two_inplace
from ._recovery_mixin import RecoveryMixin
from .utils import *
from copy import deepcopy
import numpy as np


class stack_blocks_two_inplace_recovery(RecoveryMixin, stack_blocks_two_inplace):
    """
    Stack two blocks task (inplace) with funnel-based recovery trajectory generation.

    Extends stack_blocks_two_inplace with recovery capabilities from RecoveryMixin.
    """

    # Task-specific funnel configurations (override mixin defaults)
    DEFAULT_FUNNEL_CONFIGS = {
        "approach": {
            "enabled": True,
            "radius_range": [0.08, 0.20],
            "height_range": [-0.03, 0.12],
            "orientation_range": 0.35,
            "max_attempts": 30,
            "distribution": "gaussian",
        },
        "lift": {
            "enabled": False,
            "radius_range": [0.02, 0.05],
            "height_range": [-0.02, 0.02],
            "orientation_range": 0.1,
            "max_attempts": 20,
            "distribution": "gaussian",
        },
        "move_to_target": {
            "enabled": True,
            "radius_range": [0.05, 0.15],
            "height_range": [0.0, 0.08],
            "orientation_range": 0.2,
            "max_attempts": 30,
            "distribution": "gaussian",
        },
        "place": {
            "enabled": True,
            "radius_range": [0.03, 0.08],
            "height_range": [0.02, 0.06],
            "orientation_range": 0.15,
            "max_attempts": 25,
            "distribution": "gaussian",
        },
        # Handover-specific funnel configs
        "move_to_handover": {
            "enabled": True,
            "radius_range": [0.05, 0.12],
            "height_range": [0.0, 0.06],
            "orientation_range": 0.2,
            "max_attempts": 25,
            "distribution": "gaussian",
        },
        "handover_receive": {
            "enabled": True,
            "radius_range": [0.04, 0.10],
            "height_range": [-0.02, 0.04],
            "orientation_range": 0.25,
            "max_attempts": 25,
            "distribution": "gaussian",
        },
    }

    def setup_demo(self, **kwargs):
        """Setup with recovery configuration."""
        # Initialize recovery config first
        self.setup_recovery(**kwargs)
        # Then call parent setup
        super().setup_demo(**kwargs)

    def play_once(self):
        """Execute the task with recovery trajectory generation."""
        # Initialize tracking
        self.last_gripper = None

        # Define block names using dynamic colors
        block1_color = self.block_color_names[0]
        block2_color = self.block_color_names[1]
        self.block_names = {
            id(self.block1): f"{block1_color} block",
            id(self.block2): f"{block2_color} block",
        }

        # In replay mode, use command sequence replay
        if not self.need_plan:
            return self._replay_stored_trajectories()

        # Planning mode: handle random start position
        self._random_start_used = False
        self._random_start_pose = None
        self._random_start_arm = None
        self._random_start_joint_config = None

        if self.random_start_probability > 0 and np.random.random() < self.random_start_probability:
            block2_pose = self.block2.get_pose().p
            stacking_arm_tag = "left" if block2_pose[0] < 0 else "right"

            joint_config, ee_pose = self._generate_random_start_config(
                self.block2, stacking_arm_tag
            )

            if joint_config is not None:
                self._set_arm_to_config(stacking_arm_tag, joint_config)
                self._random_start_used = True
                self._random_start_pose = ee_pose
                self._random_start_arm = stacking_arm_tag
                self._random_start_joint_config = list(joint_config)

        # Execute task: stack block2 on block1 (block1 stays in place)
        arm_tag = self.stack_block_on_target(self.block2, self.block1)

        # Return to home
        self.start_subtask("return_home", arm_tag=str(arm_tag))
        instruction = self._generate_instruction("return_home")
        self.move(self.back_to_origin(arm_tag=ArmTag(arm_tag)))
        self._create_boundary_pause()
        self.end_subtask(instruction)

        # Store info
        self.info["info"] = {
            "{A}": f"{block1_color} block",
            "{B}": f"{block2_color} block",
            "{a}": arm_tag,
        }

        self.info["block_variation"] = {
            "block1_color": block1_color,
            "block2_color": block2_color,
            "block1_size": self.block1_size,
            "block2_size": self.block2_size,
            "num_distractors": self.num_distractors,
            "distractor_colors": self.block_color_names[2:] if self.num_distractors > 0 else [],
        }

        self.info["recovery_generation"] = {
            "funnel_enabled": self.funnel_enabled,
            "funnel_probability": self.funnel_probability,
            "via_point_config": self.via_point_config,
            "boundary_pause_frames": self.boundary_pause_frames,
        }

        self.info["random_start"] = {
            "used": self._random_start_used,
            "probability": self.random_start_probability,
            "start_pose": self._random_start_pose,
        }

        # Store handover info
        self.info["handover"] = {
            "used": getattr(self, '_handover_used', False),
        }

        self.save_subtask_annotations()
        return self.info

    def _replay_stored_trajectories(self):
        """
        Replay stored trajectories using the mixin's generic replay.
        Adds task-specific info to the result.
        """
        block1_color = self.block_color_names[0]
        block2_color = self.block_color_names[1]

        # Use mixin's generic replay
        self._replay_command_sequence()

        # Set task-specific info
        self.info["info"] = {
            "{A}": f"{block1_color} block",
            "{B}": f"{block2_color} block",
        }

        return self.info

    def move(self, actions_by_arm1, actions_by_arm2=None):
        """
        Override parent's move to track command sequence for replay.
        """
        if self.need_plan:
            # Record arm/gripper commands before calling parent
            if actions_by_arm1 is not None:
                arm_tag1 = actions_by_arm1[0]
                actions1 = actions_by_arm1[1]
                if actions1 and len(actions1) > 0:
                    action = actions1[0]
                    action_type = getattr(action, 'action', "move")
                    arm_str = "left" if arm_tag1 == "left" or str(arm_tag1) == "left" else "right"

                    if action_type == "gripper":
                        target_pos = getattr(action, 'target_gripper_pos', 0.5)
                        self.command_sequence.append(("gripper", arm_str, "gripper", target_pos))
                    else:
                        path_list = self.left_joint_path if arm_str == "left" else self.right_joint_path
                        self.command_sequence.append((f"{arm_str}_arm", len(path_list)))

            if actions_by_arm2 is not None:
                arm_tag2 = actions_by_arm2[0]
                actions2 = actions_by_arm2[1]
                if actions2 and len(actions2) > 0:
                    action = actions2[0]
                    action_type = getattr(action, 'action', "move")
                    arm_str = "left" if arm_tag2 == "left" or str(arm_tag2) == "left" else "right"

                    if action_type == "gripper":
                        target_pos = getattr(action, 'target_gripper_pos', 0.5)
                        self.command_sequence.append(("gripper", arm_str, "gripper", target_pos))
                    else:
                        path_list = self.left_joint_path if arm_str == "left" else self.right_joint_path
                        self.command_sequence.append((f"{arm_str}_arm", len(path_list)))

        return super().move(actions_by_arm1, actions_by_arm2)

    def stack_block_on_target(self, block, target_block):
        """
        Pick up a block and stack it on top of target_block with recovery trajectory generation.
        Includes handover support when blocks are on opposite sides of the workspace.
        """
        block_pose = block.get_pose().p
        target_pose_p = target_block.get_pose().p

        # Determine pickup arm based on block position
        pickup_arm = ArmTag("left" if block_pose[0] < 0 else "right")

        # Check if handover is needed (blocks on opposite sides)
        needs_handover = self._needs_handover(block, target_block)

        # Determine which arm will do the final placement
        if needs_handover:
            place_arm = ArmTag("left" if target_pose_p[0] < 0 else "right")
        else:
            place_arm = pickup_arm

        # Current working arm (starts with pickup arm)
        arm_tag = pickup_arm

        block_name = self.block_names.get(id(block), "block")
        target_name = self.block_names.get(id(target_block), "target block")

        # Store handover info for metadata
        self._handover_used = needs_handover

        # Compute grasp poses
        pre_grasp_pose, grasp_pose = self.choose_grasp_pose(
            block, arm_tag=arm_tag, pre_dis=0.09, target_dis=0
        )

        # === SUB-TASK 1: Approach ===
        approach_start = None
        funnel_used = False
        teleport_pause_frames = 0

        if self._should_use_funnel("approach"):
            funnel_start = self._sample_funnel_start(pre_grasp_pose, "approach", arm_tag)
            if funnel_start is not None:
                approach_start = funnel_start
                funnel_used = True
                teleport_pause_frames = self._teleport_arm_to_pose(arm_tag, funnel_start)

        self.start_subtask("approach", obj_name=block_name, arm_tag=str(arm_tag),
                          metadata={"funnel_used": funnel_used, "teleport_pause_frames": teleport_pause_frames})
        instruction = self._generate_instruction("approach", obj_name=block_name)

        if approach_start is None:
            approach_start = self.robot.get_left_ee_pose() if arm_tag == "left" else self.robot.get_right_ee_pose()

        # Move through perturbed via-points
        via_points = self._generate_via_point_trajectory(approach_start, pre_grasp_pose, arm_tag)
        self._move_through_via_points(arm_tag, via_points)

        if pre_grasp_pose != grasp_pose:
            self.move((arm_tag, [Action(arm_tag, "move", target_pose=grasp_pose,
                                        constraint_pose=[1, 1, 1, 0, 0, 0])]))

        self._create_boundary_pause()
        self.end_subtask(instruction)

        # === SUB-TASK 2: Grasp ===
        self.start_subtask("grasp", obj_name=block_name, arm_tag=str(arm_tag))
        instruction = self._generate_instruction("grasp", obj_name=block_name)
        self.move(self.close_gripper(arm_tag, pos=0.0))
        self._create_boundary_pause()
        self.end_subtask(instruction)

        # === SUB-TASK 3: Lift ===
        self.start_subtask("lift", obj_name=block_name, arm_tag=str(arm_tag))
        instruction = self._generate_instruction("lift", obj_name=block_name)

        current_pose = self.robot.get_left_ee_pose() if arm_tag == "left" else self.robot.get_right_ee_pose()
        lift_pose = list(current_pose)
        lift_pose[2] += 0.07

        if self._should_use_funnel("lift"):
            funnel_start = self._sample_funnel_start(lift_pose, "lift", arm_tag)
            if funnel_start is not None:
                via_points = self._generate_via_point_trajectory(current_pose, lift_pose, arm_tag)
                self._move_through_via_points(arm_tag, via_points)
            else:
                self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))
        else:
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))

        self._create_boundary_pause()
        self.end_subtask(instruction)

        # === HANDOVER (if needed) ===
        giver_arm = None
        if needs_handover:
            # Perform handover: transfer from pickup_arm to place_arm
            # Returns both arms so we can do simultaneous home + pre-place
            arm_tag, giver_arm = self._perform_handover_with_recovery(
                block, pickup_arm, place_arm, block_name, self.block2_size
            )

        # Get target position (top of target block)
        target_block_pos = target_block.get_pose().p
        target_z = target_block_pos[2] + self.block1_size + self.block2_size + 0.005  # Stack height + small clearance

        if needs_handover:
            # After handover, we need to:
            # 1. Simultaneously: giver arm goes home + receive arm goes to pre-place
            # 2. Lower to stacking height
            from ._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC

            current_ee = self.robot.get_left_ee_pose() if str(arm_tag) == "left" else self.robot.get_right_ee_pose()

            # Top-down orientation (gripper pointing -Z)
            # Use the appropriate top-down variant for this arm
            if str(arm_tag) == "left":
                top_down_quat = GRASP_DIRECTION_DIC["top_down_little_right"]  # Left arm uses "little right"
                gripper_bias = self.robot.left_gripper_bias
            else:
                top_down_quat = GRASP_DIRECTION_DIC["top_down_little_left"]  # Right arm uses "little left"
                gripper_bias = self.robot.right_gripper_bias

            # Pre-place position: above target with top-down orientation
            # IMPORTANT: In top-down orientation, gripper extends downward by gripper_bias
            # So EE must be higher by gripper_bias to position the gripper tip correctly
            approach_height = target_z + gripper_bias + 0.08  # gripper_bias + 8cm clearance above stacking height
            pre_place_pose = [target_block_pos[0], target_block_pos[1], approach_height] + list(top_down_quat)

            # Final place position at stacking height
            # EE needs to be at target_z + gripper_bias so tip is at target_z
            place_height = target_z + gripper_bias
            place_pose = [target_block_pos[0], target_block_pos[1], place_height] + list(top_down_quat)

            # Get home pose for giver arm
            if str(giver_arm) == "left":
                giver_home_pose = self.robot.left_original_pose
            else:
                giver_home_pose = self.robot.right_original_pose

            # === SUB-TASK: Move above target (simultaneous with giver going home) ===
            funnel_used_transport = False
            teleport_pause_frames_transport = 0

            if self._should_use_funnel("move_to_target"):
                funnel_start = self._sample_funnel_start(pre_place_pose, "move_to_target", arm_tag)
                if funnel_start is not None:
                    funnel_used_transport = True
                    teleport_pause_frames_transport = self._teleport_arm_to_pose(arm_tag, funnel_start)

            self.start_subtask("move_above_target", obj_name=block_name, base_name=target_name, arm_tag=str(arm_tag),
                              metadata={"funnel_used": funnel_used_transport,
                                       "teleport_pause_frames": teleport_pause_frames_transport})
            instruction = self._generate_instruction("move_above_target", base_name=target_name)

            # Simultaneous move: giver arm to home + receive arm to pre-place
            self._together_move_with_tracking(
                arm_tag, pre_place_pose, current_ee,
                giver_arm, giver_home_pose, None
            )

            # Lower to place
            self.move(self.move_to_pose(arm_tag, place_pose))

            self._create_boundary_pause()
            self.end_subtask(instruction)
        else:
            # Normal placement flow (no handover)
            target_pose = target_block.get_functional_point(1)

            # Compute place poses (using the arm that now holds the block)
            place_pre_pose = self.get_place_pose(block, arm_tag, target_pose, functional_point_id=0, pre_dis=0.05, pre_dis_axis="fp")
            place_pose = self.get_place_pose(block, arm_tag, target_pose, functional_point_id=0, pre_dis=0., pre_dis_axis="fp")

            # === SUB-TASK: Move above target ===
            funnel_used_transport = False
            teleport_pause_frames_transport = 0
            transport_start = self.robot.get_left_ee_pose() if str(arm_tag) == "left" else self.robot.get_right_ee_pose()

            if self._should_use_funnel("move_to_target"):
                funnel_start = self._sample_funnel_start(place_pre_pose, "move_to_target", arm_tag)
                if funnel_start is not None:
                    transport_start = funnel_start
                    funnel_used_transport = True
                    teleport_pause_frames_transport = self._teleport_arm_to_pose(arm_tag, funnel_start)

            self.start_subtask("move_above_target", obj_name=block_name, base_name=target_name, arm_tag=str(arm_tag),
                              metadata={"funnel_used": funnel_used_transport, "teleport_pause_frames": teleport_pause_frames_transport})
            instruction = self._generate_instruction("move_above_target", base_name=target_name)

            via_points = self._generate_via_point_trajectory(transport_start, place_pre_pose, arm_tag)
            self._move_through_via_points(arm_tag, via_points)

            self._create_boundary_pause()
            self.end_subtask(instruction)

        # === SUB-TASK: Place (release_stack) ===
        funnel_used_place = False
        teleport_pause_frames_place = 0

        # In handover case, we're already at place_pose after move_above_target
        # In non-handover case, we're at place_pre_pose and need to move to place_pose
        if not needs_handover:
            place_start = self.robot.get_left_ee_pose() if str(arm_tag) == "left" else self.robot.get_right_ee_pose()

            if self._should_use_funnel("place"):
                funnel_start = self._sample_funnel_start(place_pose, "place", arm_tag)
                if funnel_start is not None:
                    place_start = funnel_start
                    funnel_used_place = True
                    teleport_pause_frames_place = self._teleport_arm_to_pose(arm_tag, funnel_start)

        self.start_subtask("release_stack", obj_name=block_name, base_name=target_name, arm_tag=str(arm_tag),
                          metadata={"funnel_used": funnel_used_place, "teleport_pause_frames": teleport_pause_frames_place})
        instruction = self._generate_instruction("release_stack", obj_name=block_name, base_name=target_name)

        if not needs_handover:
            # Tighter perturbations for place
            original_sigma = self.via_point_config["position_sigma"]
            self.via_point_config["position_sigma"] = original_sigma * 0.5

            via_points = self._generate_via_point_trajectory(place_start, place_pose, arm_tag)
            self._move_through_via_points(arm_tag, via_points)

            self.via_point_config["position_sigma"] = original_sigma

        self.move(self.open_gripper(arm_tag, pos=1.0))
        self._create_boundary_pause()
        self.end_subtask(instruction)

        # === SUB-TASK: Retract ===
        self.start_subtask("retract", obj_name=block_name, arm_tag=str(arm_tag))
        instruction = self._generate_instruction("retract")

        # Record block positions before retract (for stability check)
        block_pos_before = block.get_pose().p.copy()
        target_pos_before = target_block.get_pose().p.copy()

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))
        self._create_boundary_pause()

        # Stability check: verify blocks weren't knocked during retract
        if self.need_plan:
            # Let physics settle
            for _ in range(30):
                self.scene.step()

            block_pos_after = block.get_pose().p
            target_pos_after = target_block.get_pose().p

            # Check if blocks moved significantly (knocked over)
            block_drift = np.linalg.norm(block_pos_after[:2] - block_pos_before[:2])
            target_drift = np.linalg.norm(target_pos_after[:2] - target_pos_before[:2])
            height_drop = block_pos_before[2] - block_pos_after[2]

            if block_drift > 0.02 or target_drift > 0.02 or height_drop > 0.01:
                print(f"WARNING: Blocks moved during retract! block_drift={block_drift:.3f}, "
                      f"target_drift={target_drift:.3f}, height_drop={height_drop:.3f}")
                self.plan_success = False

        self.end_subtask(instruction)

        self.last_gripper = arm_tag
        return str(arm_tag)

    def _perform_handover_with_recovery(self, block, pickup_arm, receive_arm, block_name, block_half_size):
        """
        Perform handover with recovery trajectory generation (funnel starts and via-point perturbation).

        Uses parent's _get_handover_poses() for consistent random X-rotation and proper gripper orientations.
        Supports simultaneous moves for efficiency while tracking in command_sequence.

        Args:
            block: The block being transferred
            pickup_arm: ArmTag of the arm currently holding the block
            receive_arm: ArmTag of the arm that will receive the block
            block_name: Name of the block for instructions
            block_half_size: Half-size of the block being transferred

        Returns:
            Tuple of (receive_arm, pickup_arm) for caller to do simultaneous home + pre-place
        """
        # Get handover poses from parent (includes random X-rotation with proper 90° offset)
        holding_pose, receiving_pose = self._get_handover_poses(str(pickup_arm), block_half_size)

        # Approach offset for receiving arm (to avoid collision during approach)
        approach_offset_dist = 0.15  # 15cm offset from handover center

        # Calculate approach pose for receiving arm
        receiving_approach_pose = list(receiving_pose)
        if str(receive_arm) == "left":
            receiving_approach_pose[0] -= approach_offset_dist  # Left arm approaches from -X
        else:
            receiving_approach_pose[0] += approach_offset_dist  # Right arm approaches from +X

        # Retract position for holding arm (same offset, backing away after release)
        holding_retract_pose = list(holding_pose)
        if str(pickup_arm) == "left":
            holding_retract_pose[0] -= approach_offset_dist  # Left arm retracts toward -X
        else:
            holding_retract_pose[0] += approach_offset_dist  # Right arm retracts toward +X

        # === SUB-TASK: Move to handover position (simultaneous) ===
        funnel_used_handover = False
        teleport_pause_frames_handover = 0

        current_pose = self.robot.get_left_ee_pose() if str(pickup_arm) == "left" else self.robot.get_right_ee_pose()

        if self._should_use_funnel("move_to_handover"):
            funnel_start = self._sample_funnel_start(holding_pose, "move_to_handover", pickup_arm)
            if funnel_start is not None:
                current_pose = funnel_start
                funnel_used_handover = True
                teleport_pause_frames_handover = self._teleport_arm_to_pose(pickup_arm, funnel_start)

        self.start_subtask("move_to_handover", obj_name=block_name, arm_tag=str(pickup_arm),
                          metadata={"funnel_used": funnel_used_handover,
                                   "teleport_pause_frames": teleport_pause_frames_handover,
                                   "other_arm": str(receive_arm)})
        instruction = self._generate_instruction("move_to_handover", obj_name=block_name)

        # Simultaneous move: holding arm to handover + receive arm to approach
        # Track in command_sequence for replay
        self._together_move_with_tracking(
            pickup_arm, holding_pose, current_pose,
            receive_arm, receiving_approach_pose, None
        )

        self._create_boundary_pause()
        self.end_subtask(instruction)

        # === SUB-TASK: Receive at handover ===
        funnel_used_receive = False
        teleport_pause_frames_receive = 0

        receive_current = self.robot.get_left_ee_pose() if str(receive_arm) == "left" else self.robot.get_right_ee_pose()

        if self._should_use_funnel("handover_receive"):
            funnel_start = self._sample_funnel_start(receiving_pose, "handover_receive", receive_arm)
            if funnel_start is not None:
                funnel_used_receive = True
                teleport_pause_frames_receive = self._teleport_arm_to_pose(receive_arm, funnel_start)
                receive_current = funnel_start

        self.start_subtask("handover_receive", obj_name=block_name, arm_tag=str(receive_arm),
                          metadata={"other_arm": str(pickup_arm), "funnel_used": funnel_used_receive,
                                   "teleport_pause_frames": teleport_pause_frames_receive})
        instruction = self._generate_instruction("handover_receive", obj_name=block_name,
                                                  other_arm=str(pickup_arm))

        # Store block pose BEFORE receiver approaches (for verification)
        if self.need_plan:
            block_pose_before = block.get_pose()
            block_pos_before = np.array(block_pose_before.p)
            block_quat_before = np.array(block_pose_before.q)

        # Move receive arm to final position (via-points for smooth approach)
        via_points = self._generate_via_point_trajectory(receive_current, receiving_pose, receive_arm)
        self._move_through_via_points(receive_arm, via_points)

        # Close gripper
        self.move(self.close_gripper(receive_arm, pos=0.0))
        self._create_boundary_pause()
        self.end_subtask(instruction)

        # === SUB-TASK: Release from pickup arm ===
        self.start_subtask("handover_release", obj_name=block_name, arm_tag=str(pickup_arm),
                          metadata={"other_arm": str(receive_arm)})
        instruction = self._generate_instruction("handover_release", obj_name=block_name,
                                                  other_arm=str(receive_arm))

        self.move(self.open_gripper(pickup_arm, pos=1.0))
        self._create_boundary_pause()
        self.end_subtask(instruction)

        # Verify block wasn't displaced during handover (planning phase only)
        if self.need_plan:
            # Let physics settle briefly
            for _ in range(20):
                self.scene.step()

            block_pose_after = block.get_pose()
            block_pos_after = np.array(block_pose_after.p)
            block_quat_after = np.array(block_pose_after.q)

            # Check position displacement (XY plane primarily)
            pos_drift_xy = np.linalg.norm(block_pos_after[:2] - block_pos_before[:2])
            pos_drift_z = block_pos_before[2] - block_pos_after[2]  # Positive = dropped

            # Check orientation change using quaternion distance
            # Normalize quaternions and compute dot product (handles sign ambiguity)
            quat_dot = np.abs(np.dot(block_quat_before, block_quat_after))
            quat_dot = np.clip(quat_dot, -1.0, 1.0)
            orientation_angle = 2 * np.arccos(quat_dot)  # Angle in radians

            # Thresholds - scaled by block size for robustness
            pos_threshold = max(0.015, block_half_size * 0.6)  # Min 1.5cm or 60% of block half-size
            z_drop_threshold = block_half_size * 0.5  # Half the block size
            orientation_threshold = 0.35  # ~20 degrees

            handover_failed = False
            failure_reasons = []

            if pos_drift_xy > pos_threshold:
                failure_reasons.append(f"XY drift {pos_drift_xy:.3f}m > {pos_threshold:.3f}m")
                handover_failed = True

            if pos_drift_z > z_drop_threshold:
                failure_reasons.append(f"Z drop {pos_drift_z:.3f}m > {z_drop_threshold:.3f}m")
                handover_failed = True

            if orientation_angle > orientation_threshold:
                failure_reasons.append(f"rotation {np.degrees(orientation_angle):.1f}° > {np.degrees(orientation_threshold):.1f}°")
                handover_failed = True

            if handover_failed:
                print(f"HANDOVER VERIFICATION FAILED: {', '.join(failure_reasons)}")
                print(f"  Block size (half): {block_half_size:.3f}m")
                print(f"  Before: pos={block_pos_before}, quat={block_quat_before}")
                print(f"  After:  pos={block_pos_after}, quat={block_quat_after}")
                self.plan_success = False

        # === SUB-TASK: Retract pickup arm ===
        self.start_subtask("handover_retract", arm_tag=str(pickup_arm))
        instruction = self._generate_instruction("handover_retract")

        # Back off to safe retract position (away from receive arm and block)
        self.move(self.move_to_pose(pickup_arm, holding_retract_pose))
        self._create_boundary_pause()
        self.end_subtask(instruction)

        # Return both arms so caller can do simultaneous home + pre-place
        return receive_arm, pickup_arm

    def _together_move_with_tracking(self, arm1, target1, start1, arm2, target2, start2):
        """
        Execute simultaneous move for both arms with command_sequence tracking.

        Plans paths for both arms and executes them together, tracking in command_sequence
        so replay works correctly.

        Args:
            arm1: First arm tag
            target1: Target pose for first arm
            start1: Start pose for first arm (optional, used for via-points)
            arm2: Second arm tag
            target2: Target pose for second arm
            start2: Start pose for second arm (optional, used for via-points)
        """
        if not self.plan_success:
            return

        arm1_str = "left" if str(arm1) == "left" else "right"
        arm2_str = "left" if str(arm2) == "left" else "right"

        # Determine which is left and which is right for together_move
        if arm1_str == "left":
            left_target = target1
            right_target = target2
        else:
            left_target = target2
            right_target = target1

        if self.need_plan:
            # Track this as a simultaneous move
            self.command_sequence.append(("together_move", {
                "left_path_idx": len(self.left_joint_path),
                "right_path_idx": len(self.right_joint_path),
            }))

        # Use parent's together_move_to_pose which handles planning/replay
        self.together_move_to_pose(left_target, right_target)