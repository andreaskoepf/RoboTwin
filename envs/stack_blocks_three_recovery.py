"""
Stack Blocks Three - Recovery Data Generation Variant

This task variant generates robust training data for VLA models by combining:
1. Funnel-based subtask starts: Random valid starting positions for each subtask
2. Via-point trajectory perturbation: Smooth trajectories through perturbed waypoints

Uses RecoveryMixin for generic recovery functionality.
"""

from .stack_blocks_three import stack_blocks_three
from ._recovery_mixin import RecoveryMixin
from .utils import *
import numpy as np


class stack_blocks_three_recovery(RecoveryMixin, stack_blocks_three):
    """
    Stack three blocks task with funnel-based recovery trajectory generation.

    Extends stack_blocks_three with recovery capabilities from RecoveryMixin.
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
        self.last_actor = None

        # Define block names using dynamic colors
        block1_color = self.block_color_names[0]
        block2_color = self.block_color_names[1]
        block3_color = self.block_color_names[2]
        self.block_names = {
            id(self.block1): f"{block1_color} block",
            id(self.block2): f"{block2_color} block",
            id(self.block3): f"{block3_color} block",
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
            block1_pose = self.block1.get_pose().p
            first_arm_tag = "left" if block1_pose[0] < 0 else "right"

            joint_config, ee_pose = self._generate_random_start_config(
                self.block1, first_arm_tag
            )

            if joint_config is not None:
                self._set_arm_to_config(first_arm_tag, joint_config)
                self._random_start_used = True
                self._random_start_pose = ee_pose
                self._random_start_arm = first_arm_tag
                self._random_start_joint_config = list(joint_config)

        # Execute task - pick and place all three blocks
        arm_tag1 = self.pick_and_place_block(self.block1)
        arm_tag2 = self.pick_and_place_block(self.block2, base_block=self.block1)
        arm_tag3 = self.pick_and_place_block(self.block3, base_block=self.block2)

        # Return to home
        self.start_subtask("return_home", arm_tag=str(arm_tag3))
        instruction = self._generate_instruction("return_home")

        # Collect unique arm tags to return
        arms_to_return = list(set([arm_tag1, arm_tag2, arm_tag3]))
        if len(arms_to_return) == 1:
            self.move(self.back_to_origin(arm_tag=ArmTag(arms_to_return[0])))
        else:
            self.move(
                self.back_to_origin(arm_tag=ArmTag(arms_to_return[0])),
                self.back_to_origin(arm_tag=ArmTag(arms_to_return[1])),
            )

        self._create_boundary_pause()
        self.end_subtask(instruction)

        # Store info
        self.info["info"] = {
            "{A}": f"{block1_color} block",
            "{B}": f"{block2_color} block",
            "{C}": f"{block3_color} block",
            "{a}": arm_tag1,
            "{b}": arm_tag2,
            "{c}": arm_tag3,
        }

        self.info["block_variation"] = {
            "block1_color": block1_color,
            "block2_color": block2_color,
            "block3_color": block3_color,
            "block1_size": self.block1_size,
            "block2_size": self.block2_size,
            "block3_size": self.block3_size,
            "num_distractors": self.num_distractors,
            "distractor_colors": self.block_color_names[3:] if self.num_distractors > 0 else [],
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

        self.save_subtask_annotations()
        return self.info

    def _replay_stored_trajectories(self):
        """
        Replay stored trajectories using the mixin's generic replay.
        Adds task-specific info to the result.
        """
        block1_color = self.block_color_names[0]
        block2_color = self.block_color_names[1]
        block3_color = self.block_color_names[2]

        # Use mixin's generic replay
        self._replay_command_sequence()

        # Set task-specific info
        self.info["info"] = {
            "{A}": f"{block1_color} block",
            "{B}": f"{block2_color} block",
            "{C}": f"{block3_color} block",
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

    def pick_and_place_block(self, block, base_block=None):
        """
        Pick up a block and place it with recovery trajectory generation.
        """
        block_pose = block.get_pose().p
        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")

        block_name = self.block_names.get(id(block), "block")
        base_name = self.block_names.get(id(base_block), None) if base_block else None

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

        # Handle opposite arm if needed
        if self.last_gripper is not None and (self.last_gripper != arm_tag):
            self.move(self.back_to_origin(arm_tag=arm_tag.opposite))

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

        # Determine target pose
        if self.last_actor is None:
            target_pose = [0, -0.18, 0.75 + self.table_z_bias, 0, 1, 0, 0]
        else:
            target_pose = self.last_actor.get_functional_point(1)

        # Compute place poses
        place_pre_pose = self.get_place_pose(block, arm_tag, target_pose, functional_point_id=0, pre_dis=0.05, pre_dis_axis="fp")
        place_pose = self.get_place_pose(block, arm_tag, target_pose, functional_point_id=0, pre_dis=0., pre_dis_axis="fp")

        # === SUB-TASK 4: Move to target ===
        funnel_used_transport = False
        teleport_pause_frames_transport = 0
        transport_start = self.robot.get_left_ee_pose() if arm_tag == "left" else self.robot.get_right_ee_pose()

        if self._should_use_funnel("move_to_target"):
            funnel_start = self._sample_funnel_start(place_pre_pose, "move_to_target", arm_tag)
            if funnel_start is not None:
                transport_start = funnel_start
                funnel_used_transport = True
                teleport_pause_frames_transport = self._teleport_arm_to_pose(arm_tag, funnel_start)

        if base_block is None:
            self.start_subtask("move_to_center", obj_name=block_name, arm_tag=str(arm_tag),
                              metadata={"funnel_used": funnel_used_transport, "teleport_pause_frames": teleport_pause_frames_transport})
            instruction = self._generate_instruction("move_to_center")
        else:
            self.start_subtask("move_above_target", obj_name=block_name, base_name=base_name, arm_tag=str(arm_tag),
                              metadata={"funnel_used": funnel_used_transport, "teleport_pause_frames": teleport_pause_frames_transport})
            instruction = self._generate_instruction("move_above_target", base_name=base_name)

        via_points = self._generate_via_point_trajectory(transport_start, place_pre_pose, arm_tag)
        self._move_through_via_points(arm_tag, via_points)

        self._create_boundary_pause()
        self.end_subtask(instruction)

        # === SUB-TASK 5: Place ===
        funnel_used_place = False
        teleport_pause_frames_place = 0
        place_start = self.robot.get_left_ee_pose() if arm_tag == "left" else self.robot.get_right_ee_pose()

        if self._should_use_funnel("place"):
            funnel_start = self._sample_funnel_start(place_pose, "place", arm_tag)
            if funnel_start is not None:
                place_start = funnel_start
                funnel_used_place = True
                teleport_pause_frames_place = self._teleport_arm_to_pose(arm_tag, funnel_start)

        if base_block is None:
            self.start_subtask("release", obj_name=block_name, arm_tag=str(arm_tag),
                              metadata={"funnel_used": funnel_used_place, "teleport_pause_frames": teleport_pause_frames_place})
            instruction = self._generate_instruction("release", obj_name=block_name)
        else:
            self.start_subtask("release_stack", obj_name=block_name, base_name=base_name, arm_tag=str(arm_tag),
                              metadata={"funnel_used": funnel_used_place, "teleport_pause_frames": teleport_pause_frames_place})
            instruction = self._generate_instruction("release_stack", obj_name=block_name, base_name=base_name)

        # Tighter perturbations for place
        original_sigma = self.via_point_config["position_sigma"]
        self.via_point_config["position_sigma"] = original_sigma * 0.5

        via_points = self._generate_via_point_trajectory(place_start, place_pose, arm_tag)
        self._move_through_via_points(arm_tag, via_points)

        self.via_point_config["position_sigma"] = original_sigma

        self.move(self.open_gripper(arm_tag, pos=1.0))
        self._create_boundary_pause()
        self.end_subtask(instruction)

        # === SUB-TASK 6: Retract ===
        self.start_subtask("retract", obj_name=block_name, arm_tag=str(arm_tag))
        instruction = self._generate_instruction("retract")

        # Record block positions before retract (for stability check)
        if base_block is not None:
            block_pos_before = block.get_pose().p.copy()
            base_pos_before = base_block.get_pose().p.copy()

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))
        self._create_boundary_pause()

        # Stability check: verify blocks weren't knocked during retract
        if base_block is not None and self.need_plan:
            # Let physics settle
            for _ in range(30):
                self.scene.step()

            block_pos_after = block.get_pose().p
            base_pos_after = base_block.get_pose().p

            # Check if blocks moved significantly (knocked over)
            block_drift = np.linalg.norm(block_pos_after[:2] - block_pos_before[:2])
            base_drift = np.linalg.norm(base_pos_after[:2] - base_pos_before[:2])
            height_drop = block_pos_before[2] - block_pos_after[2]

            if block_drift > 0.02 or base_drift > 0.02 or height_drop > 0.01:
                print(f"WARNING: Blocks moved during retract! block_drift={block_drift:.3f}, "
                      f"base_drift={base_drift:.3f}, height_drop={height_drop:.3f}")
                self.plan_success = False

        self.end_subtask(instruction)

        self.last_gripper = arm_tag
        self.last_actor = block
        return str(arm_tag)
