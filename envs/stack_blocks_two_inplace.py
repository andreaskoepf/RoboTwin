from ._base_task import Base_Task
from .utils import *
import transforms3d as t3d
import math


class stack_blocks_two_inplace(Base_Task):
    """
    Stack two blocks task where the second block is stacked directly on the first
    block at its initial position (no center placement).

    Unlike stack_blocks_two, this task does NOT move the first block to the center.
    Instead, the first block's random initial position becomes the stacking target.

    This creates a more direct stacking task with variable target locations,
    useful for training models to handle diverse spatial goals.

    Features:
    - Configurable number of distractor blocks
    - Variable block sizes
    - Extended color palette for visual discrimination
    - Random robot start positions for robust training data generation
    - In-place stacking (target block stays at initial position)
    """

    # Reduce gripper padding to minimize zero-action frames in training data
    # Default is 0.5 (100 extra steps), reduced to 0.05 (10 extra steps)
    gripper_alpha = 0.05

    # Sub-task instruction templates for language generation
    SUBTASK_TEMPLATES = {
        # Phase 1: Approach the object
        "approach": [
            "Move to {obj}",
            "Approach {obj}",
            "Position gripper near {obj}",
            "Go to {obj}",
        ],
        # Phase 2: Grasp the object (close gripper)
        "grasp": [
            "Grasp {obj}",
            "Close gripper on {obj}",
            "Grip {obj}",
            "Grab {obj}",
        ],
        # Phase 3: Lift the object
        "lift": [
            "Lift {obj}",
            "Raise {obj}",
            "Pick up {obj}",
            "Lift the block up",
        ],
        # Phase 4: Move above target (stacking)
        "move_above_target": [
            "Move above {base} at stacking height",
            "Position over {base} ready to release",
            "Carry to {base} and lower to place height",
            "Align above {base} close enough to stack",
        ],
        # Phase 5: Release for stacking
        "release_stack": [
            "Release {obj} onto {base}",
            "Drop {obj} on {base}",
            "Let go to stack on {base}",
            "Open gripper above {base}",
        ],
        # Phase 6: Retract after placing
        "retract": [
            "Move away",
            "Retract gripper",
            "Pull back",
            "Clear the block",
        ],
        # Final: Return to home
        "return_home": [
            "Return to home position",
            "Go back to start",
            "Move to home pose",
            "Return to ready position",
        ],
        # Handover phases (when blocks are on opposite sides)
        "move_to_handover": [
            "Move {obj} to handover position",
            "Carry {obj} to handover zone",
            "Transport {obj} to middle for handover",
            "Bring {obj} to handover point",
        ],
        "handover_receive": [
            "Receive {obj} from {other_arm}",
            "Take {obj} at handover",
            "Grasp {obj} during handover",
            "Accept {obj} from {other_arm}",
        ],
        "handover_release": [
            "Release {obj} to {other_arm}",
            "Hand over {obj}",
            "Transfer {obj} to {other_arm}",
            "Let {other_arm} take {obj}",
        ],
        "handover_retract": [
            "Retract after handover",
            "Move away after transfer",
            "Clear handover zone",
            "Pull back from handover",
        ],
    }

    def _generate_instruction(self, subtask_type: str, obj_name: str = None,
                               base_name: str = None, other_arm: str = None) -> str:
        """
        Generate a natural language instruction for a sub-task.

        Args:
            subtask_type: Type of sub-task (key in SUBTASK_TEMPLATES)
            obj_name: Object name to substitute for {obj}
            base_name: Base object name to substitute for {base}
            other_arm: Other arm name to substitute for {other_arm}

        Returns:
            A randomly selected instruction with placeholders filled in.
        """
        templates = self.SUBTASK_TEMPLATES.get(subtask_type, ["Perform action"])
        template = templates[np.random.randint(len(templates))]

        instruction = template
        if obj_name and "{obj}" in instruction:
            instruction = instruction.replace("{obj}", f"the {obj_name}")
        if base_name and "{base}" in instruction:
            instruction = instruction.replace("{base}", f"the {base_name}")
        if other_arm and "{other_arm}" in instruction:
            instruction = instruction.replace("{other_arm}", f"{other_arm} arm")

        return instruction

    # ==================== Handover Methods ====================

    def _needs_handover(self, block, target_block):
        """
        Check if a handover is needed - when blocks are on opposite sides
        of the workspace (one requires left arm, other requires right arm).

        Args:
            block: The block to be picked up
            target_block: The block to stack on

        Returns:
            True if handover is needed, False otherwise
        """
        block_pos = block.get_pose().p
        target_pos = target_block.get_pose().p

        # Block determines which arm picks it up
        pickup_arm = "left" if block_pos[0] < 0 else "right"
        # Target determines which arm places it
        place_arm = "left" if target_pos[0] < 0 else "right"

        return pickup_arm != place_arm

    def _get_handover_poses(self, holding_arm_tag, block_half_size=0.025):
        """
        Get handover poses for both arms where grippers face each other with overlap.

        Based on validated poses from test_single_arm_orientation.py:
        - Left arm: X→+X (toward right), Z→up (identity quaternion)
        - Right arm: X→-X (toward left), Z→sideways (180° Z + 90° X roll)
        - Grippers overlap based on block size for secure transfer

        IMPORTANT: The planner applies gripper_bias offset internally. The poses must
        account for this by specifying positions offset by gripper_bias from the
        desired tip positions, in the direction opposite to gripper orientation.

        Args:
            holding_arm_tag: "left" or "right" - which arm is currently holding the block
            block_half_size: Half-size of the block being transferred (default 0.025m)

        Returns:
            Tuple of (holding_pose, receiving_pose) - both as [x, y, z, qw, qx, qy, qz]
            Poses are adjusted for gripper_bias so tips end up at intended positions.
        """
        from scipy.spatial.transform import Rotation

        # Handover center position
        base_x, base_y, base_z = 0.0, -0.15, 1.05

        # Add random variations within a radius for diverse training data
        random_radius = 0.05  # 3cm variation radius
        center_x = base_x + np.random.uniform(-random_radius, random_radius)
        center_y = base_y + np.random.uniform(-random_radius, random_radius)
        center_z = base_z + np.random.uniform(-random_radius, random_radius)

        # Gripper overlap - based on block width so both grippers can grip around it
        gripper_overlap = 2 * block_half_size  # block width

        # Get gripper bias values from robot config
        left_gripper_bias = self.robot.left_gripper_bias
        right_gripper_bias = self.robot.right_gripper_bias

        # Random X-axis rotation for variety in gripper orientation (±45°)
        random_x_roll = np.random.uniform(-45, 45)  # degrees

        # Left arm orientation: identity + random X roll
        r_left = Rotation.from_euler('X', random_x_roll, degrees=True)
        left_quat_scipy = r_left.as_quat()  # [x, y, z, w]
        left_quat = [left_quat_scipy[3], left_quat_scipy[0],
                     left_quat_scipy[1], left_quat_scipy[2]]  # [w, x, y, z]

        # Right arm orientation: 180° Z + (90° - random_roll) X roll
        # The 90° offset ensures grippers are perpendicular for secure handover
        # Negate random_x_roll because right arm's local X is inverted after Z flip
        r_flip = Rotation.from_euler('Z', 180, degrees=True)
        r_roll = Rotation.from_euler('X', 90 - random_x_roll, degrees=True)
        r_right = r_flip * r_roll
        right_quat_scipy = r_right.as_quat()  # [x, y, z, w]
        right_quat = [right_quat_scipy[3], right_quat_scipy[0],
                      right_quat_scipy[1], right_quat_scipy[2]]  # [w, x, y, z]

        # Desired TIP positions (where we want gripper tips to meet)
        # Left tip at center + overlap/2 (toward +X)
        # Right tip at center - overlap/2 (toward -X)
        left_tip_x = center_x + gripper_overlap / 2
        right_tip_x = center_x - gripper_overlap / 2

        # Adjust for gripper_bias: pose must be offset so tip ends up at desired position
        # For left arm (gripper X = +world X): pose_x = tip_x - gripper_bias
        # For right arm (gripper X = -world X): pose_x = tip_x + gripper_bias
        left_pose_x = left_tip_x - left_gripper_bias
        right_pose_x = right_tip_x + right_gripper_bias

        left_pose = [left_pose_x, center_y, center_z] + left_quat
        right_pose = [right_pose_x, center_y, center_z] + right_quat

        if holding_arm_tag == "left":
            return left_pose, right_pose
        else:
            return right_pose, left_pose

    def _compute_90deg_offset_quat(self, original_quat):
        """
        Compute a quaternion rotated 90 degrees around the Z-axis from the original.
        This creates the perpendicular gripper orientation for the receiving arm.

        Args:
            original_quat: Original quaternion [qw, qx, qy, qz] (transforms3d/wxyz convention)

        Returns:
            New quaternion [qw, qx, qy, qz] rotated 90 degrees around Z
        """
        # original_quat is already in t3d convention [qw, qx, qy, qz]
        quat_t3d = original_quat

        # 90 degree rotation around Z axis
        rot_90_z = t3d.euler.euler2quat(0, 0, math.pi / 2, 'sxyz')

        # Combine: new = rot_90_z * original
        new_quat_t3d = t3d.quaternions.qmult(rot_90_z, quat_t3d)

        # Return in t3d/wxyz convention
        return list(new_quat_t3d)

    def _perform_handover(self, block, pickup_arm, receive_arm, block_name, block_half_size):
        """
        Perform a handover maneuver: transfer block from pickup_arm to receive_arm.

        The grippers face each other directly with overlapping tips and a 90° roll
        offset between them for secure block transfer.

        The handover involves:
        1. Move block to handover position (pickup_arm with facing orientation)
        2. Receive arm approaches with opposite-facing 90° offset orientation
        3. Receive arm grasps (grippers overlap around block)
        4. Pickup arm releases
        5. Pickup arm retracts

        Args:
            block: The block being transferred
            pickup_arm: ArmTag of the arm currently holding the block
            receive_arm: ArmTag of the arm that will receive the block
            block_name: Name of the block for instructions
            block_half_size: Half-size of the block being transferred

        Returns:
            The receive_arm ArmTag (now holding the block)
        """
        # Debug: print current state
        print(f"\n[HANDOVER DEBUG] Starting handover: {pickup_arm} -> {receive_arm}")
        print(f"  Block position: {block.get_pose().p}")
        print(f"  Block half-size: {block_half_size}")
        print(f"  plan_success: {self.plan_success}")

        # Get handover poses for both arms (facing each other with overlap based on block size)
        holding_pose, receiving_pose = self._get_handover_poses(str(pickup_arm), block_half_size)
        print(f"  Holding pose ({pickup_arm}): pos={holding_pose[:3]}")
        print(f"  Receiving pose ({receive_arm}): pos={receiving_pose[:3]}")

        # Approach offset for receiving arm (to avoid collision during approach)
        approach_offset_dist = 0.15  # 15cm offset from handover center

        # Calculate approach pose for receiving arm only
        receiving_approach_pose = list(receiving_pose)
        if str(receive_arm) == "left":
            receiving_approach_pose[0] -= approach_offset_dist  # Left arm approaches from -X
        else:
            receiving_approach_pose[0] += approach_offset_dist  # Right arm approaches from +X

        print(f"  Receiving approach: pos={receiving_approach_pose[:3]}")

        # Retract position for holding arm (same offset, backing away after release)
        holding_retract_pose = list(holding_pose)
        if str(pickup_arm) == "left":
            holding_retract_pose[0] -= approach_offset_dist  # Left arm retracts toward -X
        else:
            holding_retract_pose[0] += approach_offset_dist  # Right arm retracts toward +X
        print(f"  Holding retract: pos={holding_retract_pose[:3]}")

        # === SUB-TASK: Move to handover position (simultaneous) ===
        self.start_subtask("move_to_handover", obj_name=block_name, arm_tag=str(pickup_arm),
                          metadata={"other_arm": str(receive_arm)})
        instruction = self._generate_instruction("move_to_handover", obj_name=block_name)

        # Simultaneous: holding arm to handover + receive arm to approach
        print(f"  [1] Simultaneous: {pickup_arm} to holding + {receive_arm} to approach... plan_success={self.plan_success}")
        receive_ee = self.robot.get_left_ee_pose() if str(receive_arm) == "left" else self.robot.get_right_ee_pose()
        print(f"       Current receive EE: {receive_ee[:3]}")
        print(f"       Target holding: {holding_pose[:3]}")
        print(f"       Target approach: {receiving_approach_pose[:3]}")
        if str(pickup_arm) == "left":
            self.together_move_to_pose(holding_pose, receiving_approach_pose)
        else:
            self.together_move_to_pose(receiving_approach_pose, holding_pose)
        print(f"      After simultaneous: plan_success={self.plan_success}, block z={block.get_pose().p[2]:.4f}")
        self.end_subtask(instruction)

        # === SUB-TASK: Receive at handover ===
        self.start_subtask("handover_receive", obj_name=block_name, arm_tag=str(receive_arm),
                          metadata={"other_arm": str(pickup_arm)})
        instruction = self._generate_instruction("handover_receive", obj_name=block_name,
                                                  other_arm=str(pickup_arm))

        # Move in to final overlap position
        print(f"  [5] Move receive arm to final... plan_success={self.plan_success}")
        self.move(self.move_to_pose(receive_arm, receiving_pose))
        receive_ee_final = self.robot.get_left_ee_pose() if str(receive_arm) == "left" else self.robot.get_right_ee_pose()
        holding_ee_final = self.robot.get_right_ee_pose() if str(receive_arm) == "left" else self.robot.get_left_ee_pose()
        print(f"      After receive final: plan_success={self.plan_success}")
        print(f"      Target receive pose: {receiving_pose[:3]}")
        print(f"      Actual receive EE: {receive_ee_final[:3]}")
        print(f"      Holding EE: {holding_ee_final[:3]}")
        print(f"      Block pos: {block.get_pose().p}")

        # Close gripper on block
        print(f"  [6] Close receive gripper... plan_success={self.plan_success}")
        self.move(self.close_gripper(receive_arm, pos=0.0))
        left_grip = self.robot.get_left_gripper_val()
        right_grip = self.robot.get_right_gripper_val()
        print(f"      After close: plan_success={self.plan_success}")
        print(f"      Block pos: {block.get_pose().p}")
        print(f"      Gripper vals: left={left_grip:.3f}, right={right_grip:.3f}")
        self.end_subtask(instruction)

        # === SUB-TASK: Release from pickup arm ===
        self.start_subtask("handover_release", obj_name=block_name, arm_tag=str(pickup_arm),
                          metadata={"other_arm": str(receive_arm)})
        instruction = self._generate_instruction("handover_release", obj_name=block_name,
                                                  other_arm=str(receive_arm))

        # Open pickup arm gripper
        print(f"  [7] Open pickup gripper... plan_success={self.plan_success}")
        self.move(self.open_gripper(pickup_arm, pos=1.0))
        left_grip = self.robot.get_left_gripper_val()
        right_grip = self.robot.get_right_gripper_val()
        print(f"      After open: plan_success={self.plan_success}")
        print(f"      Block pos: {block.get_pose().p}")
        print(f"      Gripper vals: left={left_grip:.3f}, right={right_grip:.3f}")
        self.end_subtask(instruction)

        # === SUB-TASK: Retract pickup arm ===
        self.start_subtask("handover_retract", arm_tag=str(pickup_arm))
        instruction = self._generate_instruction("handover_retract")

        # Back off to safe retract position (away from receive arm and block)
        # Going home will be done simultaneously with receive arm's pre-place move
        print(f"  [8] Retract pickup arm to safe position... plan_success={self.plan_success}")
        self.move(self.move_to_pose(pickup_arm, holding_retract_pose))
        print(f"      After retract: plan_success={self.plan_success}")
        print(f"      Block pos: {block.get_pose().p}")
        self.end_subtask(instruction)

        print(f"[HANDOVER DEBUG] Handover complete. Block pos: {block.get_pose().p}")
        # Return both arms so caller can do simultaneous home + pre-place
        return receive_arm, pickup_arm

    def setup_demo(self, **kwags):
        # Extract task variation config from domain_randomization or use defaults
        variation_config = kwags.get("domain_randomization", {}).get("block_variation", {})

        # Number of distractor blocks (0 = no distractors, original behavior)
        self.num_distractors = variation_config.get("num_distractors", 0)

        # Size variation: [min_scale, max_scale] relative to base size (0.025)
        self.size_variation = variation_config.get("size_variation", [1.0, 1.0])

        # Whether to randomize target block colors (vs fixed red/green)
        self.randomize_colors = variation_config.get("randomize_colors", False)

        # Random start pose configuration
        random_start_config = kwags.get("domain_randomization", {}).get("random_start_pose", {})
        self.random_start_probability = random_start_config.get("probability", 0.0)
        self.random_start_height_range = random_start_config.get("height_range", [0.08, 0.25])
        self.random_start_offset_range = random_start_config.get("offset_range", [0.10, 0.25])

        super()._init_task_env_(**kwags)
        # Enable sub-task annotations (uses base class infrastructure)
        self.enable_subtask_annotations()

    def load_actors(self):
        base_half_size = 0.025
        total_blocks = 2 + self.num_distractors

        # Select colors for all blocks
        color_names = list(self.COLOR_PALETTE.keys())

        if self.randomize_colors:
            selected_colors = list(np.random.choice(color_names, min(total_blocks, len(color_names)), replace=False))
            while len(selected_colors) < total_blocks:
                selected_colors.append(color_names[np.random.randint(len(color_names))])
        else:
            selected_colors = ["red", "green"]
            remaining_colors = [c for c in color_names if c not in selected_colors]
            for _ in range(self.num_distractors):
                if remaining_colors:
                    idx = np.random.randint(len(remaining_colors))
                    color = remaining_colors[idx]
                    selected_colors.append(color)
                    remaining_colors.remove(color)
                else:
                    non_target_colors = [c for c in color_names if c not in ["red", "green"]]
                    selected_colors.append(non_target_colors[np.random.randint(len(non_target_colors))])

        self.block_color_names = selected_colors

        # Generate sizes for all blocks
        block_sizes = []
        for _ in range(total_blocks):
            scale = np.random.uniform(self.size_variation[0], self.size_variation[1])
            block_sizes.append(base_half_size * scale)

        # Generate non-overlapping poses for all blocks
        block_pose_lst = []
        block_size_lst = []

        # For inplace stacking, we don't need a target center exclusion zone
        # But we still need gripper clearance between blocks
        gripper_clearance = 0.05

        def check_block_pose(new_pose, new_size, is_distractor=False):
            """Check if new block overlaps with existing blocks."""
            for j, (existing_pose, existing_size) in enumerate(zip(block_pose_lst, block_size_lst)):
                min_dist = new_size + existing_size + 0.02

                # Distractors need extra clearance from target blocks (first two)
                if is_distractor and j < 2:
                    min_dist += gripper_clearance

                if np.sum(pow(new_pose.p[:2] - existing_pose.p[:2], 2)) < min_dist ** 2:
                    return False

            return True

        for i in range(total_blocks):
            block_half_size = block_sizes[i]
            max_attempts = 100
            attempts = 0
            is_distractor = (i >= 2)

            block_pose = rand_pose(
                xlim=[-0.28, 0.28],
                ylim=[-0.20, 0.12],
                zlim=[0.741 + block_half_size],
                qpos=[1, 0, 0, 0],
                ylim_prop=True,
                rotate_rand=True,
                rotate_lim=[0, 0, 0.75],
            )

            def is_valid_pose(pose):
                return (
                    abs(pose.p[0]) >= 0.05 and  # Keep center strip clear for arm movement
                    check_block_pose(pose, block_half_size, is_distractor)
                )

            while attempts < max_attempts and not is_valid_pose(block_pose):
                block_pose = rand_pose(
                    xlim=[-0.28, 0.28],
                    ylim=[-0.20, 0.12],
                    zlim=[0.741 + block_half_size],
                    qpos=[1, 0, 0, 0],
                    ylim_prop=True,
                    rotate_rand=True,
                    rotate_lim=[0, 0, 0.75],
                )
                attempts += 1

            if is_valid_pose(block_pose):
                block_pose_lst.append(deepcopy(block_pose))
                block_size_lst.append(block_half_size)
            elif is_distractor:
                print(f"Warning: Could not place distractor {i-1}, skipping")
                block_sizes[i] = None
            else:
                raise RuntimeError(f"Could not find valid position for target block {i}")

        def create_block(block_pose, color_rgb, half_size, name="box"):
            return create_box(
                scene=self,
                pose=block_pose,
                half_size=(half_size, half_size, half_size),
                color=color_rgb,
                name=name,
            )

        # Create target blocks (first two)
        color1_rgb = self.COLOR_PALETTE[selected_colors[0]]
        color2_rgb = self.COLOR_PALETTE[selected_colors[1]]

        self.block1 = create_block(block_pose_lst[0], color1_rgb, block_size_lst[0], "target_block_1")
        self.block2 = create_block(block_pose_lst[1], color2_rgb, block_size_lst[1], "target_block_2")

        self.add_prohibit_area(self.block1, padding=0.07)
        self.add_prohibit_area(self.block2, padding=0.07)

        # Add blocks to size_dict for cluttered object spacing
        block1_pos = block_pose_lst[0].p
        block2_pos = block_pose_lst[1].p
        self.size_dict.append([block1_pos[0], block1_pos[1], block1_pos[2], block_size_lst[0] + 0.03])
        self.size_dict.append([block2_pos[0], block2_pos[1], block2_pos[2], block_size_lst[1] + 0.03])

        # Create distractor blocks
        self.distractor_blocks = []
        num_placed_distractors = len(block_pose_lst) - 2
        color_idx = 2
        for i in range(num_placed_distractors):
            pose_idx = i + 2
            color_rgb = self.COLOR_PALETTE[selected_colors[color_idx]]
            distractor = create_block(
                block_pose_lst[pose_idx], color_rgb, block_size_lst[pose_idx],
                f"distractor_block_{i}"
            )
            self.distractor_blocks.append(distractor)
            self.add_prohibit_area(distractor, padding=0.05)

            dist_pos = block_pose_lst[pose_idx].p
            self.size_dict.append([dist_pos[0], dist_pos[1], dist_pos[2], block_size_lst[pose_idx] + 0.03])
            color_idx += 1

        self.num_distractors = num_placed_distractors

        # Store block1's initial position as the stacking target
        # This is the key difference from stack_blocks_two!
        self.block1_initial_pose = block_pose_lst[0]

        # Store size information
        self.block1_size = block_size_lst[0]
        self.block2_size = block_size_lst[1]

    def play_once(self):
        # Initialize tracking variables
        self.last_gripper = None

        # Define block names for annotations using dynamic colors
        block1_color = self.block_color_names[0]
        block2_color = self.block_color_names[1]
        self.block_names = {
            id(self.block1): f"{block1_color} block",
            id(self.block2): f"{block2_color} block",
        }

        # Random start position logic for the stacking arm
        self._random_start_used = False
        self._random_start_pose = None

        if self.random_start_probability > 0 and np.random.random() < self.random_start_probability:
            # Determine which arm will be used for block2 (the block to move)
            block2_pose = self.block2.get_pose().p
            stacking_arm_tag = "left" if block2_pose[0] < 0 else "right"

            joint_config, ee_pose = self._generate_random_start_config(
                self.block2, stacking_arm_tag
            )

            if joint_config is not None:
                self._set_arm_to_config(stacking_arm_tag, joint_config)
                self._random_start_used = True
                self._random_start_pose = ee_pose

        # In-place stacking: only move block2 onto block1 (block1 stays in place!)
        arm_tag = self.stack_block_on_target(self.block2, self.block1)

        # Return arm to home position
        self.start_subtask("return_home", arm_tag=str(arm_tag))
        instruction = self._generate_instruction("return_home")
        self.move(self.back_to_origin(arm_tag=ArmTag(arm_tag)))
        self.end_subtask(instruction)

        # Store information about the blocks and their associated arms
        self.info["info"] = {
            "{A}": f"{block1_color} block",
            "{B}": f"{block2_color} block",
            "{a}": arm_tag,  # Only one arm used in this task
        }

        # Store additional scene information
        self.info["block_variation"] = {
            "block1_color": block1_color,
            "block2_color": block2_color,
            "block1_size": self.block1_size,
            "block2_size": self.block2_size,
            "num_distractors": self.num_distractors,
            "distractor_colors": self.block_color_names[2:] if self.num_distractors > 0 else [],
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

        # Save sub-task annotations
        self.save_subtask_annotations()

        return self.info

    def stack_block_on_target(self, block: Actor, target_block: Actor):
        """
        Pick up a block and stack it on top of the target block at its current position.

        This method handles both same-side stacking and cross-side stacking with handover.
        When the block and target are on opposite sides of the workspace, a handover
        maneuver is performed between the two arms with 90° gripper rotation offset.

        Sub-task decomposition (no handover):
        1. approach - move gripper to block
        2. grasp - close gripper
        3. lift - raise block
        4. move_above_target - transport above target block
        5. release_stack - open gripper to place
        6. retract - move away

        Sub-task decomposition (with handover):
        1. approach - move gripper to block
        2. grasp - close gripper
        3. lift - raise block
        4. move_to_handover - carry to handover zone
        5. handover_receive - second arm grasps with 90° offset
        6. handover_release - first arm releases
        7. handover_retract - first arm moves away
        8. move_above_target - second arm moves above target
        9. release_stack - open gripper to place
        10. retract - move away

        Args:
            block: The block to pick up
            target_block: The block to stack on top of

        Returns:
            The arm tag used for final placement ("left" or "right")
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

        # Get block names for instructions
        block_name = self.block_names.get(id(block), "block")
        target_name = self.block_names.get(id(target_block), "target block")

        # Store handover info for metadata
        self._handover_used = needs_handover

        # Compute grasp poses
        pre_grasp_pose, grasp_pose = self.choose_grasp_pose(
            block, arm_tag=arm_tag, pre_dis=0.14, target_dis=0  # 14cm pre-grasp height
        )

        # === SUB-TASK 1: Approach ===
        self.start_subtask("approach", obj_name=block_name, arm_tag=str(arm_tag))
        instruction = self._generate_instruction("approach", obj_name=block_name)

        # Move to pre-grasp position (let motion planner find path)
        self.move(self.move_to_pose(arm_tag, pre_grasp_pose))

        # Move to grasp pose (final approach)
        if pre_grasp_pose != grasp_pose:
            self.move((arm_tag, [Action(arm_tag, "move", target_pose=grasp_pose,
                                        constraint_pose=[1, 1, 1, 0, 0, 0])]))
        self.end_subtask(instruction)

        # === SUB-TASK 2: Grasp ===
        self.start_subtask("grasp", obj_name=block_name, arm_tag=str(arm_tag))
        instruction = self._generate_instruction("grasp", obj_name=block_name)
        self.move(self.close_gripper(arm_tag, pos=0.0))
        self.end_subtask(instruction)

        # === SUB-TASK 3: Lift ===
        self.start_subtask("lift", obj_name=block_name, arm_tag=str(arm_tag))
        instruction = self._generate_instruction("lift", obj_name=block_name)
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))
        self.end_subtask(instruction)

        # === HANDOVER (if needed) ===
        if needs_handover:
            # Perform handover: transfer from pickup_arm to place_arm
            # Returns both arms so we can do simultaneous home + pre-place
            arm_tag, giver_arm = self._perform_handover(block, pickup_arm, place_arm, block_name, self.block2_size)
            print(f"\n[POST-HANDOVER] arm_tag={arm_tag}, giver_arm={giver_arm}, plan_success={self.plan_success}")
            print(f"  Block position: {block.get_pose().p}")

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

            print(f"\n[PLACEMENT - Post Handover]")
            print(f"  Target block pos: {target_block_pos}")
            print(f"  Stacking height (target_z): {target_z:.4f}, gripper_bias: {gripper_bias:.4f}")
            print(f"  Current EE: {current_ee[:3]}, quat={current_ee[3:]}")
            print(f"  Pre-place EE z={approach_height:.3f}m (tip at {approach_height - gripper_bias:.3f}m)")
            print(f"  Place EE z={place_height:.3f}m (tip at {place_height - gripper_bias:.3f}m = target_z)")
            print(f"  plan_success before move: {self.plan_success}")

            # === SUB-TASK: Move above target (simultaneous with giver going home) ===
            self.start_subtask("move_above_target", obj_name=block_name,
                              base_name=target_name, arm_tag=str(arm_tag))
            instruction = self._generate_instruction("move_above_target", base_name=target_name)

            # Simultaneous move: giver arm to home + receive arm to pre-place
            print(f"  [1] Simultaneous: {giver_arm} to home + {arm_tag} to pre-place...")
            if str(arm_tag) == "left":
                self.together_move_to_pose(pre_place_pose, giver_home_pose)
            else:
                self.together_move_to_pose(giver_home_pose, pre_place_pose)
            print(f"      After simultaneous move: plan_success={self.plan_success}, block z={block.get_pose().p[2]:.4f}")

            # Lower to place
            print(f"  [2] Lowering to place...")
            self.move(self.move_to_pose(arm_tag, place_pose))
            print(f"      After place: plan_success={self.plan_success}, block z={block.get_pose().p[2]:.4f}")
            self.end_subtask(instruction)
        else:
            # Normal placement flow (no handover)
            target_pose = target_block.get_functional_point(1)
            print(f"\n[PLACEMENT] Target functional point: {target_pose}")

            place_pre_pose = self.get_place_pose(
                block, arm_tag, target_pose,
                functional_point_id=0, pre_dis=0.05, pre_dis_axis="fp"
            )
            place_pose = self.get_place_pose(
                block, arm_tag, target_pose,
                functional_point_id=0, pre_dis=0., pre_dis_axis="fp"
            )
            print(f"  place_pre_pose: {place_pre_pose[:3] if place_pre_pose else 'None'}")
            print(f"  place_pose: {place_pose[:3] if place_pose else 'None'}")
            print(f"  plan_success before move: {self.plan_success}")

            # === SUB-TASK: Move above target ===
            self.start_subtask("move_above_target", obj_name=block_name,
                              base_name=target_name, arm_tag=str(arm_tag))
            instruction = self._generate_instruction("move_above_target", base_name=target_name)

            print(f"  Moving to place_pre_pose...")
            self.move(self.move_to_pose(arm_tag, place_pre_pose))
            print(f"  After place_pre: plan_success={self.plan_success}, block z={block.get_pose().p[2]:.4f}")

            print(f"  Moving to place_pose...")
            self.move(self.move_to_pose(arm_tag, place_pose))
            print(f"  After place: plan_success={self.plan_success}, block z={block.get_pose().p[2]:.4f}")
            self.end_subtask(instruction)

        # === SUB-TASK: Release (stack) ===
        self.start_subtask("release_stack", obj_name=block_name,
                          base_name=target_name, arm_tag=str(arm_tag))
        instruction = self._generate_instruction("release_stack", obj_name=block_name,
                                                  base_name=target_name)
        self.move(self.open_gripper(arm_tag, pos=1.0))
        self.end_subtask(instruction)

        # === SUB-TASK: Retract ===
        self.start_subtask("retract", obj_name=block_name, arm_tag=str(arm_tag))
        instruction = self._generate_instruction("retract")
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))
        self.end_subtask(instruction)

        self.last_gripper = arm_tag
        return str(arm_tag)

    def check_success(self):
        block1_pose = self.block1.get_pose().p
        block2_pose = self.block2.get_pose().p

        # Calculate expected z-offset based on actual block sizes
        expected_z_offset = self.block1_size + self.block2_size

        # Tolerance scales with block size
        min_size = min(self.block1_size, self.block2_size)
        eps = [min_size, min_size, min_size * 0.5]

        expected_pos = np.array([block1_pose[0], block1_pose[1], block1_pose[2] + expected_z_offset])

        return (np.all(abs(block2_pose - expected_pos) < eps)
                and self.is_left_gripper_open() and self.is_right_gripper_open())
