"""
Stack Blocks Two - Perturbed Trajectory Variant

This task variant extends the standard stack_blocks_two task by adding artificial
perturbations to the motion trajectories. This creates training data that includes
"mistakes" and correction movements, which helps policies learn to recover from
imperfect intermediate positions during real-world execution.

Perturbation types:
1. Pre-grasp approach perturbation: Robot first moves to a slightly wrong position,
   then corrects to the proper pre-grasp pose before grasping.
2. Pre-place approach perturbation: Robot first moves to a slightly wrong position,
   then corrects to the proper pre-place pose before placing.

This addresses the distribution mismatch between perfect expert demonstrations
and imperfect policy rollouts during evaluation.

IMPLEMENTATION NOTE:
The perturbations are integrated into the planning phase. The number of perturbation
waypoints is saved alongside the trajectory data so that replay phase can generate
matching action counts.
"""

from ._base_task import Base_Task
from .utils import *
import sapien
import math
import transforms3d as t3d
import os
import json


class stack_blocks_two_perturbed(Base_Task):
    """
    Stack two blocks task with trajectory perturbations for more robust training data.
    """

    def setup_demo(self, **kwags):
        # Extract perturbation config from domain_randomization or use defaults
        perturbation_config = kwags.get("domain_randomization", {}).get("trajectory_perturbation", {})

        # Perturbation settings
        self.perturbation_enabled = perturbation_config.get("enabled", True)
        self.perturbation_probability = perturbation_config.get("probability", 0.5)
        self.position_perturbation_magnitude = perturbation_config.get("position_magnitude", 0.03)  # 3cm
        self.rotation_perturbation_magnitude = perturbation_config.get("rotation_magnitude", 0.1)  # ~6 degrees
        self.perturbation_type = perturbation_config.get("type", "both")  # "grasp", "place", or "both"

        # Allow multiple correction steps for more complex recovery trajectories
        self.max_correction_steps = perturbation_config.get("max_correction_steps", 1)

        # Track perturbation decisions for this episode
        self._perturbation_counts = []
        self._perturbation_idx = 0

        super()._init_task_env_(**kwags)

    def load_actors(self):
        block_half_size = 0.025
        block_pose_lst = []
        for i in range(2):
            block_pose = rand_pose(
                xlim=[-0.28, 0.28],
                ylim=[-0.08, 0.05],
                zlim=[0.741 + block_half_size],
                qpos=[1, 0, 0, 0],
                ylim_prop=True,
                rotate_rand=True,
                rotate_lim=[0, 0, 0.75],
            )

            def check_block_pose(block_pose):
                for j in range(len(block_pose_lst)):
                    if (np.sum(pow(block_pose.p[:2] - block_pose_lst[j].p[:2], 2)) < 0.01):
                        return False
                return True

            while (abs(block_pose.p[0]) < 0.05 or np.sum(pow(block_pose.p[:2] - np.array([0, -0.1]), 2)) < 0.0225
                   or not check_block_pose(block_pose)):
                block_pose = rand_pose(
                    xlim=[-0.28, 0.28],
                    ylim=[-0.08, 0.05],
                    zlim=[0.741 + block_half_size],
                    qpos=[1, 0, 0, 0],
                    ylim_prop=True,
                    rotate_rand=True,
                    rotate_lim=[0, 0, 0.75],
                )
            block_pose_lst.append(deepcopy(block_pose))

        def create_block(block_pose, color):
            return create_box(
                scene=self,
                pose=block_pose,
                half_size=(block_half_size, block_half_size, block_half_size),
                color=color,
                name="box",
            )

        self.block1 = create_block(block_pose_lst[0], (1, 0, 0))
        self.block2 = create_block(block_pose_lst[1], (0, 1, 0))
        self.add_prohibit_area(self.block1, padding=0.07)
        self.add_prohibit_area(self.block2, padding=0.07)
        target_pose = [-0.04, -0.13, 0.04, -0.05]
        self.prohibited_area.append(target_pose)
        self.block1_target_pose = [0, -0.13, 0.75 + self.table_z_bias, 0, 1, 0, 0]

    def play_once(self):
        # Initialize tracking variables for gripper and actor
        self.last_gripper = None
        self.last_actor = None

        # Reset perturbation tracking
        self._perturbation_counts = []
        self._perturbation_idx = 0

        # Pick and place the first block (block1) and get its arm tag
        arm_tag1 = self.pick_and_place_block(self.block1)
        # Pick and place the second block (block2) and get its arm tag
        arm_tag2 = self.pick_and_place_block(self.block2)

        # Store information about the blocks and their associated arms
        self.info["info"] = {
            "{A}": "red block",
            "{B}": "green block",
            "{a}": arm_tag1,
            "{b}": arm_tag2,
        }

        # Save perturbation counts to trajectory data for replay
        if self.need_plan:
            self._save_perturbation_counts()

        return self.info

    def _save_perturbation_counts(self):
        """Save perturbation counts to a file for replay phase."""
        save_path = getattr(self, 'save_dir', './data')
        perturb_file = os.path.join(save_path, "_traj_data", f"perturb_{self.ep_num}.json")
        os.makedirs(os.path.dirname(perturb_file), exist_ok=True)
        with open(perturb_file, 'w') as f:
            json.dump({"counts": self._perturbation_counts}, f)

    def _load_perturbation_counts(self):
        """Load perturbation counts from file for replay phase."""
        save_path = getattr(self, 'save_dir', './data')
        perturb_file = os.path.join(save_path, "_traj_data", f"perturb_{self.ep_num}.json")
        if os.path.exists(perturb_file):
            with open(perturb_file, 'r') as f:
                data = json.load(f)
                self._perturbation_counts = data.get("counts", [])
        else:
            self._perturbation_counts = []
        self._perturbation_idx = 0

    def _get_next_perturbation_count(self):
        """Get the next perturbation count (either generate or retrieve)."""
        if self.need_plan:
            # Planning phase: decide and store
            if self._should_perturb_now():
                count = np.random.randint(1, self.max_correction_steps + 1)
            else:
                count = 0
            self._perturbation_counts.append(count)
            return count
        else:
            # Replay phase: retrieve from stored
            if self._perturbation_idx < len(self._perturbation_counts):
                count = self._perturbation_counts[self._perturbation_idx]
                self._perturbation_idx += 1
                return count
            return 0

    def _generate_perturbed_pose(self, original_pose: list, perturbation_scale: float = 1.0) -> list:
        """
        Generate a perturbed version of the given pose.
        """
        perturbed_pose = deepcopy(original_pose)
        pos_mag = self.position_perturbation_magnitude * perturbation_scale
        rot_mag = self.rotation_perturbation_magnitude * perturbation_scale

        # Add position perturbation (x, y, z)
        perturbed_pose[0] += np.random.uniform(-pos_mag, pos_mag)
        perturbed_pose[1] += np.random.uniform(-pos_mag, pos_mag)
        # Smaller z perturbation to avoid table collision
        perturbed_pose[2] += np.random.uniform(-pos_mag * 0.3, pos_mag * 0.5)

        # Add rotation perturbation
        if rot_mag > 0:
            axis = np.random.randn(3)
            axis_norm = np.linalg.norm(axis)
            if axis_norm < 1e-6:
                axis = np.array([0, 0, 1])
            else:
                axis = axis / axis_norm
            angle = np.random.uniform(-rot_mag, rot_mag)

            delta_quat = t3d.quaternions.axangle2quat(axis, angle)
            original_quat = np.array(perturbed_pose[3:7])
            new_quat = t3d.quaternions.qmult(delta_quat, original_quat)

            quat_norm = np.linalg.norm(new_quat)
            if quat_norm > 1e-6:
                new_quat = new_quat / quat_norm
            else:
                new_quat = original_quat
            perturbed_pose[3:7] = new_quat.tolist()

        return perturbed_pose

    def _should_perturb_now(self) -> bool:
        """Determine whether to add perturbation based on settings."""
        if not self.perturbation_enabled:
            return False
        return np.random.rand() < self.perturbation_probability

    def pick_and_place_block(self, block: Actor):
        """
        Pick up a block and place it, with optional trajectory perturbations.
        """
        block_pose = block.get_pose().p
        arm_tag = ArmTag("left" if block_pose[0] < 0 else "right")

        # Load perturbation counts if in replay mode
        if not self.need_plan and self._perturbation_idx == 0:
            self._load_perturbation_counts()

        # === GRASP PHASE ===
        grasp_perturb_count = self._get_next_perturbation_count()

        if self.last_gripper is not None and (self.last_gripper != arm_tag):
            # Need to move opposite arm back
            if grasp_perturb_count > 0:
                # Add perturbation moves before the grasp
                self._execute_perturbed_grasp(block, arm_tag, grasp_perturb_count,
                                               with_opposite_arm=True)
            else:
                self.move(
                    self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09),
                    self.back_to_origin(arm_tag=arm_tag.opposite),
                )
        else:
            if grasp_perturb_count > 0:
                self._execute_perturbed_grasp(block, arm_tag, grasp_perturb_count,
                                               with_opposite_arm=False)
            else:
                self.move(self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.09))

        # Lift
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))

        # Determine target pose
        if self.last_actor is None:
            target_pose = [0, -0.13, 0.75 + self.table_z_bias, 0, 1, 0, 0]
        else:
            target_pose = self.last_actor.get_functional_point(1)

        # === PLACE PHASE ===
        place_perturb_count = self._get_next_perturbation_count()

        if place_perturb_count > 0:
            self._execute_perturbed_place(block, target_pose, arm_tag, place_perturb_count)
        else:
            self.move(
                self.place_actor(
                    block,
                    target_pose=target_pose,
                    arm_tag=arm_tag,
                    functional_point_id=0,
                    pre_dis=0.05,
                    dis=0.,
                    pre_dis_axis="fp",
                ))

        # Lift after placing
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07))

        self.last_gripper = arm_tag
        self.last_actor = block
        return str(arm_tag)

    def _execute_perturbed_grasp(self, block: Actor, arm_tag: ArmTag, num_perturbations: int,
                                  with_opposite_arm: bool = False):
        """Execute a grasp with perturbation waypoints."""
        dummy_pose = [0, 0, 0, 0, 0, 0, 0]

        if self.need_plan:
            # Planning phase: compute actual poses
            pre_grasp_pose, grasp_pose = self.choose_grasp_pose(
                block,
                arm_tag=arm_tag,
                pre_dis=0.09,
                target_dis=0,
            )

            # Generate perturbed waypoints and move through them
            for i in range(num_perturbations):
                scale = 1.0 - (i * 0.3)
                perturbed_pose = self._generate_perturbed_pose(pre_grasp_pose, scale)

                if i == 0 and with_opposite_arm:
                    self.move(
                        self.move_to_pose(arm_tag, perturbed_pose),
                        self.back_to_origin(arm_tag=arm_tag.opposite),
                    )
                else:
                    self.move(self.move_to_pose(arm_tag, perturbed_pose))

            # Now correct to proper pre-grasp and grasp
            self.move(self.move_to_pose(arm_tag, pre_grasp_pose))

            if pre_grasp_pose != grasp_pose:
                self.move((arm_tag, [Action(arm_tag, "move", target_pose=grasp_pose,
                                            constraint_pose=[1, 1, 1, 0, 0, 0])]))
        else:
            # Replay phase: use dummy poses, actual trajectories come from saved paths
            for i in range(num_perturbations):
                if i == 0 and with_opposite_arm:
                    self.move(
                        self.move_to_pose(arm_tag, dummy_pose),
                        self.back_to_origin(arm_tag=arm_tag.opposite),
                    )
                else:
                    self.move(self.move_to_pose(arm_tag, dummy_pose))

            # Correct to pre-grasp
            self.move(self.move_to_pose(arm_tag, dummy_pose))
            # Move to grasp pose
            self.move((arm_tag, [Action(arm_tag, "move", target_pose=dummy_pose,
                                        constraint_pose=[1, 1, 1, 0, 0, 0])]))

        # Close gripper
        self.move(self.close_gripper(arm_tag, pos=0.0))

    def _execute_perturbed_place(self, block: Actor, target_pose, arm_tag: ArmTag,
                                  num_perturbations: int):
        """Execute a place with perturbation waypoints."""
        dummy_pose = [0, 0, 0, 0, 0, 0, 0]

        if self.need_plan:
            # Planning phase: compute actual poses
            place_pre_pose = self.get_place_pose(
                block,
                arm_tag,
                target_pose,
                functional_point_id=0,
                pre_dis=0.05,
                pre_dis_axis="fp",
            )
            place_pose = self.get_place_pose(
                block,
                arm_tag,
                target_pose,
                functional_point_id=0,
                pre_dis=0.,
                pre_dis_axis="fp",
            )

            # Generate perturbed waypoints and move through them
            for i in range(num_perturbations):
                scale = 1.0 - (i * 0.3)
                perturbed_pose = self._generate_perturbed_pose(place_pre_pose, scale)
                self.move(self.move_to_pose(arm_tag, perturbed_pose))

            # Now correct to proper pre-place and place
            self.move(self.move_to_pose(arm_tag, place_pre_pose))
            self.move(self.move_to_pose(arm_tag, place_pose))
        else:
            # Replay phase: use dummy poses
            for i in range(num_perturbations):
                self.move(self.move_to_pose(arm_tag, dummy_pose))

            # Correct to pre-place and place
            self.move(self.move_to_pose(arm_tag, dummy_pose))
            self.move(self.move_to_pose(arm_tag, dummy_pose))

        # Open gripper
        self.move(self.open_gripper(arm_tag, pos=1.0))

    def check_success(self):
        block1_pose = self.block1.get_pose().p
        block2_pose = self.block2.get_pose().p
        eps = [0.025, 0.025, 0.012]

        return (np.all(abs(block2_pose - np.array(block1_pose[:2].tolist() + [block1_pose[2] + 0.05])) < eps)
                and self.is_left_gripper_open() and self.is_right_gripper_open())
