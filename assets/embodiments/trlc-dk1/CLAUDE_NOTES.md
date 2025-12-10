# TRLC-DK1 Integration Notes

## Current Status
The TRLC-DK1 robot embodiment has been added to RoboTwin but data collection fails with:
```
target_pose cannot be None for move action.
```

## Files Created
- `TRLC-DK1-Follower.urdf` - Robot description (6DoF arm + parallel gripper + wrist camera)
- `TRLC-DK1-Follower.srdf` - Semantic robot description
- `config.yml` - RoboTwin robot configuration
- `curobo_tmp.yml` - CuRobo config template (with `${ASSETS_PATH}`)
- `curobo.yml` - CuRobo config (generated with absolute paths)
- `collision_trlc-dk1.yml` - Collision spheres for motion planning
- `meshes/` - Visual and collision meshes

## What Works
1. URDF loading in SAPIEN - 8 active joints (6 arm + 2 gripper)
2. Base_Task environment import
3. CuRobo MotionGen initialization and warmup
4. Forward kinematics at home position works:
   - EE position: [0, 0.094, 0.16] (relative to base_link)
   - EE quaternion: [1, 0, 0, 0] (identity)

## Current Issue
CuRobo motion planning fails with:
```
RuntimeError: shape '[1, 6]' is invalid for input of size 8
```

This happens when `plan_single()` is called. The issue is that CuRobo expects 6 joints for planning (arm only) but receives 8 joints (arm + gripper).

The `lock_joints` config should handle this, but something is wrong.

## Configuration Details

### URDF Frame Conventions
- `link6-7` (EE frame): Y=forward (out of gripper), X=right, Z=up
- `tool0` frame: Z=forward, X=right, Y=down (offset 0.158m from link6-7)

### Current config.yml Settings
```yaml
move_group: ["link6-7", "link6-7"]
ee_joints: ["joint6", "joint6"]
arm_joints_name: [['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'], ...]
gripper_name:
  - base: "gripper_left"
    mimic: [["gripper_right", 1., 0.]]
gripper_bias: 0.158
gripper_scale: [-0.045, 0.0]
delta_matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
global_trans_matrix: [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
robot_pose: [[0, -0.35, 0.75, 0.707, 0, 0, 0.707]]
```

### curobo.yml Settings
```yaml
ee_link: "link6-7"
lock_joints: {"gripper_left": 0.0, "gripper_right": 0.0}
joint_names: ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper_left", "gripper_right"]
```

## Issue Found and Fixed

**Root Cause**: The `base_link` collision sphere was colliding with the table obstacle in CuRobo.

The table is positioned at Z=0.74m in world frame. The robot is mounted at Z=0.75m.
In robot base frame, the table is at Z=-0.01m (thickness 0.04m, so Z=-0.03 to Z=+0.01).

The original `base_link` sphere at `[0, 0, 0.03]` with radius `0.04` extended down to Z=-0.01,
which collided with the table top at Z=+0.01.

**Fix**: Changed `base_link` collision sphere to:
- center: `[0, 0, 0.04]` (moved up 1cm)
- radius: `0.02` (reduced from 0.04)

This ensures the sphere bottom is at Z=0.02, above the table top at Z=0.01.

## Task Config for Testing
Using `task_config/demo_randomized.yml` with:
```yaml
embodiment: [trlc-dk1, trlc-dk1, 0.7]  # Two instances, 0.7m apart
```

## Commands
```bash
# Run data collection
./collect_data.sh stack_blocks_two demo_randomized 0

# Run debug script
python script/debug_trlc_dk1.py

# Run basic test
python script/test_trlc_dk1.py
```
