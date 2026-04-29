# C11 — 任务级组合 / grasp / attach / manipulation

> arm + gripper + object + table composition: grasp failure, finger tilt, gripper collapse, attach pose error.

**Doc anchor:** `C11. 任务级组合 / grasp / attachment / manipulation 类`

**Members:** 45

## Suggested operators

- `attach_site_mutation`
- `mount_pose_mutation`
- `object_mass_size_friction_mutation`
- `reach_close_lift_seq`
- `gripper_force_sweep`
- `finger_actuator_sweep`
- `controller_switch`
- `table_floor_pose_mutation`
- `integrator_switch`
- `task_reset_mutation`

## Suggested oracles

- `grasp_success`
- `object_slip`
- `finger_symmetry`
- `gripper_collapse`
- `contact_stability`
- `unexpected_arm_motion`
- `task_metric_regression`

## Members

- `gym_robotics__gymnasium_robotics_envs_assets_fetch_pick_and_place`
- `gym_robotics__gymnasium_robotics_envs_assets_fetch_push`
- `gym_robotics__gymnasium_robotics_envs_assets_fetch_slide`
- `menagerie__aloha_aloha`
- `menagerie__aloha_scene`
- `menagerie__boston_dynamics_spot_spot_arm`
- `menagerie__franka_emika_panda_hand`
- `menagerie__franka_emika_panda_mjx_hand`
- `menagerie__franka_emika_panda_mjx_panda`
- `menagerie__franka_emika_panda_mjx_panda_nohand`
- `menagerie__franka_emika_panda_mjx_scene`
- `menagerie__franka_emika_panda_mjx_single_cube`
- `menagerie__franka_emika_panda_panda`
- `menagerie__franka_emika_panda_panda_nohand`
- `menagerie__franka_emika_panda_scene`
- `menagerie__kuka_iiwa_14_iiwa14`
- `menagerie__kuka_iiwa_14_scene`
- `menagerie__pal_tiago_dual_tiago_dual`
- `menagerie__pal_tiago_tiago`
- `menagerie__robotiq_2f85_2f85`
- `menagerie__robotiq_2f85_scene`
- `menagerie__robotiq_2f85_v4_2f85`
- `menagerie__robotiq_2f85_v4_mjx_2f85`
- `menagerie__robotiq_2f85_v4_scene`
- `menagerie__trossen_wxai_scene`
- `menagerie__trossen_wxai_trossen_ai_bimanual`
- `menagerie__ufactory_lite6_lite6`
- `menagerie__ufactory_lite6_lite6_gripper_narrow`
- `menagerie__ufactory_lite6_lite6_gripper_wide`
- `menagerie__ufactory_lite6_scene`
- `menagerie__ufactory_xarm7_hand`
- `menagerie__ufactory_xarm7_scene`
- `menagerie__ufactory_xarm7_xarm7`
- `menagerie__ufactory_xarm7_xarm7_nohand`
- `menagerie__universal_robots_ur10e_scene`
- `menagerie__universal_robots_ur10e_ur10e`
- `menagerie__universal_robots_ur5e_scene`
- `menagerie__universal_robots_ur5e_ur5e`
- `robosuite__robosuite_models_assets_grippers_panda_gripper`
- `robosuite__robosuite_models_assets_grippers_rethink_gripper`
- `robosuite__robosuite_models_assets_grippers_robotiq_gripper_140`
- `robosuite__robosuite_models_assets_grippers_robotiq_gripper_s`
- `robosuite__robosuite_models_assets_robots_spot_arm_robot`
- `robosuite__robosuite_models_assets_robots_tiago_robot`
- `robosuite__robosuite_models_assets_robots_ur5e_robot`
