# C4 — 碰撞 / 接触几何 / 穿透 / contact force

> Collision geom, contact list, penetration depth, contact force, contact-sensor backend inconsistencies.

**Doc anchor:** `C4. 碰撞 / 接触几何 / 穿透 / contact force 类`

**Members:** 40

## Suggested operators

- `collision_geom_mutation`
- `visual_collision_aabb_compare`
- `floor_table_object_pose_mutation`
- `contact_margin_friction_solref_solimp_mutation`
- `gravity_toggle`
- `contact_rich_state_reset`
- `mjx_classic_diff`
- `cylinder_plane_boundary_sweep`

## Suggested oracles

- `unexpected_initial_contact`
- `penetration_depth`
- `contact_pair_consistency`
- `contact_force_magnitude`
- `floor_tunneling`
- `self_intersection`
- `mjx_vs_classic_contact_diff`
- `contact_sensor_consistency`

## Members

- `dm_control__dm_control_locomotion_walkers_assets_humanoid_CMU_V2019`
- `dm_control__dm_control_locomotion_walkers_assets_humanoid_CMU_V2020`
- `dm_control__dm_control_suite_humanoid`
- `dm_control__dm_control_suite_humanoid_CMU`
- `dm_control__dm_control_suite_quadruped`
- `gym_robotics__gymnasium_robotics_envs_assets_fetch_pick_and_place`
- `gym_robotics__gymnasium_robotics_envs_assets_fetch_push`
- `gym_robotics__gymnasium_robotics_envs_assets_hand_manipulate_block`
- `gym_robotics__gymnasium_robotics_envs_assets_hand_manipulate_block_touch_sensors`
- `gym_robotics__gymnasium_robotics_envs_assets_hand_manipulate_egg`
- `gym_robotics__gymnasium_robotics_envs_assets_hand_manipulate_egg_touch_sensors`
- `gym_robotics__gymnasium_robotics_envs_assets_hand_manipulate_pen`
- `gym_robotics__gymnasium_robotics_envs_assets_hand_manipulate_pen_touch_sensors`
- `gym_robotics__gymnasium_robotics_envs_mujoco_assets_ant`
- `gym_robotics__gymnasium_robotics_envs_mujoco_assets_humanoid`
- `gym_robotics__gymnasium_robotics_envs_mujoco_assets_humanoidstandup`
- `gym_robotics__gymnasium_robotics_envs_mujoco_assets_inverted_double_pendulum`
- `gym_robotics__gymnasium_robotics_envs_mujoco_assets_pusher`
- `menagerie__berkeley_humanoid_berkeley_humanoid`
- `menagerie__berkeley_humanoid_scene`
- `menagerie__franka_emika_panda_mjx_single_cube`
- `menagerie__kinova_gen3_gen3`
- `menagerie__kinova_gen3_scene`
- `menagerie__robotiq_2f85_2f85`
- `menagerie__robotiq_2f85_scene`
- `menagerie__robotiq_2f85_v4_2f85`
- `menagerie__robotiq_2f85_v4_mjx_2f85`
- `menagerie__robotiq_2f85_v4_scene`
- `menagerie__universal_robots_ur10e_scene`
- `menagerie__universal_robots_ur10e_ur10e`
- `menagerie__universal_robots_ur5e_scene`
- `menagerie__universal_robots_ur5e_ur5e`
- `mjx__mjx_mujoco_mjx_test_data_humanoid_01_humanoids`
- `mjx__mjx_mujoco_mjx_test_data_humanoid_humanoid`
- `mjx__mjx_mujoco_mjx_test_data_shadow_hand_right_hand`
- `mujoco__model_humanoid_humanoid`
- `mujoco_playground__mujoco_playground__src_dm_control_suite_xmls_humanoid`
- `robosuite__robosuite_models_assets_robots_ur5e_robot`
- `safety_gymnasium__safety_gymnasium_assets_xmls_ant`
- `safety_gymnasium__safety_gymnasium_tasks_safe_multi_agent_assets_xmls_multi_ant`
