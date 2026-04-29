# C6 — 状态 reset / reproducibility / warm-start / clone

> Same seed/state/action diverges; set_state breaks contact; mocap weld reset misbehaves.

**Doc anchor:** `C6. 状态 reset / reproducibility / warm-start / clone 类`

**Members:** 14

## Suggested operators

- `same_seed_repeated_reset`
- `same_action_replay`
- `warm_start_toggle`
- `get_set_state_replay`
- `contact_rich_state_capture`
- `mocap_weld_reset`
- `deepcopy_clone`
- `hard_reset_toggle`

## Suggested oracles

- `obs_trajectory_equality`
- `initial_state_equality`
- `object_teleport_after_set_state`
- `penetration_after_set_state`
- `mocap_pose_residual`
- `state_space_key_consistency`
- `cloned_env_divergence`

## Members

- `dm_control__dm_control_suite_manipulator`
- `dm_control__dm_control_suite_stacker`
- `gym_robotics__gymnasium_robotics_envs_assets_fetch_pick_and_place`
- `gym_robotics__gymnasium_robotics_envs_assets_fetch_push`
- `gym_robotics__gymnasium_robotics_envs_assets_fetch_reach`
- `gym_robotics__gymnasium_robotics_envs_assets_fetch_slide`
- `gym_robotics__gymnasium_robotics_envs_assets_hand_manipulate_block`
- `gym_robotics__gymnasium_robotics_envs_assets_hand_manipulate_block_touch_sensors`
- `gym_robotics__gymnasium_robotics_envs_assets_hand_manipulate_egg`
- `gym_robotics__gymnasium_robotics_envs_assets_hand_manipulate_egg_touch_sensors`
- `gym_robotics__gymnasium_robotics_envs_assets_hand_manipulate_pen`
- `gym_robotics__gymnasium_robotics_envs_assets_hand_manipulate_pen_touch_sensors`
- `gym_robotics__gymnasium_robotics_envs_assets_hand_reach`
- `mujoco_playground__mujoco_playground__src_dm_control_suite_xmls_manipulator`
