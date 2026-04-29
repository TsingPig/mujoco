# C8 — 渲染 / 相机 / headless / vectorized

> Offscreen render, GL backend, RecordVideo, multi-cam buffer, deformable / skin rendering bugs.

**Doc anchor:** `C8. 渲染 / 相机 / headless / vectorized rendering 类`

**Members:** 21

## Suggested operators

- `gl_backend_switch`
- `same_state_multi_render`
- `record_video_wrapper`
- `multi_episode_render_context`
- `sync_vector_env_render`
- `multi_camera`
- `import_context_pollution`
- `deformable_skin_render`
- `python_version_matrix`
- `gpu_device_select`

## Suggested oracles

- `framebuffer_completeness`
- `black_frame`
- `pixel_hash_determinism`
- `image_flip`
- `cross_env_frame_contamination`
- `segfault`
- `render_exception_class`
- `camera_frame_schema`

## Members

- `dm_control__dm_control_locomotion_soccer_assets_boxhead_boxhead`
- `dm_control__dm_control_locomotion_walkers_assets_humanoid_CMU_V2019`
- `dm_control__dm_control_locomotion_walkers_assets_humanoid_CMU_V2020`
- `dm_control__dm_control_suite_humanoid`
- `dm_control__dm_control_suite_humanoid_CMU`
- `dm_control__dm_control_suite_manipulator`
- `dm_control__dm_control_suite_stacker`
- `gym_robotics__gymnasium_robotics_envs_assets_fetch_slide`
- `gym_robotics__gymnasium_robotics_envs_mujoco_assets_ant`
- `gym_robotics__gymnasium_robotics_envs_mujoco_assets_half_cheetah`
- `gym_robotics__gymnasium_robotics_envs_mujoco_assets_humanoid`
- `gym_robotics__gymnasium_robotics_envs_mujoco_assets_humanoidstandup`
- `menagerie__berkeley_humanoid_berkeley_humanoid`
- `menagerie__berkeley_humanoid_scene`
- `mjx__mjx_mujoco_mjx_test_data_humanoid_01_humanoids`
- `mjx__mjx_mujoco_mjx_test_data_humanoid_humanoid`
- `mujoco__model_humanoid_humanoid`
- `mujoco_playground__mujoco_playground__src_dm_control_suite_xmls_humanoid`
- `mujoco_playground__mujoco_playground__src_dm_control_suite_xmls_manipulator`
- `safety_gymnasium__safety_gymnasium_assets_xmls_ant`
- `safety_gymnasium__safety_gymnasium_tasks_safe_multi_agent_assets_xmls_multi_ant`
