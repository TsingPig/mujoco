# C2 — 启动稳定性 / 默认控制器发散

> Loads OK but zero / home / small ctrl rollout drifts, oscillates or NaNs.

**Doc anchor:** `C2. 启动稳定性 / 默认控制器发散类`

**Members:** 71

## Suggested operators

- `zero_control_rollout`
- `home_keyframe_reset`
- `small_ctrl_pulse`
- `single_joint_target_sweep`
- `kp_kd_sweep`
- `damping_armature_scale`
- `timestep_integrator_sweep`
- `solver_iter_sweep`
- `contact_exclusion_toggle`

## Suggested oracles

- `qpos_qvel_qacc_finite`
- `base_height_drop`
- `com_drift`
- `oscillation_amplitude`
- `contact_force_spike`
- `kinetic_energy_spike`

## Members

- `dm_control__dm_control_locomotion_walkers_assets_dog_v2_dog`
- `dm_control__dm_control_locomotion_walkers_assets_fruitfly_v2_floor`
- `dm_control__dm_control_locomotion_walkers_assets_fruitfly_v2_fruitfly`
- `dm_control__dm_control_locomotion_walkers_assets_humanoid_CMU_V2019`
- `dm_control__dm_control_locomotion_walkers_assets_humanoid_CMU_V2020`
- `dm_control__dm_control_suite_cheetah`
- `dm_control__dm_control_suite_hopper`
- `dm_control__dm_control_suite_humanoid`
- `dm_control__dm_control_suite_humanoid_CMU`
- `dm_control__dm_control_suite_quadruped`
- `dm_control__dm_control_suite_walker`
- `gym_robotics__gymnasium_robotics_envs_mujoco_assets_ant`
- `gym_robotics__gymnasium_robotics_envs_mujoco_assets_half_cheetah`
- `gym_robotics__gymnasium_robotics_envs_mujoco_assets_hopper`
- `gym_robotics__gymnasium_robotics_envs_mujoco_assets_humanoid`
- `gym_robotics__gymnasium_robotics_envs_mujoco_assets_humanoidstandup`
- `menagerie__agility_cassie_cassie`
- `menagerie__anybotics_anymal_b_anymal_b`
- `menagerie__anybotics_anymal_c_anymal_c`
- `menagerie__anybotics_anymal_c_anymal_c_mjx`
- `menagerie__apptronik_apollo_apptronik_apollo`
- `menagerie__berkeley_humanoid_berkeley_humanoid`
- `menagerie__booster_t1_t1`
- `menagerie__boston_dynamics_spot_spot`
- `menagerie__boston_dynamics_spot_spot_arm`
- `menagerie__fourier_n1_n1`
- `menagerie__franka_emika_panda_mjx_panda`
- `menagerie__franka_emika_panda_mjx_panda_nohand`
- `menagerie__franka_emika_panda_mjx_single_cube`
- `menagerie__franka_emika_panda_panda`
- `menagerie__franka_emika_panda_panda_nohand`
- `menagerie__google_barkour_v0_barkour_v0`
- `menagerie__google_barkour_v0_barkour_v0_mjx`
- `menagerie__google_barkour_vb_barkour_vb`
- `menagerie__google_barkour_vb_barkour_vb_mjx`
- `menagerie__kinova_gen3_gen3`
- `menagerie__kuka_iiwa_14_iiwa14`
- `menagerie__pal_talos_talos`
- `menagerie__pal_tiago_dual_tiago_dual`
- `menagerie__pal_tiago_tiago`
- `menagerie__robotis_op3_op3`
- `menagerie__toddlerbot_2xc_toddlerbot_2xc`
- `menagerie__toddlerbot_2xc_toddlerbot_2xc_mjx`
- `menagerie__toddlerbot_2xc_toddlerbot_2xc_pos`
- `menagerie__toddlerbot_2xm_toddlerbot_2xm`
- `menagerie__toddlerbot_2xm_toddlerbot_2xm_mjx`
- `menagerie__toddlerbot_2xm_toddlerbot_2xm_pos`
- `menagerie__unitree_a1_a1`
- `menagerie__unitree_g1_g1`
- `menagerie__unitree_g1_g1_mjx`
- `menagerie__unitree_g1_g1_with_hands`
- `menagerie__unitree_go1_go1`
- `menagerie__unitree_go2_go2`
- `menagerie__unitree_go2_go2_mjx`
- `menagerie__unitree_h1_h1`
- `menagerie__universal_robots_ur10e_ur10e`
- `menagerie__universal_robots_ur5e_ur5e`
- `mjx__mjx_mujoco_mjx_test_data_barkour_v0_assets_barkour_v0_mjx`
- `mjx__mjx_mujoco_mjx_test_data_humanoid_01_humanoids`
- `mjx__mjx_mujoco_mjx_test_data_humanoid_humanoid`
- `mujoco__model_humanoid_humanoid`
- `mujoco_playground__mujoco_playground__src_dm_control_suite_xmls_cheetah`
- `mujoco_playground__mujoco_playground__src_dm_control_suite_xmls_hopper`
- `mujoco_playground__mujoco_playground__src_dm_control_suite_xmls_humanoid`
- `mujoco_playground__mujoco_playground__src_dm_control_suite_xmls_walker`
- `robosuite__robosuite_models_assets_robots_spot_arm_robot`
- `robosuite__robosuite_models_assets_robots_spot_robot`
- `robosuite__robosuite_models_assets_robots_tiago_robot`
- `robosuite__robosuite_models_assets_robots_ur5e_robot`
- `safety_gymnasium__safety_gymnasium_assets_xmls_ant`
- `safety_gymnasium__safety_gymnasium_tasks_safe_multi_agent_assets_xmls_multi_ant`
