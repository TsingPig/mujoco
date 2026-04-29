# C1 — MJCF 加载 / schema / include / asset 引用

> Loader fails or differs across parsers/versions; many-mesh / nested-include / missing-asset stress.

**Doc anchor:** `C1. MJCF 加载 / schema / include / asset 引用类`

**Members:** 32

## Suggested operators

- `file_existence_scan`
- `meshdir_texturedir_mutation`
- `include_path_rewrite`
- `autolimits_toggle`
- `urdf_mjcf_roundtrip`
- `pymjcf_native_diff_load`
- `version_matrix_compile`

## Suggested oracles

- `from_xml_path_success`
- `schema_violation_class`
- `native_vs_pymjcf_consistency`
- `cross_version_compile_consistency`
- `asset_reachability`

## Members

- `dm_control__dm_control_locomotion_soccer_assets_boxhead_boxhead`
- `dm_control__dm_control_locomotion_walkers_assets_dog_v2_dog`
- `dm_control__dm_control_locomotion_walkers_assets_fruitfly_v2_floor`
- `dm_control__dm_control_locomotion_walkers_assets_fruitfly_v2_fruitfly`
- `dm_control__dm_control_locomotion_walkers_assets_humanoid_CMU_V2019`
- `dm_control__dm_control_locomotion_walkers_assets_humanoid_CMU_V2020`
- `dm_control__dm_control_suite_dog`
- `dm_control__dm_control_suite_humanoid_CMU`
- `menagerie__apptronik_apollo_apptronik_apollo`
- `menagerie__berkeley_humanoid_berkeley_humanoid`
- `menagerie__booster_t1_t1`
- `menagerie__flybody_fruitfly`
- `menagerie__pal_talos_talos`
- `menagerie__pal_tiago_dual_tiago_dual`
- `menagerie__pal_tiago_tiago`
- `menagerie__robotiq_2f85_v4_2f85`
- `menagerie__robotiq_2f85_v4_mjx_2f85`
- `menagerie__shadow_dexee_shadow_dexee`
- `menagerie__shadow_hand_left_hand`
- `menagerie__shadow_hand_right_hand`
- `menagerie__ufactory_lite6_lite6`
- `menagerie__ufactory_lite6_lite6_gripper_narrow`
- `menagerie__ufactory_lite6_lite6_gripper_wide`
- `menagerie__unitree_g1_g1`
- `menagerie__unitree_g1_g1_mjx`
- `menagerie__unitree_g1_g1_with_hands`
- `mjx__mjx_mujoco_mjx_test_data_shadow_hand_right_hand`
- `myosuite__myosuite_simhive_myo_sim_elbow_myoelbow_1dof6muscles_1dofSoftexo_Ideal`
- `robosuite__robosuite_models_assets_robots_tiago_robot`
- `safety_gymnasium__safety_gymnasium_assets_xmls_doggo`
- `safety_gymnasium__safety_gymnasium_tasks_safe_isaac_gym_envs_assets_mjcf_shadow_hand_description_shadow_hand`
- `safety_gymnasium__safety_gymnasium_tasks_safe_isaac_gym_envs_assets_mjcf_shadow_hand_description_shadow_hand1`
