from gymnasium.envs.registration import register

register(
    id='ExtruderControl-v0',
    entry_point='extruder_control_env:ExtruderControlEnv',
)

register(
    id='ExtruderControl-v1',
    entry_point='extruder_control_env_fixed_profiles:ExtruderControlEnvFixed',
)

register(
    id='ExtruderControl-v2',
    entry_point='extruder_control_env_continuous:ExtruderControlEnvContinuous'
)

register(
    id='ExtruderControl-v3',
    entry_point='extruder_control_env_multiple_layer:ExtruderControlEnvMultipleLayer'
)

register(
    id='ExtruderControl-v4',
    entry_point='extruder_control_env_hard_constraint:ExtruderControlEnvHardConstraint'
)

register(
    id='ExtruderControl-v5',
    entry_point='extruder_control_env_acceleration:ExtruderControlEnvAcceleration'
)

register(
    id='ExtruderControl-v6',
    entry_point='extruder_control_env_multiple_layer_case_study:ExtruderControlEnvAllLayers'
)
