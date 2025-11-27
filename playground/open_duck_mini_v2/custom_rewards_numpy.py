# import jax
# import jax.numpy as np
import numpy as np


# =============================================================================
# Gait Timing Rewards (NumPy version for inference)
# =============================================================================


def reward_gait_phase(
    gait_phase: float,
    contacts: np.ndarray,
    cmd: np.ndarray,
    duty_cycle: float = 0.5,
) -> float:
    """
    Reward for maintaining proper alternating gait pattern.
    
    The gait phase goes from 0 to 1 over one full gait cycle.
    For a walking gait with 50% duty cycle:
    - Left foot should be in stance when phase is in [0, 0.5]
    - Right foot should be in stance when phase is in [0.5, 1.0]
    
    Args:
        gait_phase: Current phase in the gait cycle [0, 1]
        contacts: Binary contact array [left_contact, right_contact]
        cmd: Command array (reward disabled when cmd ≈ 0)
        duty_cycle: Fraction of cycle each foot spends in stance (default 0.5)
    
    Returns:
        Reward value in [0, 1]
    """
    cmd_norm = np.linalg.norm(cmd[:3])
    
    # Expected contact pattern based on phase
    # Left foot: stance during [0, duty_cycle], swing during [duty_cycle, 1]
    # Right foot: 180° offset, stance during [0.5, 0.5+duty_cycle] (wrapped)
    left_expected = float(gait_phase < duty_cycle)
    right_phase = (gait_phase + 0.5) % 1.0  # 180° offset
    right_expected = float(right_phase < duty_cycle)
    
    expected_contacts = np.array([left_expected, right_expected])
    
    # Reward for matching expected contact pattern
    contact_match = np.sum(contacts == expected_contacts) / 2.0
    
    reward = contact_match * (cmd_norm > 0.01)
    return np.nan_to_num(reward)


def reward_foot_clearance(
    foot_heights: np.ndarray,
    gait_phase: float,
    cmd: np.ndarray,
    target_clearance: float = 0.02,
    duty_cycle: float = 0.5,
) -> float:
    """
    Reward for proper foot clearance during swing phase.
    
    Encourages feet to lift to target height at mid-swing, following
    a sinusoidal trajectory during the swing phase.
    
    Args:
        foot_heights: Height of each foot [left_z, right_z]
        gait_phase: Current phase in the gait cycle [0, 1]
        cmd: Command array (reward disabled when cmd ≈ 0)
        target_clearance: Maximum foot height at mid-swing (meters)
        duty_cycle: Fraction of cycle each foot spends in stance
    
    Returns:
        Reward value in [0, 1]
    """
    cmd_norm = np.linalg.norm(cmd[:3])
    
    swing_duration = 1.0 - duty_cycle
    
    # Left foot swing phase: [duty_cycle, 1.0]
    left_in_swing = gait_phase >= duty_cycle
    left_swing_progress = (gait_phase - duty_cycle) / swing_duration
    left_swing_progress = np.clip(left_swing_progress, 0.0, 1.0)
    # Sinusoidal target: 0 at start/end of swing, max at middle
    left_target = target_clearance * np.sin(np.pi * left_swing_progress) * left_in_swing
    
    # Right foot swing phase: offset by 0.5
    right_phase = (gait_phase + 0.5) % 1.0
    right_in_swing = right_phase >= duty_cycle
    right_swing_progress = (right_phase - duty_cycle) / swing_duration
    right_swing_progress = np.clip(right_swing_progress, 0.0, 1.0)
    right_target = target_clearance * np.sin(np.pi * right_swing_progress) * right_in_swing
    
    target_heights = np.array([left_target, right_target])
    
    # Exponential reward for matching target heights
    height_error = np.sum(np.square(foot_heights - target_heights))
    reward = np.exp(-100.0 * height_error)
    
    reward *= cmd_norm > 0.01
    return np.nan_to_num(reward)


def reward_feet_air_time(
    air_time: np.ndarray,
    first_contact: np.ndarray,
    cmd: np.ndarray,
    target_air_time: float = 0.2,
    tolerance: float = 0.05,
) -> float:
    """
    Reward for maintaining appropriate swing duration for each foot.
    
    Args:
        air_time: Time each foot has been in the air [left, right]
        first_contact: Boolean array indicating first contact after swing
        cmd: Command array (reward disabled when cmd ≈ 0)
        target_air_time: Desired swing duration in seconds
        tolerance: Acceptable deviation from target
    
    Returns:
        Reward value
    """
    cmd_norm = np.linalg.norm(cmd[:3])
    
    # Only reward when foot just made contact (end of swing)
    air_time_error = np.abs(air_time - target_air_time)
    air_time_reward = np.clip(tolerance - air_time_error, 0.0, tolerance) / tolerance
    
    # Sum rewards for feet that just landed
    reward = np.sum(air_time_reward * first_contact)
    
    reward *= cmd_norm > 0.01
    return np.nan_to_num(reward)


def cost_swing_velocity(
    foot_velocities: np.ndarray,
    contacts: np.ndarray,
    cmd: np.ndarray,
    max_swing_vel: float = 0.5,
) -> float:
    """
    Cost for excessive foot velocity during swing (prevents flailing).
    
    Args:
        foot_velocities: Velocity magnitude of each foot [left_vel, right_vel]
        contacts: Binary contact array [left_contact, right_contact]
        cmd: Command array
        max_swing_vel: Maximum acceptable swing velocity
    
    Returns:
        Cost value (to be negatively weighted)
    """
    cmd_norm = np.linalg.norm(cmd[:3])
    
    # Only penalize swing foot velocities
    in_swing = ~contacts.astype(bool)
    excess_vel = np.clip(foot_velocities - max_swing_vel, 0.0, None)
    cost = np.sum(np.square(excess_vel) * in_swing)
    
    cost *= cmd_norm > 0.01
    return np.nan_to_num(cost)


def reward_gait_frequency(
    contact_changes: int,
    episode_time: float,
    cmd: np.ndarray,
    target_frequency: float = 2.0,
    tolerance: float = 0.5,
) -> float:
    """
    Reward for maintaining target stepping frequency.
    
    Args:
        contact_changes: Number of contact state changes (foot strikes + toe-offs)
        episode_time: Time elapsed in episode
        cmd: Command array
        target_frequency: Desired steps per second (Hz)
        tolerance: Acceptable deviation from target
    
    Returns:
        Reward value in [0, 1]
    """
    cmd_norm = np.linalg.norm(cmd[:3])
    
    # Each full gait cycle has 4 contact changes (2 strikes, 2 toe-offs)
    # So frequency = contact_changes / (4 * time) for full cycles
    # Or steps per second = contact_changes / (2 * time) for steps
    actual_frequency = contact_changes / (2.0 * max(episode_time, 0.1))
    
    freq_error = np.abs(actual_frequency - target_frequency)
    reward = np.exp(-2.0 * np.square(freq_error / tolerance))
    
    reward *= cmd_norm > 0.01
    return np.nan_to_num(reward)


# =============================================================================
# Imitation Reward
# =============================================================================


def reward_imitation(
    base_qpos,
    base_qvel,
    joints_qpos,
    joints_qvel,
    contacts,
    reference_frame,
    cmd,
    use_imitation_reward=False,
):
    if not use_imitation_reward:
        return np.nan_to_num(0.0)

    # TODO don't reward for moving when the command is zero.
    cmd_norm = np.linalg.norm(cmd[:3])

    w_torso_pos = 1.0
    w_torso_orientation = 1.0
    w_lin_vel_xy = 1.0
    w_lin_vel_z = 1.0
    w_ang_vel_xy = 0.5
    w_ang_vel_z = 0.5
    w_joint_pos = 15.0
    w_joint_vel = 1.0e-3
    w_contact = 1.0

    #  TODO : double check if the slices are correct
    linear_vel_slice_start = 34
    linear_vel_slice_end = 37

    angular_vel_slice_start = 37
    angular_vel_slice_end = 40

    joint_pos_slice_start = 0
    joint_pos_slice_end = 16

    joint_vels_slice_start = 16
    joint_vels_slice_end = 32

    # root_pos_slice_start = 0
    # root_pos_slice_end = 3

    root_quat_slice_start = 3
    root_quat_slice_end = 7

    # left_toe_pos_slice_start = 23
    # left_toe_pos_slice_end = 26

    # right_toe_pos_slice_start = 26
    # right_toe_pos_slice_end = 29

    foot_contacts_slice_start = 32
    foot_contacts_slice_end = 34

    # ref_base_pos = reference_frame[root_pos_slice_start:root_pos_slice_end]
    # base_pos = qpos[:3]

    ref_base_orientation_quat = reference_frame[
        root_quat_slice_start:root_quat_slice_end
    ]
    ref_base_orientation_quat = ref_base_orientation_quat / np.linalg.norm(
        ref_base_orientation_quat
    )  # normalize the quat
    base_orientation = base_qpos[3:7]
    base_orientation = base_orientation / np.linalg.norm(
        base_orientation
    )  # normalize the quat

    ref_base_lin_vel = reference_frame[linear_vel_slice_start:linear_vel_slice_end]
    base_lin_vel = base_qvel[:3]

    ref_base_ang_vel = reference_frame[angular_vel_slice_start:angular_vel_slice_end]
    base_ang_vel = base_qvel[3:6]

    ref_joint_pos = reference_frame[joint_pos_slice_start:joint_pos_slice_end]
    # remove neck head and antennas
    ref_joint_pos = np.concatenate([ref_joint_pos[:5], ref_joint_pos[11:]])
    # joint_pos = joints_qpos
    joint_pos = np.concatenate([joints_qpos[:5], joints_qpos[9:]])

    ref_joint_vels = reference_frame[joint_vels_slice_start:joint_vels_slice_end]
    # remove neck head and antennas
    ref_joint_vels = np.concatenate([ref_joint_vels[:5], ref_joint_vels[11:]])
    # joint_vel = joints_qvel
    joint_vel = np.concatenate([joints_qvel[:5], joints_qvel[9:]])

    # ref_left_toe_pos = reference_frame[left_toe_pos_slice_start:left_toe_pos_slice_end]
    # ref_right_toe_pos = reference_frame[right_toe_pos_slice_start:right_toe_pos_slice_end]

    ref_foot_contacts = reference_frame[
        foot_contacts_slice_start:foot_contacts_slice_end
    ]

    # reward
    # torso_pos_rew = np.exp(-200.0 * np.sum(np.square(base_pos[:2] - ref_base_pos[:2]))) * w_torso_pos

    # real quaternion angle doesn't have the expected  effect, switching back for now
    # torso_orientation_rew = np.exp(-20 * self.quaternion_angle(base_orientation, ref_base_orientation_quat)) * w_torso_orientation

    # TODO ignore yaw here, we just want xy orientation
    torso_orientation_rew = (
        np.exp(-20.0 * np.sum(np.square(base_orientation - ref_base_orientation_quat)))
        * w_torso_orientation
    )

    lin_vel_xy_rew = (
        np.exp(-8.0 * np.sum(np.square(base_lin_vel[:2] - ref_base_lin_vel[:2])))
        * w_lin_vel_xy
    )
    lin_vel_z_rew = (
        np.exp(-8.0 * np.sum(np.square(base_lin_vel[2] - ref_base_lin_vel[2])))
        * w_lin_vel_z
    )

    ang_vel_xy_rew = (
        np.exp(-2.0 * np.sum(np.square(base_ang_vel[:2] - ref_base_ang_vel[:2])))
        * w_ang_vel_xy
    )
    ang_vel_z_rew = (
        np.exp(-2.0 * np.sum(np.square(base_ang_vel[2] - ref_base_ang_vel[2])))
        * w_ang_vel_z
    )

    joint_pos_rew = -np.sum(np.square(joint_pos - ref_joint_pos)) * w_joint_pos
    joint_vel_rew = -np.sum(np.square(joint_vel - ref_joint_vels)) * w_joint_vel

    ref_foot_contacts = np.where(
        np.array(ref_foot_contacts) > 0.5,
        np.ones_like(ref_foot_contacts),
        np.zeros_like(ref_foot_contacts),
    )
    contact_rew = np.sum(contacts == ref_foot_contacts) * w_contact

    reward = (
        lin_vel_xy_rew
        + lin_vel_z_rew
        + ang_vel_xy_rew
        + ang_vel_z_rew
        + joint_pos_rew
        + joint_vel_rew
        + contact_rew
        # + torso_orientation_rew
    )

    reward *= cmd_norm > 0.01  # No reward for zero commands.
    return np.nan_to_num(reward)
