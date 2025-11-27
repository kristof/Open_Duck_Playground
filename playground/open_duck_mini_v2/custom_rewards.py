import jax
import jax.numpy as jp


# =============================================================================
# Gait Timing Rewards
# =============================================================================


def reward_gait_phase(
    gait_phase: float,
    contacts: jax.Array,
    cmd: jax.Array,
    duty_cycle: float = 0.5,
) -> jax.Array:
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
    cmd_norm = jp.linalg.norm(cmd[:3])
    
    # Expected contact pattern based on phase
    # Left foot: stance during [0, duty_cycle], swing during [duty_cycle, 1]
    # Right foot: 180° offset, stance during [0.5, 0.5+duty_cycle] (wrapped)
    left_expected = (gait_phase < duty_cycle).astype(jp.float32)
    right_phase = (gait_phase + 0.5) % 1.0  # 180° offset
    right_expected = (right_phase < duty_cycle).astype(jp.float32)
    
    expected_contacts = jp.array([left_expected, right_expected])
    
    # Reward for matching expected contact pattern
    contact_match = jp.sum(contacts == expected_contacts) / 2.0
    
    reward = contact_match * (cmd_norm > 0.01)
    return jp.nan_to_num(reward)


def reward_foot_clearance(
    foot_heights: jax.Array,
    gait_phase: float,
    cmd: jax.Array,
    target_clearance: float = 0.02,
    duty_cycle: float = 0.5,
) -> jax.Array:
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
    cmd_norm = jp.linalg.norm(cmd[:3])
    
    swing_duration = 1.0 - duty_cycle
    
    # Left foot swing phase: [duty_cycle, 1.0]
    left_in_swing = gait_phase >= duty_cycle
    left_swing_progress = (gait_phase - duty_cycle) / swing_duration
    left_swing_progress = jp.clip(left_swing_progress, 0.0, 1.0)
    # Sinusoidal target: 0 at start/end of swing, max at middle
    left_target = target_clearance * jp.sin(jp.pi * left_swing_progress) * left_in_swing
    
    # Right foot swing phase: offset by 0.5
    right_phase = (gait_phase + 0.5) % 1.0
    right_in_swing = right_phase >= duty_cycle
    right_swing_progress = (right_phase - duty_cycle) / swing_duration
    right_swing_progress = jp.clip(right_swing_progress, 0.0, 1.0)
    right_target = target_clearance * jp.sin(jp.pi * right_swing_progress) * right_in_swing
    
    target_heights = jp.array([left_target, right_target])
    
    # Exponential reward for matching target heights
    height_error = jp.sum(jp.square(foot_heights - target_heights))
    reward = jp.exp(-100.0 * height_error)
    
    reward *= cmd_norm > 0.01
    return jp.nan_to_num(reward)


def reward_feet_air_time(
    air_time: jax.Array,
    first_contact: jax.Array,
    cmd: jax.Array,
    target_air_time: float = 0.2,
    tolerance: float = 0.05,
) -> jax.Array:
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
    cmd_norm = jp.linalg.norm(cmd[:3])
    
    # Only reward when foot just made contact (end of swing)
    air_time_error = jp.abs(air_time - target_air_time)
    air_time_reward = jp.clip(tolerance - air_time_error, 0.0, tolerance) / tolerance
    
    # Sum rewards for feet that just landed
    reward = jp.sum(air_time_reward * first_contact)
    
    reward *= cmd_norm > 0.01
    return jp.nan_to_num(reward)


def cost_swing_velocity(
    foot_velocities: jax.Array,
    contacts: jax.Array,
    cmd: jax.Array,
    max_swing_vel: float = 0.5,
) -> jax.Array:
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
    cmd_norm = jp.linalg.norm(cmd[:3])
    
    # Only penalize swing foot velocities
    in_swing = ~contacts.astype(bool)
    excess_vel = jp.clip(foot_velocities - max_swing_vel, 0.0, None)
    cost = jp.sum(jp.square(excess_vel) * in_swing)
    
    cost *= cmd_norm > 0.01
    return jp.nan_to_num(cost)


def reward_gait_frequency(
    contact_changes: int,
    episode_time: float,
    cmd: jax.Array,
    target_frequency: float = 2.0,
    tolerance: float = 0.5,
) -> jax.Array:
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
    cmd_norm = jp.linalg.norm(cmd[:3])
    
    # Each full gait cycle has 4 contact changes (2 strikes, 2 toe-offs)
    # So frequency = contact_changes / (4 * time) for full cycles
    # Or steps per second = contact_changes / (2 * time) for steps
    actual_frequency = contact_changes / (2.0 * jp.maximum(episode_time, 0.1))
    
    freq_error = jp.abs(actual_frequency - target_frequency)
    reward = jp.exp(-2.0 * jp.square(freq_error / tolerance))
    
    reward *= cmd_norm > 0.01
    return jp.nan_to_num(reward)


# =============================================================================
# Imitation Reward (Fixed: All terms use bounded exponential kernels)
# =============================================================================


def reward_imitation(
    base_qpos: jax.Array,
    base_qvel: jax.Array,
    joints_qpos: jax.Array,
    joints_qvel: jax.Array,
    contacts: jax.Array,
    reference_frame: jax.Array,
    cmd: jax.Array,
    use_imitation_reward: bool = False,
) -> jax.Array:
    """
    Reward for imitating reference motion from polynomial coefficients.
    
    All sub-rewards use exponential kernels for bounded [0, weight] output.
    This ensures stable gradients and makes weight tuning intuitive.
    
    Args:
        base_qpos: Base position and orientation [x, y, z, qw, qx, qy, qz]
        base_qvel: Base linear and angular velocity [vx, vy, vz, wx, wy, wz]
        joints_qpos: Joint positions
        joints_qvel: Joint velocities
        contacts: Binary foot contacts [left, right]
        reference_frame: Reference motion data from polynomial interpolation
        cmd: Command array
        use_imitation_reward: Whether to compute imitation reward
    
    Returns:
        Reward value (bounded, all positive)
    """
    if not use_imitation_reward:
        return jp.nan_to_num(0.0)

    cmd_norm = jp.linalg.norm(cmd[:3])

    # Weights represent relative importance (all rewards output [0, weight])
    # Adjusted weights since joint rewards are now properly bounded
    w_lin_vel_xy = 1.0
    w_lin_vel_z = 0.5
    w_ang_vel_xy = 0.5
    w_ang_vel_z = 0.5
    w_joint_pos = 2.0   # Most important for natural motion
    w_joint_vel = 0.5
    w_contact = 1.0

    # Reference frame slice indices
    linear_vel_slice_start = 34
    linear_vel_slice_end = 37
    angular_vel_slice_start = 37
    angular_vel_slice_end = 40
    joint_pos_slice_start = 0
    joint_pos_slice_end = 16
    joint_vels_slice_start = 16
    joint_vels_slice_end = 32
    root_quat_slice_start = 3
    root_quat_slice_end = 7
    foot_contacts_slice_start = 32
    foot_contacts_slice_end = 34

    # Extract and normalize base orientation
    ref_base_orientation_quat = reference_frame[root_quat_slice_start:root_quat_slice_end]
    ref_base_orientation_quat = ref_base_orientation_quat / jp.linalg.norm(ref_base_orientation_quat)
    base_orientation = base_qpos[3:7]
    base_orientation = base_orientation / jp.linalg.norm(base_orientation)

    # Extract velocities
    ref_base_lin_vel = reference_frame[linear_vel_slice_start:linear_vel_slice_end]
    base_lin_vel = base_qvel[:3]
    ref_base_ang_vel = reference_frame[angular_vel_slice_start:angular_vel_slice_end]
    base_ang_vel = base_qvel[3:6]

    # Extract joint positions (remove neck, head, and antennas)
    ref_joint_pos = reference_frame[joint_pos_slice_start:joint_pos_slice_end]
    ref_joint_pos = jp.concatenate([ref_joint_pos[:5], ref_joint_pos[11:]])
    joint_pos = jp.concatenate([joints_qpos[:5], joints_qpos[9:]])

    # Extract joint velocities (remove neck, head, and antennas)
    ref_joint_vels = reference_frame[joint_vels_slice_start:joint_vels_slice_end]
    ref_joint_vels = jp.concatenate([ref_joint_vels[:5], ref_joint_vels[11:]])
    joint_vel = jp.concatenate([joints_qvel[:5], joints_qvel[9:]])

    # Extract foot contacts
    ref_foot_contacts = reference_frame[foot_contacts_slice_start:foot_contacts_slice_end]
    ref_foot_contacts = jp.where(
        ref_foot_contacts > 0.5,
        jp.ones_like(ref_foot_contacts),
        jp.zeros_like(ref_foot_contacts),
    )

    # =========================================================================
    # Compute rewards (ALL use exponential kernels for bounded [0, weight] output)
    # =========================================================================
    
    # Linear velocity XY tracking
    lin_vel_xy_rew = (
        jp.exp(-8.0 * jp.sum(jp.square(base_lin_vel[:2] - ref_base_lin_vel[:2])))
        * w_lin_vel_xy
    )
    
    # Linear velocity Z tracking
    lin_vel_z_rew = (
        jp.exp(-8.0 * jp.square(base_lin_vel[2] - ref_base_lin_vel[2]))
        * w_lin_vel_z
    )

    # Angular velocity XY tracking (roll/pitch rate)
    ang_vel_xy_rew = (
        jp.exp(-2.0 * jp.sum(jp.square(base_ang_vel[:2] - ref_base_ang_vel[:2])))
        * w_ang_vel_xy
    )
    
    # Angular velocity Z tracking (yaw rate)
    ang_vel_z_rew = (
        jp.exp(-2.0 * jp.square(base_ang_vel[2] - ref_base_ang_vel[2]))
        * w_ang_vel_z
    )

    # FIXED: Joint position tracking (normalized, bounded [0, weight])
    # Using mean squared error for scale-invariance across different joint counts
    n_joints = joint_pos.shape[0]
    joint_pos_error = jp.sum(jp.square(joint_pos - ref_joint_pos)) / n_joints
    joint_pos_rew = jp.exp(-10.0 * joint_pos_error) * w_joint_pos

    # FIXED: Joint velocity tracking (normalized, bounded [0, weight])
    joint_vel_error = jp.sum(jp.square(joint_vel - ref_joint_vels)) / n_joints
    joint_vel_rew = jp.exp(-0.5 * joint_vel_error) * w_joint_vel

    # Contact pattern matching (bounded [0, weight])
    contact_matches = jp.sum(contacts == ref_foot_contacts)
    contact_rew = (contact_matches / 2.0) * w_contact

    # =========================================================================
    # Sum all rewards (now all positive and bounded)
    # =========================================================================
    reward = (
        lin_vel_xy_rew
        + lin_vel_z_rew
        + ang_vel_xy_rew
        + ang_vel_z_rew
        + joint_pos_rew
        + joint_vel_rew
        + contact_rew
    )

    reward *= cmd_norm > 0.01  # No reward for zero commands
    return jp.nan_to_num(reward)