# NumPy version for inference
import numpy as np


def reward_imitation(
    base_qpos,
    base_qvel,
    joints_qpos,
    joints_qvel,
    contacts,
    reference_frame,
    cmd,
    use_imitation_reward=False,
    gait_phase=0.0,
    ref_phase=0.0,
):
    """
    Disney-style MULTIPLICATIVE imitation reward with phase tracking.
    NumPy version for inference.
    
    Key differences from additive rewards:
    1. MULTIPLICATIVE: All components multiply together - ALL must be good
    2. PHASE TRACKING: Rewards matching the reference motion's phase
    3. BOUNDED [0, 1]: Each component in [0,1], final reward also in [0,1]
    
    Args:
        base_qpos: Base position and orientation [x, y, z, qw, qx, qy, qz]
        base_qvel: Base linear and angular velocity [vx, vy, vz, wx, wy, wz]
        joints_qpos: Joint positions
        joints_qvel: Joint velocities
        contacts: Binary foot contacts [left, right]
        reference_frame: Reference motion data from polynomial interpolation
        cmd: Command array
        use_imitation_reward: Whether to compute imitation reward
        gait_phase: Current gait phase [0, 1]
        ref_phase: Reference motion phase [0, 1]
    
    Returns:
        Reward value in [0, 1] (multiplicative product)
    """
    if not use_imitation_reward:
        return np.nan_to_num(0.0)

    cmd_norm = np.linalg.norm(cmd[:3])

    # ==========================================================================
    # Sensitivity parameters (higher = stricter matching, faster reward decay)
    # ==========================================================================
    k_joint_pos = 40.0      # Joint positions - most important
    k_joint_vel = 0.1       # Joint velocities - less sensitive
    k_lin_vel = 20.0        # Base linear velocity
    k_ang_vel = 10.0        # Base angular velocity
    k_orientation = 20.0    # Torso orientation
    k_phase = 50.0          # Phase synchronization

    # ==========================================================================
    # Reference frame slice indices
    # ==========================================================================
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

    # ==========================================================================
    # Extract reference and actual values
    # ==========================================================================
    
    # Orientation
    ref_base_orientation_quat = reference_frame[root_quat_slice_start:root_quat_slice_end]
    ref_base_orientation_quat = ref_base_orientation_quat / np.linalg.norm(ref_base_orientation_quat)
    base_orientation = base_qpos[3:7]
    base_orientation = base_orientation / np.linalg.norm(base_orientation)

    # Linear velocity
    ref_base_lin_vel = reference_frame[linear_vel_slice_start:linear_vel_slice_end]
    base_lin_vel = base_qvel[:3]

    # Angular velocity
    ref_base_ang_vel = reference_frame[angular_vel_slice_start:angular_vel_slice_end]
    base_ang_vel = base_qvel[3:6]

    # Joint positions (remove neck, head, and antennas)
    ref_joint_pos = reference_frame[joint_pos_slice_start:joint_pos_slice_end]
    ref_joint_pos = np.concatenate([ref_joint_pos[:5], ref_joint_pos[11:]])
    joint_pos = np.concatenate([joints_qpos[:5], joints_qpos[9:]])

    # Joint velocities (remove neck, head, and antennas)
    ref_joint_vels = reference_frame[joint_vels_slice_start:joint_vels_slice_end]
    ref_joint_vels = np.concatenate([ref_joint_vels[:5], ref_joint_vels[11:]])
    joint_vel = np.concatenate([joints_qvel[:5], joints_qvel[9:]])

    # Foot contacts
    ref_foot_contacts = reference_frame[foot_contacts_slice_start:foot_contacts_slice_end]
    ref_foot_contacts = np.where(
        np.array(ref_foot_contacts) > 0.5,
        np.ones_like(ref_foot_contacts),
        np.zeros_like(ref_foot_contacts),
    )

    # ==========================================================================
    # Compute individual rewards (ALL in [0, 1] range for multiplication)
    # Using exponential kernels: exp(-k * error²) → 1 when error=0, → 0 as error grows
    # ==========================================================================
    
    # Joint position reward (MOST IMPORTANT for natural pose)
    n_joints = joint_pos.shape[0]
    joint_pos_error = np.sum(np.square(joint_pos - ref_joint_pos)) / n_joints
    r_joint_pos = np.exp(-k_joint_pos * joint_pos_error)
    
    # Joint velocity reward
    joint_vel_error = np.sum(np.square(joint_vel - ref_joint_vels)) / n_joints
    r_joint_vel = np.exp(-k_joint_vel * joint_vel_error)
    
    # Base linear velocity reward
    lin_vel_error = np.sum(np.square(base_lin_vel - ref_base_lin_vel))
    r_lin_vel = np.exp(-k_lin_vel * lin_vel_error)
    
    # Base angular velocity reward
    ang_vel_error = np.sum(np.square(base_ang_vel - ref_base_ang_vel))
    r_ang_vel = np.exp(-k_ang_vel * ang_vel_error)
    
    # Torso orientation reward
    orientation_error = np.sum(np.square(base_orientation - ref_base_orientation_quat))
    r_orientation = np.exp(-k_orientation * orientation_error)
    
    # Contact pattern reward (binary match → [0, 0.5, 1])
    contact_matches = np.sum(contacts == ref_foot_contacts) / 2.0
    r_contact = contact_matches
    
    # Phase tracking reward (circular distance handles wrap-around at 0/1)
    phase_diff = min(
        abs(gait_phase - ref_phase),
        1.0 - abs(gait_phase - ref_phase)
    )
    r_phase = np.exp(-k_phase * np.square(phase_diff))

    # ==========================================================================
    # MULTIPLICATIVE combination (Disney-style)
    # Using weighted geometric mean: r^w where w sums to 1.0
    # ALL components must be good for high reward!
    # ==========================================================================
    
    # Weights (must sum to 1.0 for proper geometric mean)
    # These represent relative importance of each component
    reward = (
        np.power(r_joint_pos, 0.40) *     # 40% - pose is king
        np.power(r_joint_vel, 0.05) *     # 5%  - velocity matters less
        np.power(r_lin_vel, 0.15) *       # 15% - base linear velocity
        np.power(r_ang_vel, 0.10) *       # 10% - base angular velocity
        np.power(r_orientation, 0.10) *   # 10% - torso orientation
        np.power(r_contact, 0.10) *       # 10% - contact pattern
        np.power(r_phase, 0.10)           # 10% - phase synchronization
    )
    
    # Zero reward when command is zero
    if cmd_norm <= 0.01:
        reward = 0.0
    
    return np.nan_to_num(reward)
