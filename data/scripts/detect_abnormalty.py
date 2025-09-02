import numpy as np
from scipy import signal

# Define joint indices according to your skeleton format
LEFT_FOOT_IDX = 10    
RIGHT_FOOT_IDX = 11   

# Keypoint indices for detecting sitting/squatting
PELVIS_IDX = 0
HEAD_IDX = 15
LEFT_KNEE_IDX = 4
RIGHT_KNEE_IDX = 5
LEFT_ANKLE_IDX = 7   # 新增踝关节
RIGHT_ANKLE_IDX = 8  # 新增踝关节
LEFT_HIP_IDX = 1     # 新增髋关节
RIGHT_HIP_IDX = 2    # 新增髋关节

def is_squatting(joints3d, sitting_mask, knee_thresh=0.2, head_pelvis_thresh=0.6):
    """
    Determine whether a frame is squatting instead of sitting.

    Args:
        joints3d (np.ndarray): 3D joint coordinates of shape (N, J, 3).
        sitting_mask (np.ndarray): Boolean array of shape (N,), indicating frames detected as sitting by pelvis height.
        knee_thresh (float): Maximum pelvis-knee height difference to be considered squatting.
        head_pelvis_thresh (float): Minimum head-pelvis height difference to be considered upright enough (i.e., not sitting).

    Returns:
        np.ndarray: Boolean array of shape (N,), indicating squatting frames among sitting candidates.
    """
    pelvis_z = joints3d[:, PELVIS_IDX, 2]
    head_z = joints3d[:, HEAD_IDX, 2]
    left_knee_z = joints3d[:, LEFT_KNEE_IDX, 2]
    right_knee_z = joints3d[:, RIGHT_KNEE_IDX, 2]

    knee_z = np.minimum(left_knee_z, right_knee_z)

    # Knee and pelvis at similar height -> candidate for squatting
    knee_close = np.abs(pelvis_z - knee_z) < knee_thresh

    # Head remains high above pelvis -> standing posture (not seated)
    head_high = (head_z - pelvis_z) > head_pelvis_thresh

    squat_mask = knee_close & head_high
    return sitting_mask & squat_mask

def detect_pose_abnormalities(joints3d):
    """
    检测异常姿态（如关节穿透、非自然角度等）
    """
    if joints3d is None:
        return False, 0
    
    N = joints3d.shape[0]
    abnormal_frames = []
    
    for i in range(N):
        frame_joints = joints3d[i]
        
        # 检查膝关节是否过度弯曲（膝盖在脚踝上方太多）
        left_knee = frame_joints[LEFT_KNEE_IDX]
        right_knee = frame_joints[RIGHT_KNEE_IDX]
        left_ankle = frame_joints[LEFT_ANKLE_IDX] if LEFT_ANKLE_IDX < frame_joints.shape[0] else frame_joints[LEFT_FOOT_IDX]
        right_ankle = frame_joints[RIGHT_ANKLE_IDX] if RIGHT_ANKLE_IDX < frame_joints.shape[0] else frame_joints[RIGHT_FOOT_IDX]
        
        # 膝关节不应该比脚踝低太多（站立时）
        if (left_knee[2] - left_ankle[2] < -0.8) or (right_knee[2] - right_ankle[2] < -0.8):
            abnormal_frames.append(i)
            continue
            
        # 检查头部是否在合理位置（不应该比骨盆低太多，除非是特殊动作）
        head = frame_joints[HEAD_IDX]
        pelvis = frame_joints[PELVIS_IDX]
        if head[2] - pelvis[2] < -0.5:  # 头比骨盆低50cm以上
            abnormal_frames.append(i)
    
    if len(abnormal_frames) > N * 0.1:  # 超过10%的帧异常
        return True, abnormal_frames[0]
    
    return False, 0

def detect_motion_smoothness(trans, joints3d=None, smoothness_thresh=3.0):
    """
    检测运动的平滑性，识别异常的急停急转
    """
    if trans.shape[0] < 3:
        return False, 0
    
    # 计算三阶导数（jerk）来评估运动平滑性
    velocity = np.diff(trans, axis=0)
    acceleration = np.diff(velocity, axis=0) 
    jerk = np.diff(acceleration, axis=0)
    
    # 计算jerk的幅度
    jerk_magnitude = np.linalg.norm(jerk, axis=1)
    
    # 如果jerk过大，说明运动不平滑
    high_jerk_mask = jerk_magnitude > smoothness_thresh
    
    if high_jerk_mask.any():
        first_idx = np.where(high_jerk_mask)[0][0] + 2  # 加2因为计算了二次diff
        return True, first_idx
    
    return False, 0

def detect_trembling(joints3d, trembling_freq_thresh=5.0, trembling_amp_thresh=0.05, fps=30, 
                    min_motion_thresh=0.02, static_motion_thresh=0.01):
    """
    检测颤抖或高频抖动
    
    Args:
        joints3d: 关节点3D坐标数据 (N, num_joints, 3)
        trembling_freq_thresh: 颤抖频率阈值 (Hz)
        trembling_amp_thresh: 颤抖幅度阈值 (单位：米或相对单位)
        fps: 视频帧率
        min_motion_thresh: 最小运动阈值，低于此值认为是静态
        static_motion_thresh: 静态情况下的噪声阈值
    
    Returns:
        is_trembling: 是否检测到颤抖
        start_frame: 颤抖开始的帧数
    """
    if joints3d is None or joints3d.shape[0] < 30:  # 至少需要1秒的数据
        return False, 0
    
    # 选择几个关键关节点来检测颤抖
    key_joints = [HEAD_IDX, PELVIS_IDX, LEFT_FOOT_IDX, RIGHT_FOOT_IDX]
    
    trembling_scores = []
    motion_levels = []
    
    for joint_idx in key_joints:
        if joint_idx >= joints3d.shape[1]:
            continue
            
        joint_pos = joints3d[:, joint_idx, :]  # (N, 3)
        
        # 计算整体运动幅度
        overall_motion = _calculate_motion_magnitude(joint_pos)
        motion_levels.append(overall_motion)
        
        for axis in range(3):
            pos_axis = joint_pos[:, axis]
            
            # 先检查是否为静态或微动
            axis_motion = np.std(pos_axis)
            is_static = axis_motion < static_motion_thresh
            
            if is_static:
                # 静态情况下，更严格的检测标准
                trembling_detected = _detect_static_trembling(pos_axis, trembling_freq_thresh, fps)
            else:
                # 动态情况下，使用原有的检测方法
                trembling_detected_freq = _detect_by_frequency(pos_axis, trembling_freq_thresh, fps)
                trembling_detected_time = _detect_by_acceleration(pos_axis, trembling_amp_thresh)
                trembling_detected = trembling_detected_freq and trembling_detected_time  # 两个条件都满足
            
            trembling_scores.append(1 if trembling_detected else 0)
    
    # 检查整体运动水平
    avg_motion = np.mean(motion_levels) if motion_levels else 0
    
    # 如果整体运动太小，提高检测阈值
    if avg_motion < min_motion_thresh:
        trembling_threshold = 0.4  # 静态时需要更多证据
    else:
        trembling_threshold = 0.2  # 动态时保持原阈值
    
    trembling_ratio = np.mean(trembling_scores)
    is_trembling = trembling_ratio > trembling_threshold
    
    # 简化的开始帧检测
    start_frame = 2 if is_trembling else 0
    
    return is_trembling, start_frame

def _detect_by_frequency(pos_sequence, freq_thresh, fps):
    """使用频域分析检测颤抖"""
    if len(pos_sequence) < 30:
        return False
    
    # 移除趋势（去除整体运动）
    detrended = signal.detrend(pos_sequence)
    
    # FFT分析
    freqs = np.fft.fftfreq(len(detrended), 1/fps)
    fft_vals = np.abs(np.fft.fft(detrended))
    
    # 只看正频率部分
    pos_freqs = freqs[:len(freqs)//2]
    pos_fft = fft_vals[:len(fft_vals)//2]
    
    # 检查高频成分
    high_freq_mask = pos_freqs >= freq_thresh
    if not np.any(high_freq_mask):
        return False
    
    # 计算高频能量占比
    total_energy = np.sum(pos_fft**2)
    high_freq_energy = np.sum(pos_fft[high_freq_mask]**2)
    
    return (high_freq_energy / total_energy) > 0.1  # 高频能量占比超过10%

def _detect_by_acceleration(pos_sequence, amp_thresh):
    """使用加速度变化检测颤抖"""
    if len(pos_sequence) < 10:
        return False
    
    # 计算速度和加速度
    vel = np.diff(pos_sequence)
    acc = np.diff(vel)
    
    # 计算加速度的变化率
    acc_changes = np.abs(np.diff(acc))
    
    # 统计显著的加速度变化
    significant_changes = acc_changes > amp_thresh
    change_ratio = np.mean(significant_changes)
    
    return change_ratio > 0.4  # 40%的点有显著加速度变化

def _calculate_motion_magnitude(joint_pos):
    """计算关节点的整体运动幅度"""
    # 计算位置变化的标准差
    motion_std = np.std(joint_pos, axis=0)  # 每个轴的标准差
    return np.mean(motion_std)  # 平均标准差作为运动幅度

def _detect_static_trembling(pos_sequence, freq_thresh, fps):
    """专门用于静态情况下的颤抖检测，更严格的标准"""
    if len(pos_sequence) < 30:
        return False
    
    # 使用更严格的去噪
    from scipy.signal import savgol_filter
    try:
        # 使用Savitzky-Golay滤波器去除低频噪声
        smoothed = savgol_filter(pos_sequence, window_length=min(11, len(pos_sequence)//3), polyorder=2)
        residual = pos_sequence - smoothed
    except:
        # 如果滤波失败，使用简单的移动平均
        window_size = min(5, len(pos_sequence)//4)
        smoothed = np.convolve(pos_sequence, np.ones(window_size)/window_size, mode='same')
        residual = pos_sequence - smoothed
    
    # 检查残差的幅度是否足够大
    residual_std = np.std(residual)
    if residual_std < 0.005:  # 非常小的残差，可能只是噪声
        return False
    
    # 频域分析残差
    freqs = np.fft.fftfreq(len(residual), 1/fps)
    fft_vals = np.abs(np.fft.fft(residual))
    
    pos_freqs = freqs[:len(freqs)//2]
    pos_fft = fft_vals[:len(fft_vals)//2]
    
    # 检查特定频率范围内的能量
    trembling_band = (pos_freqs >= freq_thresh) & (pos_freqs <= 15)  # 5-15Hz为典型颤抖频率
    if not np.any(trembling_band):
        return False
    
    total_energy = np.sum(pos_fft**2)
    trembling_energy = np.sum(pos_fft[trembling_band]**2)
    
    # 静态情况下需要更高的能量比例
    return (trembling_energy / total_energy) > 0.25


def detect_penetration(joints3d, penetration_thresh=-0.2):
    """
    检测身体部位是否穿透地面
    """
    if joints3d is None:
        return False, 0
    
    N = joints3d.shape[0]
    
    # 检查关键关节点是否穿透地面
    key_joints = [LEFT_FOOT_IDX, RIGHT_FOOT_IDX, LEFT_KNEE_IDX, RIGHT_KNEE_IDX, PELVIS_IDX]
    
    for joint_idx in key_joints:
        if joint_idx >= joints3d.shape[1]:
            continue
            
        joint_z = joints3d[:, joint_idx, 2]
        penetration_mask = joint_z < penetration_thresh
        
        if penetration_mask.sum() > N * 0.1:  # 超过10%的帧穿透
            first_idx = np.where(penetration_mask)[0][0]
            return True, first_idx
    
    return False, 0

def detect_speed_abnormalities(trans, joints3d=None, min_speed_thresh=0, max_speed_thresh=0.3, 
                              abnormal_ratio_thresh=0.2):
    """
    检测运动速度异常（过快或过慢）
    
    Args:
        trans: 平移数据 (N, 3)
        joints3d: 关节3D数据 (N, J, 3)，可选
        min_speed_thresh: 最小速度阈值（m/frame）
        max_speed_thresh: 最大速度阈值（m/frame）
        abnormal_ratio_thresh: 异常帧比例阈值
    """
    if trans.shape[0] < 2:
        return False, 0
    
    # 计算整体运动速度
    velocity = np.linalg.norm(np.diff(trans, axis=0), axis=1)  # (N-1,)
    
    # 检测异常缓慢的运动（几乎静止但应该在运动）
    too_slow_mask = velocity < min_speed_thresh
    
    # 检测异常快速的运动
    too_fast_mask = velocity > max_speed_thresh
    
    slow_ratio = too_slow_mask.sum() / len(velocity)
    fast_ratio = too_fast_mask.sum() / len(velocity)
    
    # 如果有超过阈值比例的帧速度异常
    if fast_ratio > abnormal_ratio_thresh:
        first_idx = np.where(too_fast_mask)[0][0]
        return "speed_too_fast", first_idx
    
    # 对于过慢的检测，需要排除真正静止的情况
    if slow_ratio > 0.8:  # 超过80%的帧都很慢，可能是异常静止
        # 检查是否整个序列都在很小的范围内移动
        pos_range = np.max(trans, axis=0) - np.min(trans, axis=0)
        if np.max(pos_range) > 0.1:  # 如果总体移动距离>10cm，但速度很慢，则异常
            first_idx = np.where(too_slow_mask)[0][0]
            return "speed_too_slow", first_idx
    
    return False, 0

def detect_center_of_mass_shift(joints3d, com_shift_thresh=0.3, abnormal_ratio_thresh=0.3):
    """
    检测身体重心位置异常
    
    Args:
        joints3d: 关节3D数据 (N, J, 3)
        com_shift_thresh: 重心偏移阈值（米）
        abnormal_ratio_thresh: 异常帧比例阈值
    """
    if joints3d is None or joints3d.shape[0] < 2:
        return False, 0
    
    N = joints3d.shape[0]
    
    # 定义身体关键部位的权重来计算重心
    # 使用可用的关节点，根据人体质量分布设置权重
    joint_weights = {}
    available_joints = []
    
    if PELVIS_IDX < joints3d.shape[1]:
        joint_weights[PELVIS_IDX] = 0.3  # 躯干核心，权重最大
        available_joints.append(PELVIS_IDX)
    
    if HEAD_IDX < joints3d.shape[1]:
        joint_weights[HEAD_IDX] = 0.15  # 头部
        available_joints.append(HEAD_IDX)
    
    if LEFT_KNEE_IDX < joints3d.shape[1]:
        joint_weights[LEFT_KNEE_IDX] = 0.1  # 左膝
        available_joints.append(LEFT_KNEE_IDX)
    
    if RIGHT_KNEE_IDX < joints3d.shape[1]:
        joint_weights[RIGHT_KNEE_IDX] = 0.1  # 右膝
        available_joints.append(RIGHT_KNEE_IDX)
    
    if LEFT_FOOT_IDX < joints3d.shape[1]:
        joint_weights[LEFT_FOOT_IDX] = 0.075  # 左脚
        available_joints.append(LEFT_FOOT_IDX)
    
    if RIGHT_FOOT_IDX < joints3d.shape[1]:
        joint_weights[RIGHT_FOOT_IDX] = 0.075  # 右脚
        available_joints.append(RIGHT_FOOT_IDX)
    
    if len(available_joints) < 3:  # 至少需要3个关节点
        return False, 0
    
    # 计算加权重心
    center_of_mass = np.zeros((N, 3))
    total_weight = sum(joint_weights.values())
    
    for frame in range(N):
        weighted_pos = np.zeros(3)
        for joint_idx in available_joints:
            weight = joint_weights[joint_idx]
            weighted_pos += joints3d[frame, joint_idx] * weight
        center_of_mass[frame] = weighted_pos / total_weight
    
    # 检测重心的异常偏移
    # 1. 重心在XY平面的偏移（相对于脚部支撑点）
    if LEFT_FOOT_IDX < joints3d.shape[1] and RIGHT_FOOT_IDX < joints3d.shape[1]:
        # 计算支撑中心（两脚中点）
        support_center = (joints3d[:, LEFT_FOOT_IDX, :2] + joints3d[:, RIGHT_FOOT_IDX, :2]) / 2
        
        # 重心在XY平面的位置
        com_xy = center_of_mass[:, :2]
        
        # 重心相对于支撑中心的偏移距离
        com_offset = np.linalg.norm(com_xy - support_center, axis=1)
        
        # 检测异常偏移
        abnormal_shift_mask = com_offset > com_shift_thresh
        abnormal_ratio = abnormal_shift_mask.sum() / N
        
        if abnormal_ratio > abnormal_ratio_thresh:
            first_idx = np.where(abnormal_shift_mask)[0][0]
            return "center_of_mass_shift", first_idx
    
    # 2. 重心高度的异常变化
    com_z = center_of_mass[:, 2]
    if len(com_z) > 1:
        com_z_diff = np.abs(np.diff(com_z))
        # 重心高度变化过大
        large_z_change = com_z_diff > 0.3  # 30cm的高度变化
        if large_z_change.sum() > N * 0.1:  # 超过10%的帧有大幅高度变化
            first_idx = np.where(large_z_change)[0][0]
            return "center_of_mass_height_jump", first_idx
    
    return False, 0


def detect_frozen_frames(trans, joints3d=None, position_thresh=0.01, joint_thresh=0.02, 
                        min_frozen_frames=60):
    """
    检测冻结帧（连续多帧完全相同的姿态）
    
    Args:
        trans: 平移数据 (N, 3)
        joints3d: 关节3D数据 (N, J, 3)，可选
        position_thresh: 位置变化阈值 (建议 0.001-0.01)
        joint_thresh: 关节位置变化阈值 (建议 0.005-0.02)
        min_frozen_frames: 最小冻结帧数
        use_adaptive_thresh: 是否使用自适应阈值
    """
    if trans.shape[0] < min_frozen_frames:
        return False, 0
    
    N = trans.shape[0]
    
    # 检测平移的冻结
    trans_diff = np.linalg.norm(np.diff(trans, axis=0), axis=1)  # (N-1,)
    trans_frozen = trans_diff < position_thresh
    
    # 如果有关节数据，也检测关节的冻结
    joint_frozen = None
    if joints3d is not None:
        # 计算所有关节的总体变化
        joints_diff = np.diff(joints3d, axis=0)  # (N-1, J, 3)
        joints_change = np.linalg.norm(joints_diff.reshape(N-1, -1), axis=1)  # (N-1,)
        joint_frozen = joints_change < joint_thresh
    
    # 综合判断：如果位置和关节都冻结
    if joint_frozen is not None:
        frozen_mask = trans_frozen & joint_frozen
    else:
        frozen_mask = trans_frozen
    
    # 查找连续的冻结帧
    frozen_sequences = find_frozen_sequences(frozen_mask, min_frozen_frames)
    
    if frozen_sequences:
        # 返回第一个冻结序列的开始位置
        return "frozen_frames", frozen_sequences[0][0]
    
    return False, 0

def find_frozen_sequences(frozen_mask, min_frozen_frames):
    """查找连续的冻结帧序列"""
    frozen_sequences = []
    current_seq_start = None
    current_seq_length = 0
    
    for i, is_frozen in enumerate(frozen_mask):
        if is_frozen:
            if current_seq_start is None:
                current_seq_start = i
                current_seq_length = 1
            else:
                current_seq_length += 1
        else:
            if current_seq_start is not None and current_seq_length >= min_frozen_frames:
                frozen_sequences.append((current_seq_start, current_seq_length))
            current_seq_start = None
            current_seq_length = 0
    
    # 检查最后一个序列
    if current_seq_start is not None and current_seq_length >= min_frozen_frames:
        frozen_sequences.append((current_seq_start, current_seq_length))
    
    return frozen_sequences


def detect_initial_tilt(global_orient, joints3d=None, tilt_thresh=15.0):
    """
    检测初始orientation是否过于倾斜
    
    Args:
        global_orient: 全局orientation数据，形状为 (N, 3) 或 (N, 3, 3)
        joints3d: 关节点3D坐标，形状为 (N, J, 3)，用于辅助判断
        tilt_thresh: 倾斜角度阈值（度）
    
    Returns:
        bool: 是否检测到初始倾斜
    """
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    
    if global_orient is None or len(global_orient) == 0:
        return False
    
    # 方法1：如果有关节点数据，优先使用关节点计算身体倾斜
    return detect_tilt_from_joints(joints3d, tilt_thresh)


def detect_tilt_from_joints(joints3d, tilt_thresh=15.0):
    """
    通过关节点计算身体倾斜角度（更准确的方法）
    """
    import numpy as np
    
    # 获取初始几帧
    initial_frames = min(5, len(joints3d))
    initial_joints = joints3d[:initial_frames]
    
    # 假设关节点索引（需要根据你的数据调整）
    # 常见索引：spine/torso相关关节
    PELVIS = 0      # 骨盆
    SPINE1 = 3      # 脊柱1
    NECK = 12       # 脖子
    HEAD = 15       # 头部
    
    # 如果关节点不够，返回False
    if joints3d.shape[1] <= max(PELVIS, SPINE1, NECK, HEAD):
        return False
    
    tilt_angles = []
    
    for frame in initial_joints:
        # 计算躯干向量（从骨盆到头部或脖子）
        pelvis = frame[PELVIS]
        
        # 尝试使用不同的上身关节点
        if NECK < joints3d.shape[1]:
            upper_point = frame[NECK]
        elif HEAD < joints3d.shape[1]:
            upper_point = frame[HEAD]
        elif SPINE1 < joints3d.shape[1]:
            upper_point = frame[SPINE1]
        else:
            continue
        
        # 计算躯干向量
        torso_vector = upper_point - pelvis
        
        # 如果向量太短，跳过
        if np.linalg.norm(torso_vector) < 0.1:
            continue
        
        # 计算与竖直方向（z轴向上）的夹角
        vertical = np.array([0, 0, 1])
        cos_angle = np.dot(torso_vector, vertical) / np.linalg.norm(torso_vector)
        cos_angle = np.clip(cos_angle, -1, 1)
        angle_deg = np.degrees(np.arccos(cos_angle))
        
        # 躯干应该是向上的，角度应该接近0度
        # 如果角度大于90度，说明躯干向下，这是异常的
        if angle_deg > 90:
            angle_deg = 180 - angle_deg  # 取补角
        
        tilt_angles.append(angle_deg)
    
    if not tilt_angles:
        return False
    
    avg_tilt = np.mean(tilt_angles)
    return avg_tilt > tilt_thresh


def is_kneeling(joints3d, candidate_mask):
    """
    检测是否为跪姿（排除sitting检测中的跪下情况）
    
    Args:
        joints3d: 关节点3D坐标，形状为 (N, J, 3)
        candidate_mask: 候选帧的mask
        
    Returns:
        kneel_mask: 跪姿的mask
    """
    import numpy as np
    
    if joints3d is None:
        return np.zeros_like(candidate_mask, dtype=bool)
    
    # 假设关节点索引（需要根据你的具体关节点定义调整）
    # 常见的关节点索引：
    HIP_LEFT = 1    # 左髋
    HIP_RIGHT = 2   # 右髋
    KNEE_LEFT = 4   # 左膝
    KNEE_RIGHT = 5  # 右膝
    ANKLE_LEFT = 7  # 左脚踝
    ANKLE_RIGHT = 8 # 右脚踝
    
    # 如果关节点数量不够，返回全False
    if joints3d.shape[1] <= max(HIP_LEFT, HIP_RIGHT, KNEE_LEFT, KNEE_RIGHT, ANKLE_LEFT, ANKLE_RIGHT):
        return np.zeros_like(candidate_mask, dtype=bool)
    
    kneel_mask = np.zeros_like(candidate_mask, dtype=bool)
    
    for i in np.where(candidate_mask)[0]:
        # 获取关键关节点
        left_hip = joints3d[i, HIP_LEFT]
        right_hip = joints3d[i, HIP_RIGHT]
        left_knee = joints3d[i, KNEE_LEFT]
        right_knee = joints3d[i, KNEE_RIGHT]
        left_ankle = joints3d[i, ANKLE_LEFT]
        right_ankle = joints3d[i, ANKLE_RIGHT]
        
        # 计算髋部中心高度
        hip_center_z = (left_hip[2] + right_hip[2]) / 2
        
        # 计算膝盖高度
        left_knee_z = left_knee[2]
        right_knee_z = right_knee[2]
        
        # 计算脚踝高度
        left_ankle_z = left_ankle[2]
        right_ankle_z = right_ankle[2]
        
        # 跪姿特征：
        # 1. 至少一个膝盖接近地面
        # 2. 膝盖高度明显低于髋部
        # 3. 脚踝可能也接近地面（跪坐）或抬起（跪立）
        
        knee_ground_thresh = 0.3  # 膝盖接近地面的阈值
        hip_knee_ratio_thresh = 0.6  # 膝盖相对髋部高度的阈值
        
        # 检查是否有膝盖接近地面
        knee_near_ground = (left_knee_z < knee_ground_thresh) or (right_knee_z < knee_ground_thresh)
        
        # 检查膝盖相对髋部的高度比例
        if hip_center_z > 0:  # 避免除零
            left_knee_ratio = left_knee_z / hip_center_z
            right_knee_ratio = right_knee_z / hip_center_z
            knee_low_ratio = (left_knee_ratio < hip_knee_ratio_thresh) or (right_knee_ratio < hip_knee_ratio_thresh)
        else:
            knee_low_ratio = False
        
        # 额外检查：跪姿时大腿通常是垂直或接近垂直的
        # 计算大腿向量（髋到膝）
        left_thigh = left_knee - left_hip
        right_thigh = right_knee - right_hip
        
        # 计算大腿与垂直方向的夹角
        vertical = np.array([0, 0, -1])  # 向下的垂直向量
        
        def angle_with_vertical(vector):
            if np.linalg.norm(vector) < 1e-6:
                return 90  # 如果向量长度为0，返回90度
            cos_angle = np.dot(vector, vertical) / np.linalg.norm(vector)
            cos_angle = np.clip(cos_angle, -1, 1)
            return np.degrees(np.arccos(cos_angle))
        
        left_thigh_angle = angle_with_vertical(left_thigh)
        right_thigh_angle = angle_with_vertical(right_thigh)
        
        # 跪姿时大腿相对垂直（角度小），但由于z向上，跪姿时大腿向下，角度应该接近180度
        # 或者我们检查大腿是否向下倾斜
        thigh_downward_thresh = 45.0  # 大腿向下倾斜的角度阈值
        # 跪姿时大腿应该是向下的，与向上的z轴夹角较大
        thigh_downward = (left_thigh_angle > (180 - thigh_downward_thresh)) or (right_thigh_angle > (180 - thigh_downward_thresh))
        
        # 综合判断：膝盖接近地面 AND 膝盖相对髋部较低 AND 大腿向下倾斜
        if knee_near_ground and knee_low_ratio and thigh_downward:
            kneel_mask[i] = True
    
    return kneel_mask

def detect_issues(trans, joints3d=None, global_orient=None,
                  airborne_z_thresh=1.3, sitting_z_thresh=0.4,
                  airborne_frame_ratio=0.3, sitting_frame_ratio=0.3,
                  contact_z_thresh=0.15, contact_miss_ratio=0.5,
                  orientation_jump_thresh=30.0, orientation_acc_thresh=20.0,
                  translation_jump_thresh=0.3, translation_acc_thresh=0.2,
                  initial_tilt_thresh=60.0, ignore_sitting=False):  # 新增：初始倾斜角度阈值
    """
    Detect potential motion quality issues: airborne, sitting, missing floor contact, 
    orientation jumps, translation jumps, pose abnormalities, motion smoothness, 
    trembling, ground penetration, initial tilting, and kneeling.
    """
    issues = []
    issues_idx = 1000000

    if trans.shape[0] == 0:
        return issues, issues_idx

    joints3d -= joints3d[0:1, 0:1, :].reshape(1, 1, 3)  # Center the joints to the first frame
    joints3d[:, :, 2] -= joints3d[:, :, 2].min(keepdims=True)  # Adjust Z axis to start from 0 based on all frames
    trans[:, 2] = joints3d[:, 0, 2]  # Adjust Z axis to start from 0 based on all frames

    z = trans[:, 2]
    N = z.shape[0]

    # # -- 新增：初始orientation倾斜检测 --
    # if global_orient is not None:
    #     has_initial_tilt = detect_initial_tilt(global_orient, joints3d=joints3d, tilt_thresh=initial_tilt_thresh)
    #     if has_initial_tilt:
    #         issues.append("initial_tilt")

    # -- 新增：地面穿透检测（优先级最高） --
    has_penetration, penetration_idx = detect_penetration(joints3d, penetration_thresh=-0.2)
    if has_penetration:
        issues.append("ground_penetration")
        issues_idx = min(issues_idx, penetration_idx)

    # -- 新增：异常姿态检测 --
    has_abnormal_pose, abnormal_idx = detect_pose_abnormalities(joints3d)
    if has_abnormal_pose:
        issues.append("abnormal_pose")
        issues_idx = min(issues_idx, abnormal_idx)

    # -- Airborne and Sitting Detection --
    airborne_mask = z > airborne_z_thresh
    sitting_mask = z < sitting_z_thresh

    airborne_ratio = airborne_mask.sum() / N
    sitting_ratio = sitting_mask.sum() / N

    if airborne_ratio > airborne_frame_ratio:
        first_idx = np.where(airborne_mask)[0][0]
        issues.append("airborne")
        issues_idx = min(issues_idx, first_idx)

    if not ignore_sitting and sitting_ratio > sitting_frame_ratio:
        if joints3d is not None:
            # 排除蹲下情况
            squat_mask = is_squatting(joints3d, sitting_mask)
            # 排除跪下情况 - 新增
            kneel_mask = is_kneeling(joints3d, sitting_mask)
            # 真正的坐姿：排除蹲下和跪下
            true_sitting_mask = sitting_mask & (~squat_mask) & (~kneel_mask)
            true_sitting_ratio = true_sitting_mask.sum() / N
            if true_sitting_ratio > sitting_frame_ratio:
                first_idx = np.where(true_sitting_mask)[0][0]
                issues.append("sitting")
                issues_idx = min(issues_idx, first_idx)
        else:
            first_idx = np.where(sitting_mask)[0][0]
            issues.append("sitting")
            issues_idx = min(issues_idx, first_idx)

    # -- Ground Contact Detection（改进版本） --
    if joints3d is not None:
        # 检测所有关节的接触情况
        # 获取所有关节的Z坐标 (N, num_joints)
        all_joints_z = joints3d[:, :, 2]  # (N, J)
        
        # 检查每个关节是否接触地面
        all_joints_contact = all_joints_z < contact_z_thresh  # (N, J)
        
        # 检查每帧是否有任何关节接触地面
        any_contact_per_frame = np.any(all_joints_contact, axis=1)  # (N,) - 每帧是否有任何关节接触
        
        # 计算有接触的帧的比例
        contact_ratio = any_contact_per_frame.sum() / N
        
        # 如果大部分帧都没有任何关节接触地面，则认为有问题（人体完全悬空）
        if contact_ratio < (1 - contact_miss_ratio):
            first_no_contact_idx = np.where(~any_contact_per_frame)[0][0] if np.any(~any_contact_per_frame) else 0
            issues.append("no_ground_contact")
            issues_idx = min(issues_idx, first_no_contact_idx)

    # -- 新增：颤抖检测 --
    has_trembling, trembling_idx = detect_trembling(joints3d)
    if has_trembling:
        issues.append("trembling")
        issues_idx = min(issues_idx, trembling_idx)

    # -- 新增：运动平滑性检测 --
    has_rough_motion, rough_idx = detect_motion_smoothness(trans, joints3d)
    if has_rough_motion:
        issues.append("rough_motion")
        issues_idx = min(issues_idx, rough_idx)

    # -- Orientation Jump Detection（改进版本） --
    if global_orient is not None:
        try:
            if global_orient.shape[-1] == 3:
                from scipy.spatial.transform import Rotation as R
                rotmat = R.from_rotvec(global_orient.reshape(-1, 3)).as_matrix()
            else:
                rotmat = global_orient

            if rotmat.shape[0] > 1:
                relative_rot = np.matmul(rotmat[:-1].transpose(0, 2, 1), rotmat[1:])
                trace_vals = np.trace(relative_rot, axis1=1, axis2=2)
                # 确保trace值在有效范围内
                trace_vals = np.clip(trace_vals, -3, 3)
                relative_angle = np.arccos(np.clip((trace_vals - 1) / 2, -1, 1))
                relative_angle_deg = np.degrees(relative_angle)

                if (relative_angle_deg > orientation_jump_thresh).any():
                    first_idx = np.where(relative_angle_deg > orientation_jump_thresh)[0][0]
                    issues.append("orientation_jump")
                    issues_idx = min(issues_idx, first_idx)

                if len(relative_angle_deg) > 1:
                    angle_diff = np.diff(relative_angle_deg)
                    if (np.abs(angle_diff) > orientation_acc_thresh).any():
                        first_idx = np.where(np.abs(angle_diff) > orientation_acc_thresh)[0][0]
                        issues.append("orientation_acc_jump")
                        issues_idx = min(issues_idx, first_idx)
        except Exception as e:
            print(f"Warning: Error in orientation analysis: {e}")

    # # frozeon frames detection
    # is_frozen, frozen_idx = detect_frozen_frames(trans, joints3d=joints3d)
    # if is_frozen:
    #     issues.append("frozen_frames")

    # # -- Speed Abnormalities Detection（改进版本） --
    # speed_issue, speed_idx = detect_speed_abnormalities(trans, joints3d=joints3d)
    # if speed_issue:
    #     issues.append("speed_abnormality")
    
    # -- Center of Mass Shift Detection（改进版本） --
    com_issue, com_idx = detect_center_of_mass_shift(joints3d)
    if not ignore_sitting and com_issue:
        issues.append("center_of_mass_shift")
        issues_idx = min(issues_idx, com_idx)

    # -- Translation Jump Detection（改进版本） --
    if N > 1:
        trans_diff = np.linalg.norm(trans[1:] - trans[:-1], axis=1)  # (N-1,)
        if (trans_diff > translation_jump_thresh).any():
            first_idx = np.where(trans_diff > translation_jump_thresh)[0][0]
            issues.append("translation_jump")
            issues_idx = min(issues_idx, first_idx)

        # -- Translation Acceleration Detection --
        if len(trans_diff) > 1:
            trans_acc = np.abs(np.diff(trans_diff))  # (N-2,)
            if (trans_acc > translation_acc_thresh).any():
                first_idx = np.where(trans_acc > translation_acc_thresh)[0][0]
                issues.append("translation_acc_jump")
                issues_idx = min(issues_idx, first_idx)

    return issues, issues_idx

# 辅助函数：批量检测多个序列
def batch_detect_issues(sequences, **kwargs):
    """
    批量检测多个运动序列的问题
    
    Args:
        sequences: List of dict, each containing 'trans', 'joints3d', 'global_orient'
        **kwargs: Parameters for detect_issues function
    
    Returns:
        List of (issue_type, frame_idx) tuples
    """
    results = []
    for i, seq in enumerate(sequences):
        trans = seq.get('trans')
        joints3d = seq.get('joints3d')
        global_orient = seq.get('global_orient')
        
        issue_type, frame_idx = detect_issues(trans, joints3d, global_orient, **kwargs)
        results.append((issue_type, frame_idx, i))  # 包含序列索引
    
    return results