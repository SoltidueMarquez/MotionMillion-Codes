from utils.bvh_utils import read_bvh, save_bvh, read_bvh_with_end
from tqdm import tqdm
import torch
import copy
import utils.rotation as rt
import numpy as np
import outer_utils.BVH as BVH
import outer_utils.Animation as Animation
from outer_utils.Quaternions_old import Quaternions
from models.Kinematics import InverseKinematics, InverseKinematics_humdog, JacobianInverseKinematics
from scipy.signal import savgol_filter
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d


L = 24
def alpha(t):
    return 2.0 * t * t * t - 3.0 * t * t + 1


def lerp(a, l, r):
    return (1 - a) * l + a * r

def get_ee_id_by_names(joint_names, raw_bvh=False):
    """ name of foot joints in bvh file """
    if raw_bvh:
        ees = ['RightToe_End', 'LeftToe_End', 'LeftToeBase', 'RightToeBase']
    else:
        ees = ['Left_ankle', 'Left_foot', 'Right_ankle', 'Right_foot'] 

    ee_id = []
    for i, name in enumerate(joint_names):
        if ':' in name:
            joint_names[i] = joint_names[i].split(':')[1]
    for i, ee in enumerate(ees):
        ee_id.append(joint_names.index(ee))
    return ee_id


def get_hand_id_by_names(joint_names, raw_bvh=False):
    """ name of hand joints in bvh file """
    if raw_bvh:
        hand_joints = ['Left_wrist', 'Left_palm','Right_wrist', 'Right_palm']
    else:
        hand_joints = ['Left_wrist', 'Left_palm','Right_wrist', 'Right_palm']
    return [joint_names.index(joint) for joint in hand_joints]


def get_foot_contact(file_name, ref_height=None, thr=50, raw_bvh=False):
    """ thr for foot contact """

    anim, names, _ = BVH.load(file_name)

    ee_ids = get_ee_id_by_names(names, raw_bvh=raw_bvh)
    print(ee_ids)

    glb = Animation.positions_global(anim)  # [T, J, 3]

    ee_pos = glb[:, ee_ids, :]
    ee_velo = ee_pos[1:, ...] - ee_pos[:-1, ...]
    if ref_height is not None:
        ee_velo = torch.tensor(ee_velo) / ref_height
    else:
        ee_velo = torch.tensor(ee_velo)
    ee_velo_norm = torch.norm(ee_velo, dim=-1)
    contact = ee_velo_norm < thr
    contact = contact.int()
    padding = torch.zeros_like(contact)
    contact = torch.cat([padding[:1, :], contact], dim=0)
    return contact.numpy()

def softmax(x, **kw):
    softness = kw.pop("softness", 0.5)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))

def softmin(x, **kw):
    return -softmax(-x, **kw)




def slerp(q0, q1, t):
    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q0 + s1 * q1

def bezier_interpolation(q0, q1, q2, t):
    return slerp(slerp(q0, q1, t), slerp(q1, q2, t), t)

def upsample_and_smooth(data, upsample_factor=2):
    N, J, _ = data.shape
    new_N = (N - 1) * upsample_factor + 1
    upsampled_data = np.zeros((new_N, J, 4))

    for j in range(J):
        for i in range(N - 1):
            q0 = data[i, j]
            q1 = data[i + 1, j]
            for k in range(upsample_factor):
                t = k / upsample_factor
                upsampled_data[i * upsample_factor + k, j] = slerp(q0, q1, t)
        upsampled_data[(N - 1) * upsample_factor, j] = data[N - 1, j]

    smoothed_data = np.zeros_like(upsampled_data)
    for j in range(J):
        for i in range(1, new_N - 1):
            q0 = upsampled_data[i - 1, j]
            q1 = upsampled_data[i, j]
            q2 = upsampled_data[i + 1, j]
            smoothed_data[i, j] = bezier_interpolation(q0, q1, q2, 0.5)
        smoothed_data[0, j] = upsampled_data[0, j]
        smoothed_data[-1, j] = upsampled_data[-1, j]

    return smoothed_data



def upsample_and_smooth_trajectory(trajectory, upsample_factor=2, window_length=5, polyorder=2):
    """
    upsample and smooth trajectory

    params:
    - trajectory: [T, J, 3]
    - upsample_factor: 2
    - window_length: 5
    - polyorder: 2

    return:
    - smoothed_trajectory: [new_T, J, 3]
    """
    N, J, dim = trajectory.shape
    assert dim == 3, "trajectory should be 3D"

    # generate original frame's time index
    original_times = np.arange(N)

    # generate upsampled frame's time index
    upsampled_times = np.linspace(0, N - 1, num=(N - 1) * upsample_factor + 1)

    # initialize upsampled trajectory array
    upsampled_trajectory = np.zeros((len(upsampled_times), J, dim))

    # interpolate each joint's each axis
    for j in range(J):
        for i in range(dim):
            interp_func = interp1d(original_times, trajectory[:, j, i], kind='linear')
            upsampled_trajectory[:, j, i] = interp_func(upsampled_times)

    # smooth the interpolated trajectory
    smoothed_trajectory = savgol_filter(upsampled_trajectory, window_length=window_length, polyorder=polyorder, axis=0)

    return smoothed_trajectory


def smooth_motion_data_with_savgol(anim, joint_names=None, window_length=9, polyorder=3):
    """
    smooth all joints' global position with Savitzky-Golay filter
    """
    from scipy.signal import savgol_filter
    import numpy as np
    import torch

    glb = Animation.positions_global(anim).copy()  # [T, J, 3]
    T, J, _ = glb.shape

    if T < window_length:
        print("too few frames, skip smoothing")
        return anim

    # smooth each joint's X/Y/Z coordinate
    for j in range(J):
        for axis in range(3):
            glb[:, j, axis] = savgol_filter(glb[:, j, axis], window_length, polyorder)

    # update local rotations with smoothed position information (optional)
    # or only use the smoothed result for evaluation, without modifying the original pose
    # here we only use the result for debug/visualization, or we can inverse it back
    return glb

def remove_sliding(input_file, foot_file, output_file,
                     ref_height, input_raw_bvh=False, foot_raw_bvh=False, iterations=50, damping=2.0, silent=False, window_length=5):
    print(input_file)
    anim, name, ftime = BVH.load(input_file)
    anim_with_end = read_bvh_with_end(input_file)
    anim_no_end = read_bvh(input_file)
    fid = get_ee_id_by_names(name, input_raw_bvh)
    print(fid)
    # contact = get_foot_contact(foot_file, ref_height, raw_bvh=foot_raw_bvh)
    fid_l = fid[:2]
    fid_r = fid[2:]
    scale = 1. 
    height_thres = [0.06, 0.03]
    glb = Animation.positions_global(anim)  # [T, J, 3]

    fid_l, fid_r = np.array(fid_l), np.array(fid_r)
    foot_heights = np.minimum(glb[:, fid_l, 1],
                              glb[:, fid_r, 1]).min(axis=1)  # [T, 2] -> [T]
    sort_height = np.sort(foot_heights)
    temp_len = len(sort_height)
    floor_height = np.mean(sort_height[int(0.25*temp_len):int(0.5*temp_len)])
    if floor_height > 0.5: # for motion like swim
        floor_height = 0
    glb[:, :, 1] -= floor_height
    print('floor height:', floor_height)

    
    glb = smooth_motion_data_with_savgol(anim, joint_names=name, window_length=window_length, polyorder=3)
    
    def foot_detect(positions, velfactor, heightfactor):
        """
        detect foot contact points
        
        by calculating the velocity and height of left and right feet:
        1. calculate the change of foot position between adjacent frames (velocity)
        2. check if the height of feet is below the threshold
        3. when the velocity is less than velfactor and the height is less than heightfactor, consider the foot is in contact with the ground
        
        params:
        positions: global position [T, J, 3]
        velfactor: velocity threshold
        heightfactor: height threshold
        
        return:
        feet_l: left foot contact state [T-1, 2] 
        feet_r: right foot contact state [T-1, 2]
        """
        # calculate the velocity square of left foot in xyz three directions
        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        feet_l_h = positions[:-1, fid_l, 1]  # left foot height
        # judge left foot contact state
        feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float32) # 0: not contact, 1: contact

        # calculate the velocity square of right foot in xyz three directions
        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        feet_r_h = positions[:-1, fid_r, 1]  # right foot height
        # judge right foot contact state
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float32) # 0: not contact, 1: contact

        return feet_l, feet_r

    feet_vel_thre = np.array([0.05, 0.2])
    feet_h_thre = np.array(height_thres) * scale
    feet_l, feet_r = foot_detect(glb, velfactor=feet_vel_thre, heightfactor=feet_h_thre)
    foot = np.concatenate([feet_l, feet_r], axis=-1).transpose(1, 0)  # [4, T-1]
    foot = np.concatenate([foot, foot[:, -1:]], axis=-1)
    contact = foot.transpose(1, 0)  # [T, 4]

    
    
    
    print('contact shape:', contact.shape)
    T = glb.shape[0]


    # this code is used to process foot sliding problem, through the following steps:
    
    def process_foot_sliding(glb, fid, contact, T, L):
        """
        process foot sliding problem
        
        params:
        glb: global position [T, J, 3]
        fid: foot joint index list
        contact: contact state [T, 4]
        T: total frames
        L: interpolation window size
        
        return:
        glb: processed global position
        """
        # 1. traverse each foot joint
        for i, fidx in enumerate(fid):  # fidx: foot joint index
            fixed = contact[:, i]  # [T] contact state sequence of this joint
            s = 0
            # 2. process continuous contact frames, set the position of these frames to the average position
            while s < T:
                # find the start position of the next contact frame
                while s < T and fixed[s] == 0: # if not contact -> find the first contact frame
                    s += 1
                if s >= T:
                    break
                t = s
                # calculate the average position of continuous contact frames
                avg = glb[t, fidx].copy()
                while t + 1 < T and fixed[t + 1] == 1:
                    t += 1
                    avg += glb[t, fidx].copy()
                avg /= (t - s + 1)

                # set the position of continuous contact frames to the average position
                for j in range(s, t + 1):
                    glb[j, fidx] = avg.copy()
                s = t + 1

            # 3. process non-contact frames, make the motion smoother through interpolation
            for s in range(T):
                if fixed[s] == 1:
                    continue
                l, r = None, None
                consl, consr = False, False
                # find the nearest contact frame forward
                for k in range(L):
                    if s - k - 1 < 0:
                        break
                    if fixed[s - k - 1]:
                        l = s - k - 1
                        consl = True
                        break
                # find the nearest contact frame backward
                for k in range(L):
                    if s + k + 1 >= T:
                        break
                    if fixed[s + k + 1]:
                        r = s + k + 1
                        consr = True
                        break
                # interpolate according to the situation of front and back contact frames
                if not consl and not consr:
                    continue
                if consl and consr:
                    # interpolate in both directions when there are contact frames on both sides
                    litp = lerp(alpha(1.0 * (s - l + 1) / (L + 1)),
                                glb[s, fidx], glb[l, fidx])
                    ritp = lerp(alpha(1.0 * (r - s + 1) / (L + 1)),
                                glb[s, fidx], glb[r, fidx])
                    itp = lerp(alpha(1.0 * (s - l + 1) / (r - l + 1)),
                               ritp, litp)
                    glb[s, fidx] = itp.copy()
                    continue
                if consl:
                    # interpolate forward when there is only a contact frame in front
                    litp = lerp(alpha(1.0 * (s - l + 1) / (L + 1)),
                                glb[s, fidx], glb[l, fidx])
                    glb[s, fidx] = litp.copy()
                    continue
                if consr:
                    # interpolate backward when there is only a contact frame in back
                    ritp = lerp(alpha(1.0 * (r - s + 1) / (L + 1)),
                                glb[s, fidx], glb[r, fidx])
                    glb[s, fidx] = ritp.copy()
        
        return glb

    
    
    
    # call function to process foot sliding
    glb = process_foot_sliding(glb, fid, contact, T, L)


    hid = get_hand_id_by_names(name, input_raw_bvh)
    hid_l = hid[:2]
    hid_r = hid[2:]
    hid_l, hid_r = np.array(hid_l), np.array(hid_r) 
    
    def hand_detect(positions, velfactor, heightfactor):
        """
        detect hand contact points
        
        by calculating the velocity and height of left and right hands:
        1. calculate the change of hand position between adjacent frames (velocity)
        2. check if the height of hands is below the threshold
        3. when the velocity is less than velfactor and the height is less than heightfactor, consider the hand is in contact with the ground
        
        params:
        positions: global position [T, J, 3]
        velfactor: velocity threshold
        heightfactor: height threshold
        
        return:
        hand_l: left hand contact state [T-1, 2] 
        hand_r: right hand contact state [T-1, 2]
        """
        # calculate the velocity square of left hand in xyz three directions
        hand_l_x = (positions[1:, hid_l, 0] - positions[:-1, hid_l, 0]) ** 2
        hand_l_y = (positions[1:, hid_l, 1] - positions[:-1, hid_l, 1]) ** 2
        hand_l_z = (positions[1:, hid_l, 2] - positions[:-1, hid_l, 2]) ** 2
        hand_l_h = positions[:-1, hid_l, 1]  # left hand height
        # judge left hand contact state
        hand_l = (((hand_l_x + hand_l_y + hand_l_z) < velfactor) & (hand_l_h < heightfactor)).astype(np.float32) # 0: not contact, 1: contact

        # calculate the velocity square of right hand in xyz three directions
        hand_r_x = (positions[1:, hid_r, 0] - positions[:-1, hid_r, 0]) ** 2
        hand_r_y = (positions[1:, hid_r, 1] - positions[:-1, hid_r, 1]) ** 2
        hand_r_z = (positions[1:, hid_r, 2] - positions[:-1, hid_r, 2]) ** 2
        hand_r_h = positions[:-1, hid_r, 1]  # right hand height
        # judge right hand contact state
        hand_r = (((hand_r_x + hand_r_y + hand_r_z) < velfactor) & (hand_r_h < heightfactor)).astype(np.float32) # 0: not contact, 1: contact

        return hand_l, hand_r
    
    hand_vel_thre = np.array([0.05, 0.2])
    hand_h_thre = np.array(height_thres) * scale
    hand_l, hand_r = hand_detect(glb, velfactor=hand_vel_thre, heightfactor=hand_h_thre)
    hand = np.concatenate([hand_l, hand_r], axis=-1).transpose(1, 0)  # [4, T-1]
    hand = np.concatenate([hand, hand[:, -1:]], axis=-1)
    contact = hand.transpose(1, 0)  # [T, 4]
    
    def process_hand_sliding(glb, hid, contact, T, L):
        """
        process hand sliding problem
        
        params:
        glb: global position [T, J, 3]
        hid: hand joint index list
        contact: contact state [T, 4]
        T: total frames
        L: interpolation window size
        
        return:
        glb: processed global position
        """
        for i, fidx in enumerate(hid):
            fixed = contact[:, i]
            s = 0
            while s < T:
                while s < T and fixed[s] == 0:
                    s += 1
                if s >= T:
                    break
                t = s
                avg = glb[t, fidx].copy()
                while t + 1 < T and fixed[t + 1] == 1:
                    t += 1
                    avg += glb[t, fidx].copy()
                avg /= (t - s + 1)
                for j in range(s, t + 1):
                    glb[j, fidx] = avg.copy()
                s = t + 1
            for s in range(T):
                if fixed[s] == 1:
                    continue
                l, r = None, None
                consl, consr = False, False
                for k in range(L):
                    if s - k - 1 < 0:
                        break
                    if fixed[s - k - 1]:
                        l = s - k - 1
                        consl = True
                        break
                for k in range(L):
                    if s + k + 1 >= T:
                        break
                    if fixed[s + k + 1]:
                        r = s + k + 1
                        consr = True
                        break
                if not consl and not consr:
                    continue
                if consl and consr:
                    litp = lerp(alpha(1.0 * (s - l + 1) / (L + 1)),
                                glb[s, fidx], glb[l, fidx])
                    ritp = lerp(alpha(1.0 * (r - s + 1) / (L + 1)),
                                glb[s, fidx], glb[r, fidx])
                    itp = lerp(alpha(1.0 * (s - l + 1) / (r - l + 1)),
                               ritp, litp)
                    glb[s, fidx] = itp.copy()
                    continue
                if consl:
                    litp = lerp(alpha(1.0 * (s - l + 1) / (L + 1)),
                                glb[s, fidx], glb[l, fidx])
                    glb[s, fidx] = litp.copy()
                    continue
                if consr:
                    ritp = lerp(alpha(1.0 * (r - s + 1) / (L + 1)),
                                glb[s, fidx], glb[r, fidx])
                    glb[s, fidx] = ritp.copy()
        return glb
    
    anim = anim.copy()
    targetmap = {}

    for j in range(glb.shape[1]):
        targetmap[j] = glb[:, j]

    glb = torch.tensor(glb, dtype=torch.float)
    print('begin to optimize')
    ik = JacobianInverseKinematics(anim, targetmap, iterations=iterations, damping=damping, silent=silent)
    ik()
    anim = ik.animation
    anim.positions[:, :, 1] -= floor_height
    anim_no_end.quats = upsample_and_smooth(anim.rotations.qs, upsample_factor=2)
    anim_no_end.pos = upsample_and_smooth_trajectory(anim.positions, upsample_factor=2)
    end_offset = anim_with_end.offsets[anim_with_end.endsite, :]
    save_bvh(output_file, anim_no_end, anim_no_end.bones, ftime/2,
             order='zyx', with_end=False,
             end_offset=end_offset)



def normalize(x):
    return x/torch.norm(x, dim=-1, p=2, keepdim=True)


