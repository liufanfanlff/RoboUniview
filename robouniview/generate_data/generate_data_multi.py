""" Main  """
import sys
import os  


env = os.environ    
current_path = os.getcwd()  
robouniview_path =  current_path  
env['PATH'] = env['PATH'] + ':'+  robouniview_path
sys.path.append(robouniview_path)
new_path = '.../new_calvin_ABC_D_multi/'
sys.path.append(new_path+'pjt/calvin/calvin_models')
sys.path.append(new_path+'pjt/calvin/calvin_env')
sys.path.append(new_path+'pjt/calvin/calvin_env/tacto_env')
dataset_path = '.../CALVIN/task_ABC_D_'


from pathlib import Path
from robouniview.generate_data.data import CalvinDataset
import argparse
import copy
import glob
import os
import random
from collections import OrderedDict
import numpy as np
import torch
import wandb
from moviepy.editor import ImageSequenceClip
from robouniview.eval.eval_utils import make_env
import cv2 
import yaml
import os
from xml.etree import ElementTree as ET
from pathlib import Path
from omegaconf import OmegaConf
from multiprocessing import Process
import pybullet as pb
import open3d 


dataset_path_env = new_path + 'env_config/'
new_calvin = new_path + 'training_npz_pcd_new/'
if not os.path.exists(new_calvin):
    os.makedirs(new_calvin)
urdf_path = new_path +'pjt/calvin/calvin_env/data/franka_panda/panda_longer_finger.urdf'
config_path = dataset_path_env + 'validation/.hydra/merged_config.yaml'

cam_config = {'static': {'_target_': 'calvin_env.camera.static_camera.StaticCamera', 'name': 'static', 'fov': 10, 'aspect': 1, 
                        'nearval': 0.01, 'farval': 10, 'width': 200, 'height': 200, 
                        'look_at': [-0.02262423511594534, -0.0302329882979393, 0.3920000493526459], 
                        'look_from': [-2.871459009488717, 1.1666602199425595, 4.555159848480571], 
                        'up_vector': [0.4041403970338857, 0.22629790978217404, 0.8862616969685161]}, 
            'gripper': {'_target_': 'calvin_env.camera.gripper_camera.GripperCamera', 'name': 'gripper', 'fov': 75, 'aspect': 1, 
                        'nearval': 0.01, 'farval': 2, 'width': 84, 'height': 84},
            'xyzrpy':'\t  <origin xyz="-0.1 0 0" rpy="1.3  0 -1.57"/>\n'}


    
def get_gripper_camera_view_matrix(cam):
    camera_ls = pb.getLinkState(
        bodyUniqueId=cam.robot_uid,
        linkIndex=cam.gripper_cam_link,
        physicsClientId=cam.cid
    )
    camera_pos, camera_orn = camera_ls[:2]
    cam_rot = pb.getMatrixFromQuaternion(camera_orn)
    cam_rot = np.array(cam_rot).reshape(3, 3)
    cam_rot_y, cam_rot_z = cam_rot[:, 1], cam_rot[:, 2]
    # camera: eye position, target position, up vector
    view_matrix = pb.computeViewMatrix(
        camera_pos, camera_pos + cam_rot_y, -cam_rot_z
    )
    return view_matrix




def generate_npz(dataset_path_env,data,clip_files,cam_config,ii):
    img_queue = []
    env = make_env(dataset_path_env)
    for i,data_slice in enumerate(data):
        robot_obs = data_slice['robot_obs']
        scene_obs = data_slice['scene_obs']
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        obs = env.get_obs()
        static_cam = env.cameras[0]
        gripper_cam = env.cameras[1]
        gripper_cam.viewMatrix = get_gripper_camera_view_matrix(gripper_cam)
        static_extrinsic = np.array(static_cam.viewMatrix).reshape((4, 4)).T
        gripper_extrinsic = np.array(gripper_cam.viewMatrix).reshape((4, 4)).T
        static_foc = static_cam.height / (2 * np.tan(np.deg2rad(static_cam.fov) / 2))
        gripper_foc = gripper_cam.height / (2 * np.tan(np.deg2rad(gripper_cam.fov) / 2))
        static_intrinsic = np.array([[static_foc , 0.0, static_cam.height/2], [0.0, static_foc , static_cam.height/2], [0.0, 0.0, 1.0]])
        gripper_intrinsic = np.array([[gripper_foc , 0.0, gripper_cam.height/2], [0.0, gripper_foc , gripper_cam.height/2], [0.0, 0.0, 1.0]])

        calib = {'rgb_static':{'extrinsic_matrix':static_extrinsic,
                                'intrinsic_matrix':static_intrinsic,
                                'distCoeffs_matrix':np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,])},
                'rgb_gripper':{'extrinsic_matrix':gripper_extrinsic,
                                'intrinsic_matrix':gripper_intrinsic,
                                'distCoeffs_matrix':np.array([0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0,])}}
        file = clip_files[i]
        basename = os.path.basename(file)
        iii = ii//2000
        if 'task_D_D' in dataset_path :
            newfile = new_calvin + basename

        else:
            if not os.path.exists(new_calvin +f"training_{iii}/"):
                os.makedirs(new_calvin +f"training_{iii}/")
            
            newfile = new_calvin + f"training_{iii}/" + basename
        max_data_fetch_iteration = 4
        for _ in range(max_data_fetch_iteration):
            try:
                pass
                np.savez(newfile, actions = data_slice['actions'], 
                                rel_actions = data_slice['rel_actions'],
                                robot_obs = data_slice['robot_obs'],
                                scene_obs = data_slice['scene_obs'],
                                rgb_static = obs['rgb_obs']['rgb_static'],
                                rgb_gripper = obs['rgb_obs']['rgb_gripper'],
                                rgb_tactile = data_slice['rgb_tactile'],
                                depth_static = obs['depth_obs']['depth_static'],
                                depth_gripper = obs['depth_obs']['depth_gripper'],
                                depth_tactile = data_slice['depth_tactile'],
                                cam_config = cam_config,
                                calib = calib,
                )
            except Exception:
                print(
                            "save warning:"+newfile
                        )
        print(newfile)

    #     rgb_static_img = obs['rgb_obs']['rgb_static']
    #     rgb_gripper_img = obs['rgb_obs']['rgb_gripper']
    #     rgb_gripper_img = cv2.resize(rgb_gripper_img, (200,200))
    #     rgb_static_img_org = data_slice['rgb_static']
    #     rgb_gripper_img_org = data_slice['rgb_gripper']
    #     rgb_gripper_img_org = cv2.resize(rgb_gripper_img_org, (200,200))
    #     img1 = cv2.hconcat((rgb_static_img, rgb_gripper_img))
    #     img2 = cv2.hconcat((rgb_static_img_org, rgb_gripper_img_org))
    #     img = cv2.vconcat((img1, img2))
    #     img_queue.append(img)
    # img_clip = ImageSequenceClip(img_queue, fps=30)
    # img_clip.write_gif(f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-vacv/liufanfan/data/debug{ii}.gif', fps=30)


    del env
    


    

def run_command(dataset_path_env,data,clip_files,cam_config,ii):
    #generate_pcd(dataset_path_env,data,clip_files,cam_config,ii)
    generate_npz(dataset_path_env,data,clip_files,cam_config,ii)


hist_d = np.array([ [4.410e+02, 1.117e+03, 7.200e+02],
                    [0.000e+00, 0.000e+00, 0.000e+00],
                    [0.000e+00, 0.000e+00, 6.000e+00],
                    [0.000e+00, 0.000e+00, 6.200e+01],
                    [0.000e+00, 0.000e+00, 3.940e+02],
                    [0.000e+00, 1.600e+01, 4.077e+03],
                    [3.400e+01, 1.090e+02, 5.815e+03],
                    [3.300e+02, 8.640e+02, 5.873e+03],
                    [4.020e+02, 3.374e+03, 6.357e+03],
                    [3.650e+02, 4.061e+03, 2.378e+03],
                    [1.378e+03, 4.548e+03, 6.880e+02],
                    [2.882e+03, 4.472e+03, 1.040e+02],
                    [2.955e+03, 4.222e+03, 6.000e+01],
                    [2.828e+03, 3.957e+03, 1.090e+02],
                    [2.394e+03, 6.110e+02, 2.420e+02],
                    [4.253e+03, 9.700e+01, 1.750e+02],
                    [3.273e+03, 1.360e+02, 1.870e+02],
                    [4.279e+03, 2.630e+02, 3.090e+02],
                    [2.238e+03, 6.950e+02, 7.470e+02],
                    [4.690e+02, 3.730e+02, 4.380e+02],
                    [8.710e+02, 7.740e+02, 8.690e+02],
                    [9.730e+02, 8.760e+02, 9.690e+02],
                    [6.070e+02, 5.230e+02, 5.900e+02],
                    [1.717e+03, 1.669e+03, 1.677e+03],
                    [3.192e+03, 3.163e+03, 3.142e+03]])

hist_a = np.array([[  315.,   785.,   254.],
                    [    0.,     0.,     0.],
                    [    0.,     0.,   372.],
                    [    0.,    28.,  6560.],
                    [    0.,   933., 13214.],
                    [    0.,  6340.,  6445.],
                    [   77.,  9237.,    88.],
                    [  554.,  8593.,   346.],
                    [ 2035.,  2241.,   335.],
                    [ 3991.,   383.,   302.],
                    [ 5579.,   778.,   827.],
                    [ 4987.,    45.,    63.],
                    [ 5553.,    76.,    29.],
                    [ 4706.,   118.,    72.],
                    [ 1313.,    43.,   120.],
                    [   47.,    16.,    85.],
                    [   73.,    41.,    86.],
                    [  163.,   125.,   180.],
                    [  660.,   626.,   675.],
                    [  588.,   573.,   588.],
                    [ 1276.,  1249.,  1273.],
                    [  601.,   548.,   606.],
                    [  378.,   313.,   396.],
                    [ 2012.,  1962.,  2024.],
                    [ 1954.,  1907.,  1945.]])

hist_b = np.array([[ 999.,  832.,  852.],
                    [   0.,    0.,    0.],
                    [   0.,    0.,    0.],
                    [   0.,    0.,    0.],
                    [   0.,    0.,    0.],
                    [   0.,    0.,    0.],
                    [  49.,   49.,   49.],
                    [ 465.,  464.,  453.],
                    [ 357.,  356.,  357.],
                    [ 430.,  432., 1678.],
                    [ 676.,  674., 5111.],
                    [ 109.,  111., 1590.],
                    [  91.,   97., 2456.],
                    [ 238.,  866., 4731.],
                    [  81., 3480., 6724.],
                    [  24., 2053., 4861.],
                    [ 108., 1459.,  164.],
                    [2940., 1707.,  149.],
                    [3044., 1654.,  532.],
                    [1080., 5781.,  471.],
                    [1598., 5341.,  691.],
                    [2804., 5877.,  824.],
                    [ 928.,  799.,  405.],
                    [2444., 1565., 1610.],
                    [8724., 2460., 2503.]])

hist_c = np.array([[ 783.,  728.,  860.],
                    [   0.,    0.,    0.],
                    [   0.,    0.,    0.],
                    [   0.,    0.,  654.],
                    [   0.,    0., 4657.],
                    [   0.,  533., 5617.],
                    [  15., 2031., 6197.],
                    [ 328., 1935., 2984.],
                    [ 300., 3495.,  277.],
                    [ 695., 2690.,  296.],
                    [1667., 4324.,  417.],
                    [2852., 4010.,  120.],
                    [2518., 2301.,   68.],
                    [1817.,  279.,   96.],
                    [2280.,   87.,  223.],
                    [2444.,  108.,  180.],
                    [3400.,  165.,  194.],
                    [3245.,  324.,  350.],
                    [1538.,  749.,  740.],
                    [ 258.,  217.,  236.],
                    [2638., 2591., 2616.],
                    [1077., 1032., 1070.],
                    [ 588.,  648.,  590.],
                    [2445., 2430., 2446.],
                    [4276., 4276., 4276.]])
def generte_data():
    dataset = CalvinDataset(
        Path(dataset_path) ,
       )
    len_dataset = len(dataset)

    Process_num = 120
    Process_num_t = Process_num
    processes = []
    for ii,(data,clip_files,task) in enumerate(dataset):

        if ii<0:
            pass
        else:

            img = data[0]['rgb_static']
            hist0 = cv2.calcHist([img],[0],None,[25],[0,230])
            hist1 = cv2.calcHist([img],[1],None,[25],[0,230])
            hist2 = cv2.calcHist([img],[2],None,[25],[0,230])
            hist = np.concatenate([hist0,hist1,hist2], axis=1)

            d_hist = [abs(hist-hist_a).mean(),abs(hist-hist_b).mean(),abs(hist-hist_c).mean(),abs(hist-hist_d).mean()]
            min_value = min(d_hist) 
            min_index = d_hist.index(min_value)
            

            if 'task_D_D' in dataset_path:
                pink = ['blocks/block_pink_small.urdf','blocks/block_pink_middle.urdf','blocks/block_pink_big.urdf']
                blue = ['blocks/block_blue_small.urdf','blocks/block_blue_middle.urdf','blocks/block_blue_big.urdf']
                red = ['blocks/block_red_small.urdf','blocks/block_red_middle.urdf','blocks/block_red_big.urdf']    
                table = 'calvin_table_D/urdf/calvin_table_D.urdf'
                scene = 'calvin_scene_D'
                movable_objects = {
                        'block_red': {'file':random.choice(red), 'initial_pos': 'any', 'initial_orn': 'any'},
                        'block_blue': {'file':random.choice(blue), 'initial_pos': 'any', 'initial_orn': 'any'},
                        'block_pink': {'file':random.choice(pink), 'initial_pos': 'any', 'initial_orn': 'any'}}

            else :
                pink = ['blocks/block_pink_small.urdf','blocks/block_pink_middle.urdf','blocks/block_pink_big.urdf']
                blue = ['blocks/block_blue_small.urdf','blocks/block_blue_middle.urdf','blocks/block_blue_big.urdf']
                red = ['blocks/block_red_small.urdf','blocks/block_red_middle.urdf','blocks/block_red_big.urdf']

                if min_index == 0:
                    table = 'calvin_table_A/urdf/calvin_table_A.urdf'
                    scene = 'calvin_scene_A'
                    movable_objects = {
                        'block_red': {'file':random.choice(pink), 'initial_pos': 'any', 'initial_orn': 'any'},
                        'block_blue': {'file':random.choice(blue), 'initial_pos': 'any', 'initial_orn': 'any'},
                        'block_pink': {'file':random.choice(red), 'initial_pos': 'any', 'initial_orn': 'any'}}

                if min_index == 1:
                    table = 'calvin_table_B/urdf/calvin_table_B.urdf'
                    scene = 'calvin_scene_B'
                    movable_objects = {
                        'block_red': {'file':random.choice(red), 'initial_pos': 'any', 'initial_orn': 'any'},
                        'block_blue': {'file':random.choice(blue), 'initial_pos': 'any', 'initial_orn': 'any'},
                        'block_pink': {'file':random.choice(pink), 'initial_pos': 'any', 'initial_orn': 'any'}}

                if min_index == 2:
                    table = 'calvin_table_C/urdf/calvin_table_C.urdf'
                    scene = 'calvin_scene_C'
                    movable_objects = {
                        'block_red': {'file':random.choice(blue), 'initial_pos': 'any', 'initial_orn': 'any'},
                        'block_blue': {'file':random.choice(red), 'initial_pos': 'any', 'initial_orn': 'any'},
                        'block_pink': {'file':random.choice(pink), 'initial_pos': 'any', 'initial_orn': 'any'}}

                if min_index == 3:
                    
                    table = 'calvin_table_D/urdf/calvin_table_D.urdf'
                    scene = 'calvin_scene_D'
                    movable_objects = {
                        'block_red': {'file':random.choice(red), 'initial_pos': 'any', 'initial_orn': 'any'},
                        'block_blue': {'file':random.choice(blue), 'initial_pos': 'any', 'initial_orn': 'any'},
                        'block_pink': {'file':random.choice(pink), 'initial_pos': 'any', 'initial_orn': 'any'}}
       
            with open(config_path, "r") as infile:
                render_conf = yaml.safe_load(infile)


            cam_list = [{"a":[30,151], "z": 4.55, "f":[10,13],"r":2, "u":[0,361] },
                        {"a":[30,151], "z": 2.55, "f":[12,13],"r":3, "u":[0,361] },
                        {"a":[30,151], "z": 2.55, "f":[10,13],"r":3.5, "u":[0,361] },
                        {"a":[30,151], "z": 4.55, "f":[10,13],"r":1, "u":[0,361] },
                        {"a":[30,151], "z": 3.55, "f":[11,14],"r":3, "u":[0,361] }]
            cam = random.choice(cam_list)

            a = random.randrange(cam['a'][0],cam['a'][1])

            a = (a/180)*np.pi
            x = np.cos(-a) * 3
            y = np.sin(-a) * 3

            cam_config['static']['look_from'] = [float(x), float(y), cam['z']]
            cam_config['static']['fov'] = random.randrange(cam['f'][0],cam['f'][1])

            u = random.randrange(cam['u'][0],cam['u'][1])

            u = (u/180)*np.pi

            global_up = np.array([np.cos(-u), np.sin(-u), 0])

            look_from = np.array(cam_config['static']['look_from'])
            look_at = np.array(cam_config['static']['look_at'])
            forward_vector = look_at - look_from
            forward_vector = forward_vector / np.linalg.norm(forward_vector)  
            right_vector = np.cross(global_up, forward_vector)
            right_vector = right_vector / np.linalg.norm(right_vector) 
            up_vector = np.cross(forward_vector, right_vector)
            cam_config['static']['up_vector'] = [float(up_vector[0]), float(up_vector[1]), float(up_vector[2])]
            cam_config['gripper']['fov'] = random.randrange(40,100)
            q = random.randrange(0,361)
            q = (q/180)*np.pi
            x = np.sin(q) *0.27
            y = np.cos(q) *0.27
            cam_config['xyzrpy'] = f'\t  <origin xyz="-0.1 0 0" rpy="{1.57-y}  {0-x} -{1.57+q}"/>\n'
            render_conf['cameras']['static'] = cam_config['static']
            render_conf['cameras']['gripper'] = cam_config['gripper']
            render_conf['scene']['objects']['fixed_objects']['table']['file'] = table
            render_conf['scene']['objects']['movable_objects'] = movable_objects
            render_conf = dict(render_conf)
            with open(Path(config_path), 'w') as f:
                yaml.dump(render_conf, f, default_flow_style=None, sort_keys=False)

            with open(urdf_path, "r") as file:
                content = file.readlines()
            content[342] = cam_config['xyzrpy']
            with open(urdf_path, "w") as file:
                for content_i in content:
                    file.write(content_i)

            p = Process(target=run_command, args=(dataset_path_env,data,clip_files,cam_config,ii))
            p.start()
            processes.append(p)
            if Process_num_t > 0:
                Process_num_t-=1
            else:
                for p in processes:
                    p.join()
                
                Process_num_t = Process_num
                processes = []
                print('++++++++++++++++'+ str(ii)+"/"+str(len_dataset)+'++++++++++++++++')

if __name__ == "__main__":
    
    generte_data()
