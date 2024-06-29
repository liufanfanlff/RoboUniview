import os
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
from copy import deepcopy
from functools import lru_cache
import cv2
import glob
import random
import open3d as o3d

def build_dataset(data_path, time_sequence_length=6, num_train_episode=-1, num_val_episode=-1,
                  cam_view=["rgb_static"], lang_dir='lang_annotations', target_tasks=['lift_red_block_table'], actions_abs=False):
    train_dataset = CalvinDataset(data_path, 'training', time_sequence_length, num_train_episode, cam_view, lang_dir,
                                  target_tasks,actions_abs)
    val_dataset = CalvinDataset(data_path, 'validation', time_sequence_length, num_val_episode, cam_view, lang_dir,
                                target_tasks,actions_abs)
    return train_dataset, val_dataset


class CalvinDataset(Dataset):
    def __init__(self, data_path, split='training', time_sequence_length=6, num_episode=-1, cam_view=['rgb_static'],
                 lang_dir='lang_annotations', target_tasks=None,actions_abs=False):
        super().__init__()
        self._data_path = data_path
        self._split = split
        lang_info = np.load(os.path.join(self._data_path, split, lang_dir, 'auto_lang_ann.npy'), allow_pickle=True).item()
        self._episode_start_end = lang_info['info']['indx']
        self._language_emb = lang_info['language']['emb']
        self.task = lang_info['language']['task']
        episode_len = [end - start + 1 for start, end in self._episode_start_end]
        self._episode_upper_bound = np.cumsum(episode_len)
        self._episode_lower_bound = self._episode_upper_bound - episode_len
        self._time_sequence_length = time_sequence_length
        for view in cam_view:
            assert view in ['rgb_static', 'rgb_gripper']
        self._views = cam_view
        self.actions_abs = actions_abs

    def __len__(self):
        return len(self._episode_start_end)

    def __getitem__(self, idx):
        
        episode_start_step, episode_end_step = self._episode_start_end[idx]
        clip,clip_files = self.get_sequence(episode_start_step, episode_end_step)
        task = self.task[idx]
        return clip,clip_files,task 

    def get_sequence(self, start_step, end_step):
       
      
        clip = []
        clip_files = []
        for i in range(start_step, end_step+1):
            data, file= self.load_one_step(i)
            clip_files.append(file)
            clip.append(data)
       
        return clip,clip_files

    @lru_cache(maxsize=512, typed=False)
    def load_one_step(self, step_idx):
        
        if 'task_D_D' in str(self._data_path):
            file = os.path.join(self._data_path, self._split,'episode_%07d.npz' % step_idx)
        else:
            files = glob.glob(os.path.join(self._data_path, self._split,"*", 'episode_%07d.npz' % step_idx))
            file = files[0]
        
        return np.load(file),file
       

class CalvinDataset_pcd(Dataset):
    def __init__(self, data_path, split='training', time_sequence_length=6, num_episode=-1, cam_view=['rgb_static'],
                 lang_dir='lang_annotations', target_tasks=None,actions_abs=False):
        super().__init__()
        self._data_path = data_path
        self._split = split
        lang_info = np.load(os.path.join(self._data_path, split, lang_dir, 'auto_lang_ann.npy'), allow_pickle=True).item()
        if target_tasks != None:
            self._episode_start_end = []
            self._language_emb = []
            for i, task in enumerate(lang_info['language']['task']):
                if task in target_tasks:
                    self._episode_start_end.append(lang_info['info']['indx'][i])
                    self._language_emb.append(lang_info['language']['emb'][i])
            
        else:
            
            self._episode_start_end = lang_info['info']['indx']
            self._language_emb = lang_info['language']['emb']
            self.task = lang_info['language']['task']
        if num_episode > 0:
            self._episode_start_end = self._episode_start_end[:num_episode]
        episode_len = [end - start + 1 for start, end in self._episode_start_end]
        self._episode_upper_bound = np.cumsum(episode_len)
        self._episode_lower_bound = self._episode_upper_bound - episode_len
        self._time_sequence_length = time_sequence_length
        for view in cam_view:
            assert view in ['rgb_static', 'rgb_gripper']
        self._views = cam_view
        self.actions_abs = actions_abs

    def __len__(self):
        #return self._episode_upper_bound[-1]
        return len(self._episode_start_end)

    def __getitem__(self, idx):
        
        episode_start_step, episode_end_step = self._episode_start_end[idx]
        clip,clip_files = self.get_sequence(episode_start_step, episode_end_step)
        task = self.task[idx]
        return clip,clip_files,task 

    def get_sequence(self, start_step, end_step):
       
      
        clip = []
        clip_files = []
        for i in range(start_step, end_step):
            data, file= self.load_one_step(i)
            clip_files.append(file)
            clip.append(data)
       
        return clip,clip_files

    @lru_cache(maxsize=512, typed=False)
    def load_one_step(self, step_idx):
        file = os.path.join('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mlm/liufanfan/data/new_calvin_D_1/training',
                            'episode_%07d.pcd' % step_idx)
        
        return o3d.io.read_point_cloud(file),file

if __name__ == '__main__':
    pass