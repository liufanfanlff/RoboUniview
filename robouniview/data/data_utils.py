import pybullet as pb
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as visionF
import torch.nn.functional as F
import torch
import torch.nn as nn

class cam:
    """cam object class for calvin"""
    def __init__(self,viewMatrix,height,width,fov):
        self.viewMatrix = viewMatrix
        self.height = height
        self.width = width
        self.fov = fov
        
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



def deproject(cam, depth_img, homogeneous=False, sanity_check=False):
    """
    Deprojects a pixel point to 3D coordinates
    Args
        point: tuple (u, v); pixel coordinates of point to deproject
        depth_img: np.array; depth image used as reference to generate 3D coordinates
        homogeneous: bool; if true it returns the 3D point in homogeneous coordinates,
                     else returns the world coordinates (x, y, z) position
    Output
        (x, y, z): (3, npts) np.array; world coordinates of the deprojected point
    """
    h, w = depth_img.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    u, v = u.ravel(), v.ravel()

    # Unproject to world coordinates
    T_world_cam = np.linalg.inv(np.array(cam.viewMatrix))
    z = depth_img[v, u]
    foc = cam.height / (2 * np.tan(np.deg2rad(cam.fov) / 2))
    x = (u - cam.width // 2) * z / foc
    y = -(v - cam.height // 2) * z / foc
    z = -z
    ones = np.ones_like(z)

    cam_pos = np.stack([x, y, z, ones], axis=0)
    world_pos = T_world_cam @ cam_pos

    # Sanity check by using camera.deproject function.  Check 2000 points.

    if not homogeneous:
        world_pos = world_pos[:3]

    return world_pos


class OccupancyVFE:
    def __init__(self, voxel_range, voxel_size):
        """
        Args:
        voxel_range = [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        voxel_size = [x_step, y_step, z_step]
        """
        self.voxel_range = np.array(voxel_range)
        self.voxel_size = np.array(voxel_size)
        self.image_range = np.ceil(
            (self.voxel_range[:, 1] - self.voxel_range[:, 0]) / voxel_size
        ).astype(np.int32)
       
    def generate(self, points,rgb):
   
        points_loc = points[:, :3]
        mask = np.logical_and(points_loc > self.voxel_range[:, 0], points_loc < self.voxel_range[:, 1])
        mask = mask.all(axis=-1)
        valid_points = points[mask]
        rgb = rgb[mask]
        valid_loc = valid_points[:, :3]
        coors = np.floor((valid_loc - self.voxel_range[:, 0]) / self.voxel_size)
        coors = np.clip(coors, 0, self.image_range - 1).astype(np.int32)
        occ_label = np.zeros([*self.image_range], dtype=np.float32)
        occ_label[coors[:, 0], coors[:, 1], coors[:, 2]] = 1
        r_label = np.zeros([*self.image_range], dtype=np.float32)
        r_label[coors[:, 0], coors[:, 1], coors[:, 2]] = rgb[:,0]
        g_label = np.zeros([*self.image_range], dtype=np.float32)
        g_label[coors[:, 0], coors[:, 1], coors[:, 2]] = rgb[:,1]
        b_label = np.zeros([*self.image_range], dtype=np.float32)
        b_label[coors[:, 0], coors[:, 1], coors[:, 2]] = rgb[:,2]
        grid_labels = np.stack([occ_label,r_label,g_label,b_label], -1)

        return grid_labels

    @staticmethod
    def decode_occupied_grid(label):
        grid = label[:,:,:,0]
        occupied_loc = np.where(grid > 0.5)
        occupied_points = np.stack(occupied_loc).T
        occupied_rgb = label[occupied_points[:, 0], occupied_points[:, 1], occupied_points[:, 2]][:,1:]

        return occupied_points,occupied_rgb
    
    
    
class ColorJitter_ctm(transforms.ColorJitter):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__(brightness, contrast, saturation, hue)

    def forward(self, img, fn_idx = None, brightness_factor = None, contrast_factor= None, saturation_factor= None, hue_factor= None):
        if brightness_factor is None:
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue
            )
        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = visionF.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = visionF.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = visionF.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = visionF.adjust_hue(img, hue_factor)

        return img,fn_idx,brightness_factor, contrast_factor, saturation_factor, hue_factor
    
    
    
    
class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)

    def forward_traj(self, x):
        n, t, c, h, w = x.size()
        x = x.view(n*t, *x.shape[2:])
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        base_grid = base_grid.unsqueeze(1).repeat(1, t, 1, 1, 1)
        base_grid = base_grid.view(n*t, *base_grid.shape[2:])
        shift = torch.randint(1,
                              2 * self.pad + 1,
                              size=(n*t, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        x = F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)
        x = x.view(n, t, *x.shape[1:])
        return x