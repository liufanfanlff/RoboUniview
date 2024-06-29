import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch import Tensor, nn
from robouniview.data.data_utils import cam, deproject
from robouniview.models.transformers.transformer_utils import inverse_sigmoid

# class cam:
#     def __init__(self,viewMatrix,height,width,fov):
#         self.viewMatrix = viewMatrix
#         self.height = height
#         self.width = width
#         self.fov = fov
        
           
class PETR(nn.Module):
    def __init__(self,
                 hidden_dim = 1024, 
                 depth_step = 0.05,
                 depth_num = 4,
                 depth_start = 0.05,
                 position_range = [-0.2, -0.2, -0.05, 0.2, 0.2, 0.3],
                 ):
        super().__init__()

        self.embed_dims = hidden_dim
        self.depth_step = depth_step
        self.depth_num = depth_num
        self.position_dim = 3 * self.depth_num
        self.position_range = position_range
        self.depth_start = depth_start
        self.position_level = 0
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the transformer head."""
 
        self.adapt_pos3d = nn.Sequential(
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
        )

        self.position_encoder = nn.Sequential(
            nn.Conv2d(self.position_dim, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
        )

    def position_embeding(self, img_feats, cams, masks=None, img_h=84, img_w=84):
        B, C, H, W = img_feats.shape
        
        coords3ds = []
        for cam in cams:
            coords3d = []
            for depth_index in range(self.depth_num):
                depth = self.depth_start + (depth_index*self.depth_step) 
                coords3d_1d = deproject(cam,depth).transpose(1, 0)
                coords3d_1d = np.reshape(
                                        coords3d_1d, (img_h, img_w, 3)
                                    )
                coords3d_1d[..., 0:1] = (coords3d_1d[..., 0:1] - self.position_range[0]) / (self.position_range[3] - self.position_range[0])
                coords3d_1d[..., 1:2] = (coords3d_1d[..., 1:2] - self.position_range[1]) / (self.position_range[4] - self.position_range[1])
                coords3d_1d[..., 2:3] = (coords3d_1d[..., 2:3] - self.position_range[2]) / (self.position_range[5] - self.position_range[2])
                coords3d.append(coords3d_1d)
            coords3d = np.concatenate(coords3d,axis=-1)
            coords3ds.append(coords3d)
        coords3ds = torch.from_numpy(np.array(coords3ds))
        coords3ds = coords3ds.permute(0, 3, 1, 2).to(img_feats.device)
        coords3ds = F.interpolate(
                coords3ds,
                (H, W),
                mode='bilinear'
            )
        coords3ds = inverse_sigmoid(coords3ds.float())
        coords_position_embeding = self.position_encoder(coords3ds)
        return coords_position_embeding#.view(B, N, self.embed_dims, H, W)
    

    def forward(self, mlvl_feats, calibs, pos):
        """Forward function.
       
        """
        gripper_cams = []
        for extrinsic in calibs['rgb_gripper']['extrinsic_matrix']:
            gripper_cam = cam(np.array(extrinsic)*np.array([[1,1,1,1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1]]),84,84,75)
            gripper_cams.append(gripper_cam)
        
        x = mlvl_feats
        coords_position_embeding = self.position_embeding(mlvl_feats, gripper_cams)
        pos_embed = coords_position_embeding
        # pos_embeds = []
        # for i in range(num_cams):
        sin_embed = pos
        sin_embed = self.adapt_pos3d(sin_embed)
        pos_embed = pos_embed + sin_embed

        return pos_embed


# def inverse_sigmoid(x: Tensor, eps: float = 1e-5) -> Tensor:
#     """Inverse function of sigmoid.

#     Args:
#         x (Tensor): The tensor to do the inverse.
#         eps (float): EPS avoid numerical overflow. Defaults 1e-5.
#     Returns:
#         Tensor: The x has passed the inverse function of sigmoid, has the same
#         shape with input.
#     """
#     x = x.clamp(min=0, max=1)
#     x1 = x.clamp(min=eps)
#     x2 = (1 - x).clamp(min=eps)
#     return torch.log(x1 / x2)


