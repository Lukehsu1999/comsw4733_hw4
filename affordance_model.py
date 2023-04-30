from typing import Tuple, Optional, Dict

import numpy as np
from matplotlib import cm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

from common import draw_grasp
from collections import deque

def get_gaussian_scoremap(
        shape: Tuple[int, int], 
        keypoint: np.ndarray, 
        sigma: float=1, dtype=np.float32) -> np.ndarray:
    """
    Generate a image of shape=:shape:, generate a Gaussian distribtuion
    centered at :keypont: with standard deviation :sigma: pixels.
    keypoint: shape=(2,)
    """
    coord_img = np.moveaxis(np.indices(shape),0,-1).astype(dtype)
    sqrt_dist_img = np.square(np.linalg.norm(
        coord_img - keypoint[::-1].astype(dtype), axis=-1))
    scoremap = np.exp(-0.5/np.square(sigma)*sqrt_dist_img)
    return scoremap

class AffordanceDataset(Dataset):
    """
    Transformational dataset.
    raw_dataset is of type train.RGBDataset
    """
    def __init__(self, raw_dataset: Dataset):
        super().__init__()
        self.raw_dataset = raw_dataset
    
    def __len__(self) -> int:
        return len(self.raw_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Transform the raw RGB dataset element into
        training targets for AffordanceModel.
        return: 
        {
            'input': torch.Tensor (3,H,W), torch.float32, range [0,1]
            'target': torch.Tensor (1,H,W), torch.float32, range [0,1]
        }
        Note: self.raw_dataset[idx]['rgb'] is torch.Tensor (H,W,3) torch.uint8
        """
        # checkout train.RGBDataset for the content of data
        data = self.raw_dataset[idx]
        # TODO: (problem 2) complete this method and return the correct input and target as data dict
        # Hint: Use get_gaussian_scoremap
        # Hint: https://imgaug.readthedocs.io/en/latest/source/examples_keypoints.html
        # ===============================================================================
        # Post 787
        # might need to numpy the tensors below
        rgb = data['rgb'].numpy()
        center_point = data['center_point'].numpy()
        angle = data['angle'].numpy()

        # Step 1: Discretize angles into 8 bins with np.argmin
        angle_bin = np.argmin(np.abs(np.arange(8) * 22.5 - angle))
        angle = angle_bin * 22.5

        # Step 2: use center_point and use KeypointsOnImage, iaa.Rotate to get the rotated rgb image and rotated keypoints. -> data['input']
        kps = KeypointsOnImage([Keypoint(x=center_point[0], y=center_point[1])], shape=rgb.shape)
        seq = iaa.Sequential([iaa.Affine(rotate=angle)])
        rgb = seq(image=rgb)
        kps = seq(keypoints=kps)

        # Step 3: use get_gaussian_scoremap to get the target -> data['target']
        target = get_gaussian_scoremap(rgb.shape[:2], np.array([kps[0].x, kps[0].y]), sigma=1)

        # Step 4: normalize rgb to [0,1]
        rgb = rgb / 255
        target = target / np.max(target)

        # Step 5: convert rgb and target to torch.Tensor
        rgb = torch.from_numpy(rgb).permute(2,0,1).float()
        target = torch.from_numpy(target).unsqueeze(0).float()
        
        # Step 6: return data dict
        data = dict(input=rgb, target=target)

        return data 


class AffordanceModel(nn.Module):
    def __init__(self, n_channels: int=3, n_classes: int=1, n_past_actions: int=0, **kwargs):
        """
        A simplified U-Net with twice of down/up sampling and single convolution.
        ref: https://arxiv.org/abs/1505.04597, https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
        :param n_channels (int): number of channels (for grayscale 1, for rgb 3)
        :param n_classes (int): number of segmentation classes (num objects + 1 for background)
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = nn.Sequential(
            # For simplicity:
            #     use padding 1 for 3*3 conv to keep the same Width and Height and ease concatenation
            nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential( 
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            )
        self.outc = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1)
        # hack to get model device
        self.dummy_param = nn.Parameter(torch.empty(0))
        self.past_actions = deque(maxlen=n_past_actions)

    @property
    def device(self) -> torch.device:
        return self.dummy_param.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_inc = self.inc(x)  # N * 64 * H * W
        x_down1 = self.down1(x_inc)  # N * 128 * H/2 * W/2
        x_down2 = self.down2(x_down1)  # N * 256 * H/4 * W/4
        x_up1 = self.upconv1(x_down2)  # N * 128 * H/2 * W/2
        x_up1 = torch.cat([x_up1, x_down1], dim=1)  # N * 256 * H/2 * W/2
        x_up1 = self.conv1(x_up1)  # N * 128 * H/2 * W/2
        x_up2 = self.upconv2(x_up1)  # N * 64 * H * W
        x_up2 = torch.cat([x_up2, x_inc], dim=1)  # N * 128 * H * W
        x_up2 = self.conv2(x_up2)  # N * 64 * H * W
        x_outc = self.outc(x_up2)  # N * n_classes * H * W
        return x_outc

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict affordance using this model.
        This is required due to BCEWithLogitsLoss
        """
        return torch.sigmoid(self.forward(x))

    @staticmethod
    def get_criterion() -> torch.nn.Module:
        """
        Return the Loss object needed for training.
        Hint: why does nn.BCELoss does not work well?
        """
        return nn.BCEWithLogitsLoss()

    @staticmethod
    def visualize(input: np.ndarray, output: np.ndarray, 
            target: Optional[np.ndarray]=None) -> np.ndarray:
        """
        Visualize rgb input and affordance as a single rgb image.
        """
        cmap = cm.get_cmap('viridis')
        in_img = np.moveaxis(input, 0, -1)
        pred_img = cmap(output[0])[...,:3]
        row = [in_img, pred_img]
        if target is not None:
            gt_img = cmap(target[0])[...,:3]
            row.append(gt_img)
        img = (np.concatenate(row, axis=1)*255).astype(np.uint8)
        return img


    def predict_grasp(
        self, 
        rgb_obs: np.ndarray,  
    ) -> Tuple[Tuple[int, int], float, np.ndarray]:
        """
        Given an RGB image observation, predict the grasping location and angle in image space.
        return coord, angle, vis_img
        :coord: tuple(int x, int y). By OpenCV convension, x is left-to-right and y is top-to-bottom.
        :angle: float. By OpenCV convension, angle is clockwise rotation.
        :vis_img: np.ndarray(shape=(H,W,3), dtype=np.uint8). Visualize prediction as a RGB image.

        Note: torchvision's rotation is counter clockwise, while imgaug,OpenCV's rotation are clockwise.
        """
        device = self.device
        coord, angle = None, None 
        # 1. Take RGB input
        image = rgb_obs
        print("image.shape: ",image.shape)

        # 2. Rotate to [0-7] * 22.5°
        # define augmentation
        augs = [iaa.Affine(rotate=i*22.5) for i in range(8)] #imgaug counter-clockwise
        # apply augmentations to input images
        images_rotated = [aug(image=image) for aug in augs]
        images_tensor = torch.tensor(images_rotated, dtype=torch.float32).to(device)
        images_tensor = images_tensor.permute(0,3,1,2)
        print("after rotate images_tensor: ",images_tensor.shape)

        # 3. Feed into network 
        affordance_map = self.predict(images_tensor)
        affordance_map = torch.clip(affordance_map,0,1)
        print("affordance_map.shape: ",affordance_map.shape) 

        # 5. Find the max affordance pixel across all 8 images
        # find index of max pixel value
        max_idx = torch.argmax(affordance_map.view(-1), dim=None).numpy()
        # find index of rotated image and corresponding location
        idx_rotated, idx_loc = divmod(max_idx, affordance_map.shape[2]*affordance_map.shape[3])
        print("max_idx: ",max_idx, "idx_rotated: ",idx_rotated, "idx_loc: ",idx_loc)
        # find rotation angle and location in original image
        best_img_rotate_angle = 22.5 * idx_rotated
        # OpenCV clockwise convention 
        angle = -1*best_img_rotate_angle
        best_img_center = KeypointsOnImage([Keypoint(x=idx_loc % 128, y=idx_loc // 128)], shape=rgb_obs.shape)
        best_img_coord = best_img_center.keypoints[0]
        best_img_coord = (best_img_coord.x, best_img_coord.y)
        rotate_to_original = iaa.Sequential([iaa.Affine(rotate=angle)])
        original_center = rotate_to_original(keypoints=best_img_center)
        center = original_center.keypoints[0]    
        coord = (center.x, center.y)
        print("Step 5: angle", angle, "coord", coord)


        # TODO: (problem 3, skip when finishing problem 2) avoid selecting the same failed actions
        # ===============================================================================
        for max_coord in list(self.past_actions):  
            bin = max_coord[0] 
            # supress past actions and select next-best action
        # ===============================================================================
        
        # TODO: (problem 2) complete this method (visualization)
        # :vis_img: np.ndarray(shape=(H,W,3), dtype=np.uint8). Visualize prediction as a RGB image.
        # Hint: use common.draw_grasp
        # Hint: see self.visualize
        # Hint: draw a grey (127,127,127) line on the bottom row of each image.
        # ===============================================================================
        vis_img = None
        # Step 1: : iterate through all 8 pairs of (img, pred) and np.concatenate them together. use draw_grasp to visualize grasp pose
        for i in range(8):
            this_img = images_rotated[i]
            this_affordance = affordance_map[i]
            this_prediction = this_affordance.detach().numpy()

            cmap = cm.get_cmap('viridis')
            in_img = this_img
            pred_img = cmap(this_prediction[0])[...,:3]
            
            # Test
            # find index of max pixel value
            #this_max_idx = torch.argmax(this_affordance.view(-1), dim=None).numpy()
            #best_img_coord = (this_max_idx % 128, this_max_idx // 128)
            if i == idx_rotated:
                in_image = draw_grasp(img=in_img, coord=best_img_coord, angle=0)

            row = [in_img, pred_img]
            img = (np.concatenate(row, axis=1)*255).astype(np.uint8)

            vis_img = img if vis_img is None else np.concatenate([vis_img, img], axis=0)
        # Step 2: draw a grey (127,127,127) line on the bottom row of each image.
        
        # ===============================================================================
        return coord, angle, vis_img

