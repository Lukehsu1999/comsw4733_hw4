from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

from pick_labeler import draw_grasp


class ActionRegressionDataset(Dataset):
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
        training targets for ActionRegressionModel.
        return: 
        {
            'input': torch.Tensor (3,H,W), torch.float32
            'target': torch.Tensor (3,), torch.float32
        }
        Note: self.raw_dataset[idx]['rgb'] is torch.Tensor (H,W,3) torch.uint8
        Note: target: [x, y, angle] scaled to between 0 and 1.
        """
        # checkout train.RGBDataset for the content of data
        data = self.raw_dataset[idx]
        rgb = data['rgb'].numpy()
        center_point = data['center_point'].numpy()
        angle = data['angle'].item()

        # normalize
        rgb = rgb / 255.0
        center_point = center_point / 128.0
        angle = angle / 360.0
        target = np.array([center_point[0], center_point[1], angle])

        # convert to tensor
        rgb = torch.from_numpy(rgb).permute(2,0,1).float()
        target = torch.from_numpy(target).float()

        # TODO: complete this method
        # ===============================================================================
        return dict(
            input=rgb,
            target=target
        )
        # ===============================================================================


def recover_action(
        action: np.ndarray, 
        shape=(128,128)
        ) -> Tuple[Tuple[int, int], float]:
    """
    :action: np.ndarray([x,y,angle], dtype=np.float32)
    return:
    coord: tuple(x, y) in pixel coordinate between 0 and :shape:
    angle: float in degrees, clockwise
    """
    # TODO: complete this function
    # =============================================================================== 
    coord, angle = (action[0]*shape[0], action[1]*shape[1]), action[2]*360
    # ===============================================================================
    return coord, angle


class ActionRegressionModel(nn.Module):
    def __init__(self, pretrained=False, out_channels=3, **kwargs):
        super().__init__()
        # load backbone model
        model = mobilenet_v3_small(pretrained=pretrained)
        # replace the last linear layer to change output dimention to 3
        ic = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(
            in_features=ic, out_features=out_channels)
        self.model = model
        # normalize RGB input to zero mean and unit variance
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        # hack to get model device
        self.dummy_param = nn.Parameter(torch.empty(0))

    @property
    def device(self) -> torch.device:
        return self.dummy_param.device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(self.normalize(x))

    def predict(self, x):
        """
        Think: Why is this the same as forward 
        (comparing to AffordanceModel.predict)
        """
        return self.forward(x)

    @staticmethod
    def get_criterion():
        """
        Return the Loss object needed for training.
        """
        # TODO: complete this method
        # =============================================================================== 
        # MSE loss
        return nn.MSELoss()
        # =============================================================================== 

    @staticmethod
    def visualize(input: np.ndarray, output: np.ndarray, 
            target: Optional[np.ndarray]=None) -> np.ndarray:
        """
        Visualize rgb input and affordance as a single rgb image.
        """        
        vis_img = (np.moveaxis(input,0,-1).copy() * 255).astype(np.uint8)
        # target
        if target is not None:
            coord, angle = recover_action(target, shape=vis_img.shape[:2])
            draw_grasp(vis_img, coord, angle, color=(255,255,255))
        # pred
        coord, angle = recover_action(output, shape=vis_img.shape[:2])
        draw_grasp(vis_img, coord, angle, color=(0,255,0))
        return vis_img

    def predict_grasp(self, rgb_obs: np.ndarray
            ) -> Tuple[Tuple[int, int], float, np.ndarray]:
        """
        Given a RGB image observation, predict the grasping location and angle in image space.
        return coord, angle, vis_img
        :coord: tuple(int x, int y). By OpenCV convension, x is left-to-right and y is top-to-bottom.
        :angle: float. By OpenCV convension, angle is clockwise rotation.
        :vis_img: np.ndarray(shape=(H,W,3), dtype=np.uint8). Visualize prediction as a RGB image.

        Hint: use recover_action
        """
        device = self.device
        # TODO: complete this method (prediction)
        # Hint: why do we provide the model's device here?
        # ===============================================================================
        coord, angle = None, None

        # image = rgb_obs / 255.0
        # image_tensor = torch.from_numpy(image).permute(2,0,1).float().unsqueeze(0).to(device)

        # output = self.predict(image_tensor)
        # print("model output: ", output")
        # action = recover_action(action=output.cpu().detach().numpy()[0], shape=image.shape[:2]
        # ===============================================================================
        # visualization
        vis_img = self.visualize(input, action)
        return coord, angle, vis_img

