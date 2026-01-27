# SO-101 Policy transforms for OpenPi Pi0.5
# Adapted from Ilia Larchenko's LeKiwi implementation
# https://github.com/IliaLarchenko/lerobot_random/blob/main/vla/pi/lekiwi_policy.py
#
# Copy this file to: openpi/src/openpi/policies/so101_policy.py

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


# SO-101 has 6 DOF: 5 arm joints + 1 gripper
SO101_ACTION_DIM = 6


def make_so101_example() -> dict:
    """Creates a random input example for testing SO-101 policy."""
    return {
        "observation/state": np.random.rand(SO101_ACTION_DIM).astype(np.float32),
        "observation/images/overhead": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/images/wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "prompt": "pick up the orange ball and put it in the pink cup",
    }


def _parse_image(image) -> np.ndarray:
    """Convert image to HWC uint8 format expected by Pi0."""
    image = np.asarray(image)
    # LeRobot stores as float32 CHW, convert to uint8 HWC
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class SO101Inputs(transforms.DataTransformFn):
    """
    Convert SO-101 observations to Pi0 model input format.
    
    SO-101 has:
    - 6 DOF state (5 arm joints + 1 gripper)
    - 2 cameras (overhead + wrist)
    
    Pi0 expects 3 camera slots, so we duplicate overhead for the third slot.
    """
    
    # Model's action dimension (SO-101 actions will be padded to this)
    action_dim: int
    
    # Model type (PI0, PI05, PI0_FAST)
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # Pad state from 6 DOF to model's action_dim
        state = transforms.pad_to_dim(data["observation/state"], self.action_dim)

        # Parse images from SO-101's camera keys
        overhead_image = _parse_image(data["observation/images/overhead"])
        wrist_image = _parse_image(data["observation/images/wrist"])

        # Map to Pi0's expected camera slots:
        # - base_0_rgb: overhead camera (top-down view)
        # - left_wrist_0_rgb: wrist camera
        # - right_wrist_0_rgb: duplicate overhead (we only have 2 cameras)
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": overhead_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": overhead_image,  # Duplicate overhead
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # For Pi0 (not FAST), mask the duplicated camera
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Pad actions during training
        if "action" in data:
            actions = transforms.pad_to_dim(data["action"], self.action_dim)
            inputs["actions"] = actions

        # Pass language prompt to model
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class SO101Outputs(transforms.DataTransformFn):
    """
    Convert Pi0 model outputs back to SO-101 action format.
    
    Only return the first 6 actions (5 arm joints + 1 gripper),
    discarding any padding.
    """

    def __call__(self, data: dict) -> dict:
        # Return only first 6 actions for SO-101
        return {"actions": np.asarray(data["actions"][:, :SO101_ACTION_DIM])}
