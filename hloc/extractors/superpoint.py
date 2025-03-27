import sys
from pathlib import Path

import torch

from ..utils.base_model import BaseModel

sys.path.append(str(Path(__file__).parent / "../../third_party"))
from SuperGluePretrainedNetwork.models import superpoint  # noqa E402


# The original keypoint sampling is incorrect. We patch it here but
# we don't fix it upstream to not impact exisiting evaluations.
def sample_descriptors_fix_sampling(keypoints, descriptors, s: int = 8):
    """Interpolate descriptors at keypoint locations"""
    b, c, h, w = descriptors.shape
    keypoints = (keypoints + 0.5) / (keypoints.new_tensor([w, h]) * s)
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode="bilinear", align_corners=False
    )
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1
    )
    return descriptors


class SuperPoint(BaseModel):
    default_conf = {
        "nms_radius": 4,
        "keypoint_threshold": 0.005,
        "max_keypoints": -1,
        "remove_borders": 4,
        "fix_sampling": False,
    }
    required_inputs = ["image"]
    detection_noise = 2.0

    def _init(self, conf):
        if conf["fix_sampling"]:
            superpoint.sample_descriptors = sample_descriptors_fix_sampling
        self.net = superpoint.SuperPoint(conf)

    def _forward(self, data):
        return self.net(data)


def process_image(fname):
    import numpy as np
    image = read_image(fname, True)
    image = image.astype(np.float32)
    size = image.shape[:2][::-1]

    grayscale = True
    if grayscale:
        image = image[None]
    else:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    image = image / 255.0

    data = {
        "image": torch.tensor(image).unsqueeze(0),  # add batch dimension
        "original_size": np.array(size),
    }
    return data

if __name__ == "__main__":
    from hloc.utils.io import read_image
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.no_grad():
        model = SuperPoint(conf=SuperPoint.default_conf).to(device)

        data = process_image("datasets/sacre_coeur/mapping/02928139_3448003521.jpg")
        response = model({"image": data["image"].to(device, non_blocking=True)})

        response = {k: v[0].cpu().numpy() for k, v in response.items()}

        print(response.keys())
        for k in response.keys():
            print(k, response[k].shape)
            print(k, response[k].dtype)