import torch
from ..utils.base_model import BaseModel
from easy_local_features import getExtractor

class EasyFeatures(BaseModel):
    default_conf = {
        'name': 'easy-features',
        'elf_conf': {
            'model_name': 'alike-t',
            'top_k': -1,
            'scores_th': 0.2,
            'n_limit': 2048,
            'sub_pixel': True,
            'model_path': None
        }
    }

    def _init(self, conf):
        print(f"{conf=}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = getExtractor(conf["elf_model"], conf["elf_conf"])
        self.model.to(self.device)

    def _forward(self, data):
        response = self.model.detectAndCompute(data['image'], return_dict=True)

        if 'scores' in response:
            scores = response['scores']
            response['keypoint_scores'] = scores
            del response['scores']

        return response
