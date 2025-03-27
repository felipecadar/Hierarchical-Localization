import torch
from ..utils.base_model import BaseModel
from easy_local_features import getExtractor

class EasyFeatures(BaseModel):
    default_conf = {
        'name': 'easy-features',
        'elf_model': 'alike',
        'elf_conf': {
            'model_name': 'alike-t',
            'top_k': 4096,
            'scores_th': 0.0,
            'n_limit': 10000,
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
        
        # invert descriptors shape. Why ? I don't know, but other implementations are doing it. 
        response['descriptors'] = response['descriptors'].permute(0, 2, 1)

        return response


if __name__ == "__main__":
    from hloc.utils.io import read_image
    model = EasyFeatures(conf=EasyFeatures.default_conf)
    
    image = read_image("datasets/sacre_coeur/mapping/02928139_3448003521.jpg")
    print(image.shape)
    
    response = model({"image":image})
    response = {k: v[0].cpu().numpy() for k, v in response.items()}

    print(response.keys())
    for k in response.keys():
        print(k, response[k].shape)
        print(k, response[k].dtype)