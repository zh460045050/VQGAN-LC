import numpy as np
import os
import torch

dir_path = "clustering_centers_random_100K_VitB"
files = os.listdir("clustering_centers_random_100K_VitB")


features = [torch.from_numpy(np.load(os.path.join(dir_path, file))) for file in files]
features = torch.cat(features, dim=0)
torch.save(features, "random_codebook_1000cls_100000_vitb.pth")
print(features.shape)