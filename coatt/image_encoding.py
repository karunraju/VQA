import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
from six.moves import cPickle as pickle

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class VqaImgDataset(Dataset):

    def __init__(self, image_dir, name, img_prefix):
        self.image_dir = image_dir
        self.img_names = [f for f in os.listdir(self.image_dir) if '.jpg' in f]
        self.transform = transforms.Compose([transforms.Resize((448, 448)),
                                             transforms.ToTensor()])

        img_ids = {}
        for idx, fname in enumerate(self.img_names):
            img_id = fname.split('.')[0].rpartition(img_prefix)[-1]
            img_ids[int(img_id)] = idx

        with open('/home/ubuntu/hw3_release/data/' + name + '_enc_idx.npy', 'wb') as f:
            pickle.dump(img_ids, f)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img = default_loader(self.image_dir + '/' + self.img_names[idx])
        imgT = self.transform(img)

        return imgT.float()

tr_image_dir = '/home/ubuntu/hw3_release/data/train2014'
va_image_dir = '/home/ubuntu/hw3_release/data/val2014'
tr_out_dir = '/home/ubuntu/hw3_release/data/tr_enc'
va_out_dir = '/home/ubuntu/hw3_release/data/va_enc'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = models.resnet18(pretrained=True)
modules = list(model.children())[:-2]
model = nn.Sequential(*modules)
for params in model.parameters():
    params.requires_grad = False

if DEVICE == 'cuda':
    model = model.cuda()

tr_img_dataset = VqaImgDataset(image_dir=tr_image_dir, name='train', img_prefix="COCO_train2014_")
tr_img_dataset_loader = DataLoader(tr_img_dataset, batch_size=50, shuffle=False, num_workers=10)

va_img_dataset = VqaImgDataset(image_dir=va_image_dir, name='val', img_prefix="COCO_val2014_")
va_img_dataset_loader = DataLoader(va_img_dataset, batch_size=50, shuffle=False, num_workers=10)

print('Dumping Training images encodings.')
for idx, imgT in enumerate(tr_img_dataset_loader):
    imgT = imgT.to(DEVICE)
    out = model(imgT)
    out = out.view(out.size(0), out.size(1), -1)
    out = out.cpu().numpy()

    path = tr_out_dir + '/' + str(idx) + '.npz'
    #np.savez(path, out=out)
    np.savez_compressed(path, out=out)
    print(path)

print('Dumping Validation images encodings.')
for idx, imgT in enumerate(va_img_dataset_loader):
    imgT = imgT.to(DEVICE)
    out = model(imgT)
    out = out.view(out.size(0), out.size(1), -1)
    out = out.cpu().numpy()

    path = va_out_dir + '/' + str(idx) + '.npz'
    #np.savez(path, out=out)
    np.savez_compressed(path, out=out)
    print(path)
