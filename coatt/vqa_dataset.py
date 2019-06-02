import os
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image
from six.moves import cPickle as pickle
from torch.utils.data import Dataset
from external.vqa.vqa import VQA
from coatt.dataset import pre_process_dataset

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
    return img


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class VqaDataset(Dataset):
    """
    Load the VQA dataset using the VQA python API. We provide the necessary subset in the External folder, but you may
    want to reference the full repo (https://github.com/GT-Vision-Lab/VQA) for usage examples.
    """

    def __init__(self, image_dir, question_json_file_path, annotation_json_file_path,
                 image_filename_pattern, collate=False, q2i=None, a2i=None, i2a=None,
                 a2i_count=None, img_names=None, img_ids=None, ques_ids=None,
                 method='simple', dataset_type='train', enc_dir=''):
        """
        Args:
            image_dir (string): Path to the directory with COCO images
            question_json_file_path (string): Path to the json file containing the question data
            annotation_json_file_path (string): Path to the json file containing the annotations mapping images, questions, and
                answers together
            image_filename_pattern (string): The pattern the filenames of images in this dataset use (eg "COCO_train2014_{}.jpg")
        """
        print(method)
        self.image_dir = image_dir
        self.qjson = question_json_file_path
        self.ajson = annotation_json_file_path
        img_prefix = image_filename_pattern.split('{}')[0]
        self.collate = collate
        self.q2i = q2i
        self.a2i = a2i
        self.i2a = i2a
        self.a2i_count = a2i_count
        self.img_ids = img_ids
        self.ques_ids = ques_ids
        self.img_names = img_names
        self.method = method
        self.vqa = VQA(self.ajson, self.qjson)

        if self.method == 'simple':
            self.transform = transforms.Compose([transforms.Resize((224, 224)),
                                                 transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([transforms.Resize((448, 448)),
                                                 transforms.ToTensor()])


        if not collate:
            self.img_names = [f for f in os.listdir(self.image_dir) if '.jpg' in f]
            self.img_ids = []
            for fname in self.img_names:
                img_id = fname.split('.')[0].rpartition(img_prefix)[-1]
                self.img_ids.append(int(img_id))

            self.ques_ids = self.vqa.getQuesIds(self.img_ids)

            self.q2i, self.a2i, self.i2a, self.a2i_count = pre_process_dataset(image_dir, self.qjson,
                                                                               self.ajson, img_prefix)

        self.q2i_len = len(self.q2i)
        self.a2i_len = len(self.a2i.keys())
        self.q2i_keys = self.q2i.keys()
        self.enc_dir = enc_dir

        if collate and dataset_type == 'train':
            with open('/home/ubuntu/hw3_release/data/train_enc_idx.npy', 'rb') as f:
                self.enc_idx = pickle.load(f)
        elif collate and dataset_type == 'val':
            with open('/home/ubuntu/hw3_release/data/val_enc_idx.npy', 'rb') as f:
                self.enc_idx = pickle.load(f)

    def __len__(self):
        return len(self.ques_ids)

    def __getitem__(self, idx):
        ques_id = self.ques_ids[idx]
        img_id = self.vqa.getImgIds([ques_id])[0]

        qa = self.vqa.loadQA(ques_id)[0]
        qqa = self.vqa.loadQQA(ques_id)[0]
        img_name = self.img_names[self.img_ids.index(img_id)]

        if self.method == 'simple':
            img = default_loader(self.image_dir + '/' + img_name)
            #imgT = self.transform(img).permute(1, 2, 0)
            imgT = self.transform(img).float()
        else:
            #file_idx = self.enc_idx[img_id] // 50
            #arr_idx = self.enc_idx[img_id] % 50
            #path = self.enc_dir + '/' + str(file_idx) + '.npz'
            #img = np.load(path)['out'][arr_idx, :, :]               # 512 x 196
            #imgT = torch.from_numpy(img).float()

            img = default_loader(self.image_dir + '/' + img_name)
            imgT = self.transform(img).float()

        ques = qqa['question'][:-1]
        quesI = [self.q2i["<sos>"]] + [self.q2i[x.lower()] for x in ques.split(" ") if x.lower() in self.q2i_keys] + [self.q2i["<eos>"]]
        if not self.collate:
            quesI = quesI + [self.q2i["<pad>"]]*(8 - len(quesI))
        if self.method == 'simple':
            quesT = torch.zeros(self.q2i_len).float()
            for idx in quesI:
                quesT[idx] = 1
        else:
            quesT = torch.from_numpy(np.array(quesI)).long()

        answers = qa['answers']
        max_count = 0
        answer = ""
        for ans in answers:
            #if not ans['answer_confidence'] == 'yes':
            #    continue
            ans = ans['answer'].lower()
            if ans in self.a2i.keys() and self.a2i_count[ans] > max_count:
                max_count = self.a2i_count[ans]
                answer = ans

        if answer == "":                                              # only for validation
            gT = torch.from_numpy(np.array([self.a2i_len])).long()
        else:
            gT = torch.from_numpy(np.array([self.a2i[answer]])).long()

        if not self.collate:
            return {'img' : imgT, 'ques' : quesT, 'gt': gT}

        return imgT, quesT, gT
