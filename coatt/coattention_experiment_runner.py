import torch
import numpy as np
from six.moves import cPickle as pickle
from torch.utils.data import DataLoader

from coatt.coattention_net import CoattentionNet
from coatt.experiment_runner_base import ExperimentRunnerBase
from coatt.vqa_dataset import VqaDataset


def collate_lines(seq_list):
    imgT, quesT, gT = zip(*seq_list)
    lens = [len(ques) for ques in quesT]
    seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
    imgT = torch.stack([imgT[i] for i in seq_order])
    quesT = [quesT[i] for i in seq_order]
    gT = torch.stack([gT[i] for i in seq_order])
    return imgT, quesT, gT


class CoattentionNetExperimentRunner(ExperimentRunnerBase):
    """
    Sets up the Co-Attention model for training. This class is specifically responsible for creating the model and optimizing it.
    """
    def __init__(self, train_image_dir, train_question_path, train_annotation_path,
                 test_image_dir, test_question_path,test_annotation_path, batch_size, num_epochs,
                 num_data_loader_workers):

        self.method = 'coattention'
        print('Loading numpy files. \n')
        with open('/home/ubuntu/hw3_release/data/q2i.pkl', 'rb') as f:
            q2i = pickle.load(f)
        with open('/home/ubuntu/hw3_release/data/a2i.pkl', 'rb') as f:
            a2i = pickle.load(f)
        with open('/home/ubuntu/hw3_release/data/i2a.pkl', 'rb') as f:
            i2a = pickle.load(f)
        with open('/home/ubuntu/hw3_release/data/a2i_count.pkl', 'rb') as f:
            a2i_count = pickle.load(f)

        tr_img_names = np.load('/home/ubuntu/hw3_release/data/tr_img_names.npy', encoding='latin1').tolist()
        tr_img_ids = np.load('/home/ubuntu/hw3_release/data/tr_img_ids.npy', encoding='latin1').tolist()
        tr_ques_ids = np.load('/home/ubuntu/hw3_release/data/tr_ques_ids.npy', encoding='latin1').tolist()

        va_img_names = np.load('/home/ubuntu/hw3_release/data/va_img_names.npy', encoding='latin1').tolist()
        va_img_ids = np.load('/home/ubuntu/hw3_release/data/va_img_ids.npy', encoding='latin1').tolist()
        #va_ques_ids = np.load('/home/ubuntu/hw3_release/data/va_ques_ids.npy', encoding='latin1').tolist()
        va_ques_ids = np.load('/home/ubuntu/hw3_release/data/va_ques_ids_orig.npy', encoding='latin1').tolist()

        print('Creating Datasets.')
        train_dataset = VqaDataset(image_dir=train_image_dir, collate=True,
                                   question_json_file_path=train_question_path,
                                   annotation_json_file_path=train_annotation_path,
                                   image_filename_pattern="COCO_train2014_{}.jpg",
                                   q2i=q2i, a2i=a2i, i2a=i2a, a2i_count=a2i_count,
                                   img_names=tr_img_names, img_ids=tr_img_ids,
                                   ques_ids=tr_ques_ids, method=self.method,
                                   dataset_type="train", enc_dir='/home/ubuntu/hw3_release/data/tr_enc')

        val_dataset = VqaDataset(image_dir=test_image_dir, collate=True,
                                 question_json_file_path=test_question_path,
                                 annotation_json_file_path=test_annotation_path,
                                 image_filename_pattern="COCO_val2014_{}.jpg",
                                 q2i=q2i, a2i=a2i, i2a=i2a, a2i_count=a2i_count,
                                 img_names=va_img_names, img_ids=va_img_ids,
                                 ques_ids=va_ques_ids, method=self.method,
                                 dataset_type="validation", enc_dir='/home/ubuntu/hw3_release/data/va_enc')

        self._train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=5, collate_fn=collate_lines)

        self._val_dataset_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=5, collate_fn=collate_lines)


        print('Creating Co Attention Model.')
        model = CoattentionNet(len(q2i), 1000).float()

        super().__init__(train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers)


    def _optimize(self, predicted_answers, true_answer_ids):
        self.optimizer.zero_grad()
        loss = self.criterion(predicted_answers, true_answer_ids)
        loss.backward()
        self.optimizer.step()

        return loss
