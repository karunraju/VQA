import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

from external.googlenet.googlenet import googlenet

class SimpleBaselineNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Simple Baseline for Visual Question Answering (Zhou et al, 2017) paper.
    """
    def __init__(self, num_embeddings, num_classes):
        super().__init__()

        self.gnet = googlenet(pretrained=True, remove_fc=True)
        #self.embed = nn.Embedding(num_embeddings, 1024)
        self.embed = nn.Linear(num_embeddings, 1024)
        self.fc = nn.Linear(1024 + 1024, num_classes)

    def forward(self, image, question_encoding):
        img = self.gnet(image)
        ques = self.embed(question_encoding)

        con = torch.cat((img, ques), dim=1)
        return self.fc(con)
