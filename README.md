# VQA
Hierarchical Question-Image Co-Attention for Visual Question Answering

## Train
### Simple Baseline
python3 -m coatt.main --model simple

### Coattention Network
python3 -m coatt.main --model coattention

## The code is based on the following papers:
[1] VQA: Visual Question Answering (Agrawal et al, 2016): https://arxiv.org/pdf/1505.00468v6.pdf

[2] Simple Baseline for Visual Question Answering (Zhou et al, 2015): https://arxiv.org/pdf/1512.02167.pdf

[3] Hierarchical Question-Image Co-Attention for Visual Question Answering (Lu et al, 2017):  https://arxiv.org/pdf/1606.00061.pdf
