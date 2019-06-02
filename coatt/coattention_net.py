import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.nn.utils.rnn as rnn


class CoattentionNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Hierarchical Question-Image Co-Attention
    for Visual Question Answering (Lu et al, 2017) paper.
    """
    def __init__(self, num_embeddings, num_classes, embed_dim=512, k=30):
        super().__init__()

        self.embed = nn.Embedding(num_embeddings, embed_dim)

        self.unigram_conv = nn.Conv1d(embed_dim, embed_dim, 1, stride=1, padding=0)
        self.bigram_conv  = nn.Conv1d(embed_dim, embed_dim, 2, stride=1, padding=1, dilation=2)
        self.trigram_conv = nn.Conv1d(embed_dim, embed_dim, 3, stride=1, padding=2, dilation=2)
        self.max_pool = nn.MaxPool2d((3, 1))
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, num_layers=3, dropout=0.4)
        self.tanh = nn.Tanh()

        self.W_b = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.W_v = nn.Parameter(torch.randn(k, embed_dim))
        self.W_q = nn.Parameter(torch.randn(k, embed_dim))
        self.w_hv = nn.Parameter(torch.randn(k, 1))
        self.w_hq = nn.Parameter(torch.randn(k, 1))

        #self.W_w = nn.Parameter(torch.randn(embed_dim, embed_dim))
        #self.W_p = nn.Parameter(torch.randn(embed_dim*2, embed_dim))
        #self.W_s = nn.Parameter(torch.randn(embed_dim*2, embed_dim))

        self.W_w = nn.Linear(embed_dim, embed_dim)
        self.W_p = nn.Linear(embed_dim*2, embed_dim)
        self.W_s = nn.Linear(embed_dim*2, embed_dim)

        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, image, question):                    # Image: B x 512 x 196
        question, lens = rnn.pad_packed_sequence(question)
        question = question.permute(1, 0)                  # Ques : B x L
        words = self.embed(question).permute(0, 2, 1)      # Words: B x L x 512

        unigrams = torch.unsqueeze(self.tanh(self.unigram_conv(words)), 2) # B x 512 x L
        bigrams  = torch.unsqueeze(self.tanh(self.bigram_conv(words)), 2)  # B x 512 x L
        trigrams = torch.unsqueeze(self.tanh(self.trigram_conv(words)), 2) # B x 512 x L
        words = words.permute(0, 2, 1)

        phrase = torch.squeeze(self.max_pool(torch.cat((unigrams, bigrams, trigrams), 2)))
        phrase = phrase.permute(0, 2, 1)                                    # B x L x 512

        hidden = None
        phrase_packed = nn.utils.rnn.pack_padded_sequence(torch.transpose(phrase, 0, 1), lens)
        sentence_packed, hidden = self.lstm(phrase_packed, hidden)
        sentence, _ = rnn.pad_packed_sequence(sentence_packed)
        sentence = torch.transpose(sentence, 0, 1)                          # B x L x 512

        v_word, q_word = self.parallel_co_attention(image, words)
        v_phrase, q_phrase = self.parallel_co_attention(image, phrase)
        v_sent, q_sent = self.parallel_co_attention(image, sentence)

        #h_w = self.tanh(torch.matmul((q_word + v_word), self.W_w))
        #h_p = self.tanh(torch.matmul(torch.cat(((q_phrase + v_phrase), h_w), dim=1), self.W_p))
        #h_s = self.tanh(torch.matmul(torch.cat(((q_sent + v_sent), h_p), dim=1), self.W_s))

        h_w = self.tanh(self.W_w(q_word + v_word))
        h_p = self.tanh(self.W_p(torch.cat(((q_phrase + v_phrase), h_w), dim=1)))
        h_s = self.tanh(self.W_s(torch.cat(((q_sent + v_sent), h_p), dim=1)))

        logits = self.fc(h_s)

        return logits

    def parallel_co_attention(self, V, Q):  # V : B x 512 x 196, Q : B x L x 512
        C = torch.matmul(Q, torch.matmul(self.W_b, V)) # B x L x 196

        H_v = self.tanh(torch.matmul(self.W_v, V) + torch.matmul(torch.matmul(self.W_q, Q.permute(0, 2, 1)), C))                            # B x k x 196
        H_q = self.tanh(torch.matmul(self.W_q, Q.permute(0, 2, 1)) + torch.matmul(torch.matmul(self.W_v, V), C.permute(0, 2, 1)))           # B x k x L

        #a_v = torch.squeeze(fn.softmax(torch.matmul(torch.t(self.w_hv), H_v), dim=2)) # B x 196
        #a_q = torch.squeeze(fn.softmax(torch.matmul(torch.t(self.w_hq), H_q), dim=2)) # B x L

        a_v = fn.softmax(torch.matmul(torch.t(self.w_hv), H_v), dim=2) # B x 1 x 196
        a_q = fn.softmax(torch.matmul(torch.t(self.w_hq), H_q), dim=2) # B x 1 x L

        v = torch.squeeze(torch.matmul(a_v, V.permute(0, 2, 1))) # B x 512
        q = torch.squeeze(torch.matmul(a_q, Q))                  # B x 512

        return v, q
