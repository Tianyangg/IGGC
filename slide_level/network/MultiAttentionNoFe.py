import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiAttentionNoFe(nn.Module):
    def __init__(self, n_class=2, n_head=5, len_featurelist=2):
        super(MultiAttentionNoFe, self).__init__()
        self.L = 128
        self.D = 64
        self.K = 1
        self.n_head = n_head

        self.bn1 = nn.BatchNorm1d(1)
        self.bn2 = nn.BatchNorm1d(1)
        self.feature_extractor_part1 = nn.Sequential(nn.Linear(1024 * len_featurelist, 128),
                                                     nn.ReLU())
        self.max = nn.AdaptiveMaxPool1d(1)

        self.attention = nn.Sequential()
        for i in range(self.n_head):
            self.attention.add_module('attention%i' % i,
                                      nn.Sequential(
                                            nn.Linear(self.L, self.D),
                                            nn.Tanh(),
                                            nn.Linear(self.D, self.K))
                                      )
        self.activation = nn.Softmax()
        self.bn3 = nn.BatchNorm1d(1)
        self.bn4 = nn.BatchNorm1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(128 * n_head, 128),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(128, n_class),
        )

    def forward(self, x):
        nbatch, nbag, nfeature = x.shape

        x = x.view(-1, nfeature)
        x = x.unsqueeze(1)

        x = self.bn1(x)

        x = self.feature_extractor_part1(x)
        x = x.view(-1, 128)

        for i in range(self.n_head):
            A = self.attention[i](x)  # Nbag x K
            A = torch.transpose(A, 1, 0)  # K x Nbag
            A = self.activation(A)  # softmax over Nbag

            M = torch.mm(A, x)  # K x L
            M = self.bn3(M.reshape(1, 1, 128)).squeeze(1)
            if i == 0:
                feature = M
                attention = A
            else:
                feature = torch.cat([feature, M], 1)
                attention = torch.cat([attention, A], 0)

        feature = feature.view(-1, 128 * self.n_head)
        feature = self.bn4(feature.unsqueeze(1)).view(-1, 128 * self.n_head)
        output = self.classifier(feature)

        return output, attention