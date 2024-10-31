import torch
import torch.nn as nn
from torchvision import models

class ImageMIL(nn.Module):
    def __init__(self, n_class, n_head):
        super(ImageMIL, self).__init__()
        self.n_head = n_head
        self.n_class = n_class
        self.fearure_extractor = models.densenet121(pretrained=True)  # 1024
        
        self.L = 128
        self.D = 64
        self.K = 1

        self.bn1 = nn.BatchNorm1d(1)
        self.bn2 = nn.BatchNorm1d(1)
        self.feature_encoder = nn.Sequential(nn.ReLU(),
                                             nn.Linear(1024, self.L))
        

        self.attention = nn.Sequential()
        for i in range(self.n_head):
            self.attention.add_module("attention%i" % i, 
                                      nn.Sequential(
                                            nn.Linear(self.L, self.D), 
                                            nn.Tanh(),
                                            nn.Linear(self.D, self.K)))
        self.activation = nn.Softmax()
 
        self.bn3 = nn.BatchNorm1d(1)
        
        self.bn4 = nn.BatchNorm1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.n_head, self.n_class), 
            # nn.BatchNorm1d(1),
            # nn.LeakyReLU(), 
            # nn.Linear(64, self.n_class)
        )

    def forward(self, x):
        n_batch = x.shape[0]
        n_bag = x.shape[1]
        x = x.squeeze() # ellinimate the fake batch dimension
        
        x = self.fearure_extractor.features(x)  # [instance, 3, h, w] -> [instance, feature map]
       
        # x = self.adaptive_average_pool(x) # [instance, fm] -> [instance 32*32]
        x = x.reshape(-1, 768)
        x = self.bn1(x.unsqueeze(1)).squeeze(1)
        x = self.feature_encoder(x)  # [instance xxx] -> [instance 512] 
        # apply batchnorm
        x = self.bn2(x.unsqueeze(1)).squeeze(1)

        for i in range(self.n_head):
            A = self.attention[i](x)  # [instance * 1]
            A = torch.transpose(A, 1, 0)  # [1 instance]
            A = self.activation(A)  # softmax over Nbag

            M = torch.mm(A, x)  # [1 128]
            M = self.bn3(M.unsqueeze(1)).squeeze(1)
            
            if i == 0:
                feature = M
                attention = A
            else:
                feature = torch.cat([feature, M], 1)
                attention = torch.cat([attention, A], 0)

        feature = feature.view(-1, self.L * self.n_head)  
        feature = self.bn4(feature.unsqueeze(1)).view(-1, self.L * self.n_head)
        output = self.classifier(feature)

        return output, A