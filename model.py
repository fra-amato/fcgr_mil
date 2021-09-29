import torch
import torch.nn as nn
import math

class MilModel(nn.Module):
    def __init__(self, kmers_len = 2):
        super(MilModel, self).__init__()
        device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu')
        max_pool_output = math.floor((math.pow(2,kmers_len)-2)/2) + 1
        self.convolutional = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=10,kernel_size=(3,3),padding='same',stride=1,device=device,dtype=torch.float32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2),stride=2)
        )

        self.linear = nn.Sequential(
            nn.Linear(10 * max_pool_output,500,device=device,dtype=torch.float32),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(500, 2,device=device,dtype=torch.float32),
            nn.Softmax()
        )
    
    def forward(self,input):
        out = self.convolutional(input)
        out = out.flatten(start_dim = 1) #start_dim = 1 esclude la dimensione del batch size dal flatten
        out = self.linear(out)
        return self.classifier(out)
