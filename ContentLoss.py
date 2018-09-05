import torch
import torch.nn as nn

class ContentLoss(nn.Module):
    def __init__(self,target,weight):
        super(ContentLoss,self).__init__()
        self.target=target.detach()*weight
        self.weight=weight
        self.criterion=nn.MSELoss()
    
    def forward(self,input):
        self.loss=self.criterion(input*self.weight,self.target)
        self.output=input
        return self.output

    def backward(self,retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

