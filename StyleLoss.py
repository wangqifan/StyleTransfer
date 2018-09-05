import torch
import torch.nn as nn

from GramMatrix import GramMatrix

class StyleLoss(nn.Module):
    def __init__(self,target,weight):
        super(StyleLoss,self).__init__()
        self.target=target.detach()*weight
        self.weight=weight
        self.gram=GramMatrix()
        self.criterion=nn.MSELoss()

    def forward(self,input):
        self.output=input.clone()
        self.G=self.gram(input)
        self.G.mul_(self.weight)
        self.loss=self.criterion(self.G,self.target)
        del input
        return self.output
    
    def backward(self,retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

    

