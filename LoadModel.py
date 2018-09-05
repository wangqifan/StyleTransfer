import torch
import torch.nn as nn
import torchvision.models as models
import copy
from GramMatrix import GramMatrix
from ContentLoss import  ContentLoss
from StyleLoss import StyleLoss
import gc


cnn=models.vgg19(pretrained=True).features
cnn=cnn.cuda()
content_layers_default = ['conv_4']
style_layers_default =['conv_1','conv_2', 'conv_3','conv_4', 'conv_5']

def get_style_model_and_losses(style_img,content_img,
                            style_weight,content_weight,
                            content_layers=content_layers_default,
                            style_layers=style_layers_default):
    global cnn
    print("CALL")
    content_losses=[]
    style_losses=[]
    print(cnn)
    model=nn.Sequential()
    gram=GramMatrix()

    model=model.cuda()
    gram=gram.cuda()
    i=1
    for layer in list(cnn):
        if isinstance(layer,nn.Conv2d):
            name = "conv_" + str(i)
            model.add_module(name, layer)

            if name in content_layers:
                # 加内容损失:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)
        if isinstance(layer,nn.ReLU):
            name = "relu_" + str(i)
            model.add_module(name, layer)

            if name in content_layers:
                # 加内容损失:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # 加风格损失:
                target_feature = model(style_img).clone()
                target_feature_gram = gram(target_feature)
                style_loss = StyleLoss(target_feature_gram, style_weight)
                model.add_module("style_loss_" + str(i), style_loss)
                style_losses.append(style_loss)
        if isinstance(layer,nn.MaxPool2d):
            name = "pool_" + str(i)
            model.add_module(name, layer) 
        i+=1
    del cnn,gram
    gc.collect()
    return model,style_losses,content_losses