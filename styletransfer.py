from LoadModel import get_style_model_and_losses
from optimizer import get_input_param_optimizer
import gc
from PIL import Image
import torchvision
from imageshow import imshow

def run_style_transfer(content_img,style_img,input_img,num_steps=1000,
                     style_weight=1000,content_weight=1):
    model,style_losses,content_losses = get_style_model_and_losses(style_img,content_img,style_weight,content_weight)
    input_param, optimizer =get_input_param_optimizer(input_img)
    print("optimizig..")
    run=[0]
    while run[0]<=num_steps:
        def closure():
            input_param.data.clamp_(0,1)
            optimizer.zero_grad()
            model(input_param)
            style_score=0
            content_score=0
            for sl in style_losses:
                 style_score+=sl.backward()
            for cl in content_losses:
                 content_score+=cl.backward()
            run[0]+=1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score, content_score))
                print()
            return style_score + content_score
        optimizer.step(closure)
        gc.collect()
    input_param.data.clamp_(0, 1)
    return input_param.data



  