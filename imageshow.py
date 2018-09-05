import torch
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms



from loadimage import content_img,style_img,h,w

unloader = transforms.ToPILImage()  # 转回 PIL 图像
plt.ion()

def imshow(tensor):
    image = tensor.clone().cpu()  # 克隆是为了不改变它
    image = image.view(3, h, w)  # 移除 batch 维度
    image = unloader(image)
    plt.imshow(image)
    plt.show(image)



plt.figure()
imshow(style_img.data)

plt.figure()
imshow(content_img.data)