import torch
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

h=3120
w=4160
dtype = torch.cuda.FloatTensor

loader=transforms.Compose([
    transforms.Resize([h,w]),
    transforms.ToTensor()
])


def imageloader(image_name):
    image=Image.open(image_name)
    image=Variable(loader(image))
    image=image.unsqueeze(0)
    return image



style_img=imageloader("images/nahan.jpg").type(dtype)
content_img=imageloader("images/fj.jpg").type(dtype)

print(style_img.shape)
print(content_img.shape)
