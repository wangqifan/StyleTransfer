from PIL import Image  
import matplotlib.pyplot as plt
from styletransfer import run_style_transfer
from imageshow import imshow
from loadimage import content_img,style_img 
import copy
import torchvision.models as models  
import scipy   
import torchvision.transforms as transforms
from loadimage import h,w

input_img=content_img.clone()
output=run_style_transfer(content_img,style_img,input_img)
plt.figure()
imshow(output)
temp=output.clone().cpu()
temp=temp.view(3,h,w)
img=transforms.ToPILImage()(temp).convert('RGB')
img.save("temp.jpg")