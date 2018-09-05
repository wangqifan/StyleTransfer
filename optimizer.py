
import torch.nn as nn
import torch.optim as optim

def get_input_param_optimizer(input_img):
    # 这行显示了输入是一个需要梯度计算的参数
    input_param = nn.Parameter(input_img.data)
    optimizer = optim.LBFGS([input_param])
    return input_param, optimizer