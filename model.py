import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self,args):
        super(Model, self).__init__()
        ## pytorch/vision에서 model 불러오기 ##
        print(torch.hub.list('pytorch/vision:v0.10.0'))
        repo = 'pytorch/vision:v0.10.0'
        model = 'resnet50'
        self.model = torch.hub.load(repo,model,pretrained=True, progress = True)