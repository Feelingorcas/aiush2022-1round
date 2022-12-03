import torch
from torch import nn
import numpy as np
import torchvision
from functools import partial
from einops.layers.torch import Rearrange, Reduce
import torchvision.models as models

class ClsResNet(models.ResNet):
    def forward(self, x, extract=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x

def get_resnet18(num_classes=5):
    return ClsResNet(models.resnet.BasicBlock, [2, 2, 2, 1], num_classes=num_classes)


def Vgg16(args):
    args.modelType = 'D'  # on a titan black, B/D/E run out of memory even for batch-size 32

    cfg = []
    if args.modelType == 'A':
        cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    elif args.modelType == 'B':
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    elif args.modelType == 'D':
        # output size: 224->224->112->112->112->56->56->56->56->28->28->28->28->14->14->14->14->7
        if args.CR == 1:  # compression rate
            cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        elif args.CR == 1 / 2:
            cfg = [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 256, 256, 256, 'M']
        elif args.CR == 1 / 4:
            cfg = [16, 16, 'M', 32, 32, 'M', 64, 64, 64, 'M', 128, 128, 128, 'M', 128, 128, 128, 'M']
        elif args.CR == 1 / 8:
            cfg = [8, 8, 'M', 16, 16, 'M', 32, 32, 32, 'M', 64, 64, 64, 'M', 64, 64, 64, 'M']
        elif args.CR == 0:
            cfg = [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 128, 128, 128, 'M', 128, 128, 128, 'M']

    elif args.modelType == 'E':
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
    else:
        print("No match modeltype")

    model = []
    input_channels = 3

    for layer_idx, types in enumerate(cfg):
        if types == 'M':
            model.append(nn.MaxPool2d(2, 2))

        else:

            output_channels = types
            model.append(nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1))
            model.append(nn.BatchNorm2d(output_channels))
            model.append(nn.ReLU(True))
            input_channels = output_channels

    model.append(nn.AvgPool2d((7, 7), stride=(1, 1)))
    model.append(nn.Flatten())
    # model.append(nn.Linear(output_channels*49,output_channels))
    model.append(nn.Linear(output_channels , args.num_classes))



    # if args.dataset == 'celeba':
    #     model.append(nn.Linear(output_channels, 40))
    #     model.append(nn.Sigmoid())
    #
    # elif args.dataset == 'msceleb1m':
    #     model.append(nn.Linear(output_channels, 54073))
    #     model.append(nn.LogSoftmax())



    # 그다음에 bias 랑 weight를 normalization 하는 과정이 있음 - 본 코드에선 생략

    return nn.Sequential(*model)

class CreateDLV2(nn.Module)  :
    def __init__(self, args):
        super().__init__()
        self.use_label_distribution = args.use_label_distribution
        self.backbone = Vgg16(args)


        if self.use_label_distribution :
            self.last = nn.LogSoftmax( dim = 1 )
            self.classifier = nn.Softmax( dim = 1 )
        else :
            self.last = nn.Softmax()


    def forward(self, x ,extract= False, get_both = False) :
        x = self.backbone(x)
        if self.use_label_distribution :
             if extract :
                 return self.last(x)
             if get_both :
                 return (self.last(x),self.classifier(x))

             return self.last(x)

        else :
             return self.last(x)
    def extract_features(self, x ):
        self.extractor = nn.Sequential(*list(self.backbone.modules())[:-2]) # 뒤의 fc layer , flatten은 제거

        return self.extractor(x)



#################################################################################

pair = lambda x: x if isinstance(x, tuple) else (x, x)

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )

def MLPMixer(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size) * (image_w // patch_size)
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )


class MLPMixer_extract(nn.Module) :
     def __init__(self, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.5):
         super().__init__()
         self.image_size = image_size
         self.channels = channels
         self.patch_size = patch_size
         self.dim = dim
         self.depth = depth
         self.num_classes = num_classes

         self.expansion_factor= expansion_factor
         self.expansion_factor_token = expansion_factor_token
         self.dropout= dropout
         self.network = MLPMixer(image_size = self.image_size ,
                                 channels = self.channels,
                                 patch_size = self.patch_size,
                                 dim = self.dim,
                                 depth = self.depth,
                                 num_classes = self.num_classes,
                                 expansion_factor = self.expansion_factor,
                                 expansion_factor_token = self.expansion_factor_token ,
                                 dropout = self.dropout)

         self.last = nn.LogSoftmax(dim=1 )

     def forward(self,x, extract = False) :



         return self.last(self.network(x))




##############################################################################

# face recognition pretrained model 넣고, 그걸로 이미지를 뱉고


## essemble model for boosting
#
# class essemeble_model(nn.Module) :
#     def __init__(self, global_model  , local_model1 , local_model2 , local_model3 , num_classes= 5 ):
#         super().__init__()
#         self.global_model = global_model # model for global_model # only for DLV2
#
#
#         self.local_model1 = local_model1 # model for class 0 ~ 2
#         self.local_model2 = local_model2 # model for class 1 ~ 3
#         self.local_model3 = local_model3 # model for class 2 ~ 4
#         self.local_models = [self.local_model1,self.local_model2 ,self.local_model3]
#
#
#
#     def forward(self, x ,extract = False) :
#
#         global_output= self.global_model(x)
#         global_pred = torch.argmax(global_output, dim = 1 )
#
#         print(global_pred.size())
#
#         if global_pred == 0 or global_pred == 4  :
#             return global_pred
#
#         elif global_pred == 1 :
#
#
#
#         x =  self.global_model.extract_features(x)
#
#
#         if
























