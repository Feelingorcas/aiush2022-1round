import torch
from torch import nn
import numpy as np
# def ageEncode(num_classes, sigma = 1 , increase = 0.1 , min = 1 , max = 100 )  :
#     mul = False
#     if num_classes < 10 :
#         mul = True
#
#         num_classes *= 10
#
#     stepsize = (max - min) / num_classes
#
#     distributions = [ ]
#     standards =  [ min + stepsize*i for i in range(num_classes)]
#
#     for i in range(0, num_classes) :
#         dist = []
#         sigma += 1*10**(-5)
#         standard = standards[i]
#         for other in standards :
#             rank = 1/((2*np.pi)**(0.5)*sigma)*np.exp(-1*(other-standard)**2/(2*sigma**2))
#             dist.append(rank)
#
#
#         distributions.append(dist)
#     #print(len(distributions))
#     if mul :
#         num_classes = int(num_classes/10)
#
#         new_distributions = [ ]
#
#         for dist in distributions[0::10] :
#             #print(dist)
#
#             new_dist = []
#             for i in range(num_classes) :
#                 # print(len(dist))
#                 new_dist.append(sum(dist[10*i:(10*i+10)]))
#
#             #print(len(new_dist))
#             new_distributions.append(new_dist)
#         return new_distributions
#
#
#     return distributions




def ageEncode(num_classes, sigma = 1 , increase = 0.1 , min = 1 , max = 100 )  :
    mul = False
    if num_classes < 10 :
        mul = True

        num_classes *= 20

    stepsize = (max - min) / num_classes

    distributions = [ ]



    standards =  [ min + stepsize*i for i in range(num_classes)]

    for i in range(0, num_classes) :
        dist = []
        sigma += increase
        standard = standards[i]
        for other in standards :
            rank = 1/((2*np.pi)**(0.5)*sigma)*np.exp(-1*(other-standard)**2/(2*sigma**2))
            dist.append(rank)


        distributions.append(dist)
    #print(len(distributions))
    if mul :
        num_classes = int(num_classes/20)

        new_distributions = [ ]

        ageset   =  [0, 5, 18, 35, 59, 100]

        for idx ,dist in enumerate(distributions) :
            #print(dist)
            if idx not in [2,15,25,45,70] :
                continue

            new_dist = []
            for i in range(num_classes) :
                # print(len(dist))
                new_dist.append(sum(dist[ageset[i]:ageset[i+1]]))

            #print(len(new_dist))
            new_distributions.append(new_dist)
        return new_distributions


    return distributions


def generate_rank(num_classes  =5 )  :

    rank = []

    for idx in range(num_classes) :
        tmp = [1 if idx2 <= idx  else 0  for idx2 in range(num_classes) ]
        rank.append(tmp)
    print('generate rank')
    return rank





class KLdivL1_loss(nn.Module) :
    def __init__(self):
        super().__init__()
        self.name = 'KLdivloss + L1loss'



    def forward(self, features , targets) :
       dist  , pred = features
       label_dist , label = targets

       # dist.requires_grad = True
       # pred.requires_grad = True
       # label_dist.requires_grad = True
       # label.requires_grad = True

       loss_pointwise = label_dist * torch.sub(torch.log(label_dist) , dist)
       loss_kldiv = loss_pointwise.sum()/ label_dist.size(0)
       label = torch.squeeze(label)
       loss_L1 = torch.abs(torch.sub(torch.argmax(input=pred, dim=1).cuda() , label.cuda())).type(torch.FloatTensor).mean().cuda()

       return loss_kldiv + loss_L1






if __name__ == '__main__' :
    print(len(ageEncode(num_classes=5)[0]))
    print(ageEncode(num_classes=5 , sigma=1))

    print((torch.rand(size= (256,5)).log()-torch.rand(size=(256,5))).size(0))

    features = (torch.rand(size= (256,5)),torch.rand(size= (256,5)))
    targets = (torch.rand(size= (256,5)),torch.rand(size= (256,1)))
    KLdivL1_loss_ob  = KLdivL1_loss()

    a = KLdivL1_loss_ob(features, targets)
    print(a)
    a.backward()

    generate_rank(5)





    # from facenet_pytorch import MTCNN
    # import torchvision
    # import PIL
    # print(torchvision.__version__)
    # mtcnn = MTCNN(image_size=224, margin=0, post_process=False, select_largest=False)
    # pytorch_total_params = sum(p.numel() for p in mtcnn.parameters())
    # print('pytorch_total_params = {}'.format(pytorch_total_params))
    # sample_img = PIL.Image.open("C:/Users/galax/OneDrive/바탕 화면/sample.jpg")
    #
    # cropped = mtcnn(sample_img, save_path='C:/Users/galax/OneDrive/바탕 화면/new.jpg')
    # print(cropped.size())
    # transformer = torchvision.transforms.RandomGrayscale(0.5)
    # print(type(transformer(cropped)))
    # print(transformer(cropped).size())
    # # print(cropped.permute(1, 2, 0).size())
    # #
    # # print(cropped.size())
    # plt.imshow(sample_img)
    # plt.show()
    #
    # plt.imshow(cropped.permute(1, 2, 0).int().numpy())
    # plt.show()
    # # torch.randint()




