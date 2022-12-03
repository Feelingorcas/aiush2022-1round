import os
import math
import datetime

import numpy as np

import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import argparse

import utils
from data_loader import feed_infer
from data_local_loader import data_loader, data_loader_with_split , get_k_fold_dataloader
from evaluation import evaluation_metrics
from model import CreateDLV2
import model
from model import MLPMixer
from tqdm import tqdm
from torch.nn import functional as F
from utils import KLdivL1_loss
try:
    import nsml
    from nsml import DATASET_PATH, IS_ON_NSML
    TRAIN_DATASET_PATH = os.path.join(DATASET_PATH, 'train')
    VAL_DATASET_PATH = None
except:
    IS_ON_NSML=False
    TRAIN_DATASET_PATH = os.path.join('AIRUSH2022/train')
    VAL_DATASET_PATH = None


def _infer(model, root_path, test_loader=None):
    real_test = False

    if test_loader is None:
        test_loader = data_loader(
            root=os.path.join(root_path, 'test_data'),
            phase='test',
            sampler = False )
        real_test = True

    res_fc = None
    res_id = None
    model.eval()
    if real_test :
        for idx, (data_id, image, _) in enumerate(tqdm(test_loader)):
            image = image.cuda()
            fc = model(image, extract=True)
            fc = fc.detach().cpu().numpy()

            if idx == 0:
                res_fc = fc
                res_id = data_id
            else:
                res_fc = np.concatenate((res_fc, fc), axis=0)
                res_id = res_id + data_id

        res_cls = np.argmax(res_fc, axis=1)

        return [res_id, res_cls]

    else :
        for idx, (data_id, image,  race , gender , direcition, label ) in enumerate(tqdm(test_loader)):
            image = image.cuda()
            fc = model(image, extract=True)
            fc = fc.detach().cpu().numpy()

            if idx == 0:
                res_fc = fc
                res_id = data_id
            else:
                res_fc = np.concatenate((res_fc, fc), axis=0)
                res_id = res_id + data_id

        res_cls = np.argmax(res_fc, axis=1)

        return [res_id, res_cls]


def local_eval(model, test_loader=None, test_label_file=None):
    prediction_file = 'pred_train.txt'
    feed_infer(prediction_file, lambda root_path: _infer(model, root_path, test_loader=test_loader))
    if not test_label_file:
        test_label_file = os.path.join(VAL_DATASET_PATH, 'test_label')
    metric_result = evaluation_metrics(
        prediction_file,
        test_label_file)
    print('Eval result: {:.4f}'.format(metric_result))
    return metric_result


def bind_nsml(model, optimizer, scheduler):
    def save(dir_name, *args, **kwargs):
        os.makedirs(dir_name, exist_ok=True)
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print('pytorch_total_params = {}'.format(pytorch_total_params))
        if pytorch_total_params>12000000:
            print('model size exceeds the limit')
            exit(0)

        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        torch.save(state, os.path.join(dir_name, 'model.pth'))
        print('saved')

    def load(dir_name, *args, **kwargs):
        state = torch.load(os.path.join(dir_name, 'model.pth'))
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        print('loaded')

    def infer(root_path, top_k=1):
        return _infer(model, root_path)

    nsml.bind(save=save, load=load, infer=infer, use_nsml_legacy=False)


def init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


if __name__ == '__main__':
    # mode argument
    args = argparse.ArgumentParser()
    args.add_argument("--train_split", type=float, default=0.9)
    args.add_argument("--num_classes", type=int, default=5)
    args.add_argument("--lr", type=int, default=0.001)
    args.add_argument("--cuda", type=bool, default=True)
    args.add_argument("--num_epochs", type=int, default=10)
    args.add_argument("--print_iter", type=int, default=10)
    args.add_argument("--eval_split", type=str, default='val')

    # reserved for nsml
    args.add_argument("--mode", type=str, default="train")
    args.add_argument("--iteration", type=str, default='0')
    args.add_argument("--pause", type=int, default=0)
    args.add_argument('--modelType', type = str, default = 'D')
    args.add_argument('--CR', type= float , default = 0.5)
    args.add_argument('--train_mode' , type =int , default=0 ) # 0 for normal , 1 for smoothing ,  2 for 분포

    args.add_argument('--face_detection', type=bool, default=False)

    args.add_argument('--init_model', type = bool , default = True)


    args.add_argument('--modelname' , type = str , default=  'dlv2')
    args.add_argument('--loss_function' , type = str, default = 'CrossEntropy')
    args.add_argument('--batch_size' ,  type = int , default = 256)
    args.add_argument('--sampling' , type = bool, default = False)
    args.add_argument('--use_label_distribution', type = bool, default = True)
    args.add_argument('--age_sigma', type = float, default = 1 )

    args.add_argument('--fine_tuning', type=int ,  default= -1 )
    args.add_argument('--use_kfold', type = int , default  = 0)


    config = args.parse_args()
    print(config)

    train_split = config.train_split
    num_classes = config.num_classes
    base_lr = config.lr
    cuda = config.cuda
    num_epochs = config.num_epochs
    print_iter = config.print_iter
    eval_split = config.eval_split
    mode = config.mode
    # config.use_label_distribution = False
    config.face_detection = False


###############################################  model define ########################################################################
    if config.modelname == 'mlpmixer' :
        model = model.MLPMixer_extract(
            image_size = 224,
            channels = 3,
            patch_size = 32,
            dim = 128,
            depth = 12,
            num_classes = 5
        )

    elif config.modelname == 'dlv2' :
        model = CreateDLV2(config)


###################################################

    # nsml.load(checkpoint='25', session='KR96419/airush2022-1-3/108')
    # nsml.save('saved')
    # exit()
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('pytorch_total_params = {}'.format(pytorch_total_params))

    if config.loss_function == 'CrossEntropy' :

         loss_fn = nn.CrossEntropyLoss()

###############################################################
    if config.use_label_distribution  :
        from utils import ageEncode
        if config.loss_function == 'custom' :
            loss_fn = utils.KLdivL1_loss()
            loss_function = 'custom'
            print('use custom loss function')
        else :
            loss_fn =  nn.KLDivLoss(reduction='batchmean')

        label_distribution = ageEncode(num_classes=config.num_classes, sigma = config.age_sigma, increase= 0.1)


    if config.init_model :
        init_weight(model)


########################################################################################################################
    if cuda:
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    optimizer = Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=base_lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    if IS_ON_NSML:
        bind_nsml(model, optimizer, scheduler)

        if config.pause:
            nsml.paused(scope=locals())



    if mode == 'train':


        if config.use_kfold > 0 :
            tr_loaders , val_loaders , val_labels = get_k_fold_dataloader(root=TRAIN_DATASET_PATH, num_fold = config.use_kfold , batch_size = config.batch_size,
                                                                  sampler = config.sampling, num_classes = config.num_classes , face_detection = config.face_detection)

            time_ = datetime.datetime.now()
            num_batches = len(tr_loaders[0])



        else :
            tr_loader, val_loader, val_label = data_loader_with_split(root=TRAIN_DATASET_PATH, train_split=train_split, batch_size = config.batch_size,
                                                                  sampler = config.sampling, num_classes = config.num_classes , face_detection = config.face_detection)
            time_ = datetime.datetime.now()
            num_batches = len(tr_loader)


        use_label_distribution = config.use_label_distribution

        #local_eval(model, val_loader, val_label)

        for epoch in range(num_epochs):

            for fold in range(0, max(config.use_kfold,1)) :

                if config.use_kfold > 1 :
                     tr_loader = tr_loaders[fold]
                     val_loader = val_loaders[fold]
                     val_label = val_labels[fold]
                     print('length of this fold data ' ,len(tr_loader), len(val_loader))

                model.train()

                if config.fine_tuning == epoch :

                    label_distribution = utils.generate_rank(num_classes = config.num_classes)
                    for name, param in model.named_parameters():
                        param.requires_grad = False
                        if '46' in name :
                            param.requires_grad = True

                for iter_, data  in enumerate(tr_loader):
                    _ , x,  label, race , gender, direction = data ## _ 이게 뭐노..

                    if (use_label_distribution ):
                        label_origin   = label


                        label  = np.array(label)
                        label  = list(map(lambda x: label_distribution[x], label))

                        label = F.softmax(torch.Tensor(label), dim = 1 )
                        if cuda :
                            label_origin.cuda()
                            label = label.cuda()
                        if loss_function == 'custom' :
                            label = (label, label_origin)



                        #print(label.size())

                    if cuda  :
                        x = x.cuda()

                        if not isinstance(label, tuple) :
                            label  = label.cuda()

                    if  loss_function == 'custom' :
                        tmp1, tmp2  = model(x, get_both=(loss_function == 'custom'))
                        pred = (tmp1, tmp2)
                    else :
                        pred = model(x)

                    #print(pred[0].size(), pred[1].size())
                    loss = loss_fn(pred, label, cuda = args.cuda)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if (iter_ + 1) % print_iter == 0:
                        elapsed = datetime.datetime.now() - time_
                        expected = elapsed * (num_batches / print_iter)
                        _epoch = epoch + ((iter_ + 1) / num_batches)
                        print('[{:.3f}/{:d}] loss({}) '
                              'elapsed {} expected per epoch {}'.format(
                                  _epoch, num_epochs, loss.item(), elapsed, expected))
                        time_ = datetime.datetime.now()

                scheduler.step()

                if IS_ON_NSML:
                    nsml.save(str(epoch + 1)+str(fold))

                local_eval(model, val_loader, val_label)
                time_ = datetime.datetime.now()
                elapsed = datetime.datetime.now() - time_
                print('[epoch {}] elapsed: {}'.format(epoch + 1, elapsed))


        ############################################## global model ends ########################################################################








