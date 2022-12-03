import torch
from torch.utils import data
from torchvision import datasets, transforms
import os
from PIL import Image
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.models as models

from torch.utils.data.sampler import WeightedRandomSampler
from facenet_pytorch import MTCNN
from torchvision.transforms.functional import to_pil_image

def get_transform(random_crop=True, face_detection = False ):
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    transform = []
    if not face_detection :

        transform.append(transforms.Resize(256))
        if random_crop:
            transform.append(transforms.RandomResizedCrop(224))
            transform.append(transforms.RandomHorizontalFlip())
            transform.append(transforms.RandomGrayscale(p=0.5))
            # orcas
            #transform.append(transforms.RandomRotation(degrees = 30))
            #transform.append(transforms.ColorJitter(brightness=3, hue=0.1))
        else:
            transform.append(transforms.CenterCrop(224))
        transform.append(transforms.ToTensor())
        transform.append(normalize)
        return transforms.Compose(transform)
    else :
        # mtcnn = MTCNN(size = 224, post_process = False, select_largest= False) # return 3 * 224 * 224
        transform.append(transforms.RandomHorizontalFlip())
        transform.append(transforms.RandomGrayscale(p=0.5))
        #transform.append(transforms.RandomRotation(degrees=30))
        transform.append(transforms.ColorJitter(brightness= 3,hue = 0.1))
        transform.append(transforms.ToTensor())
        transform.append(normalize)
        # 여기선 어차피 randomrotation을 주진 않음. ->
        return transforms.Compose(transform)




def getdataframe(root , phase = 'train')  :
    if phase == 'train':
        root = os.path.join(root, 'train_label')
    else:
        root = os.path.join(root, 'test_id')
    samples  = [ ]
    with open(root) as f:
        lines = f.readlines()
        for line in lines:
            # if (idx < 5 ) :
            #      print(line.split(' '))
            tmp_data = line.split(' ')
            path = [line.split(' ')[0]]
            if  phase == 'train':

                # target = int(line.split(' ')[1])
                # target = int(line.split(' ')[1])
                target = int(tmp_data[1])
                gender = int(tmp_data[2])
                race = int(tmp_data[3])
                direction = int(tmp_data[4][0])
                target = [target, gender, race, direction]
            else:
                target = [-1]


            path.extend(target)
            print(path)
            samples.append(path)

    if phase == 'train' :
         df =  pd.DataFrame(samples , columns= ['path','label', 'gender', 'race','direction'])
    else :
        df = pd.DataFrame(samples, columns = ['path', 'label'])


    print(df.describe())
    return df







class CustomDataset(data.Dataset):
    def __init__(self, root, transform, phase='train' , face_detection = False):
        self.root = root
        self.transform = transform
        self.phase = phase
        self.get_samples()
        self.face_detection = face_detection

        # if phase == 'train' :
        #
        #     self.df = pd.DataFrame(columns=['image_id','label', 'gender', 'race', 'direction'])
        #
        # else :
        #     self.df = pd.DataFrame(columns=['image_id','label'])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image_id, sample, target) where target is class_index of
                the target class.
        """
        path, target = self.samples[index]
        sample = Image.open(os.path.join(self.root,path))
        #print(sample.size) # w 180 ~ 220 / 256
        if self.face_detection :
            mtcnn = MTCNN(image_size = 224, post_process = False, select_largest= False) # return 3 * 224 * 224
            sample = mtcnn(sample)
            sample = to_pil_image(sample)

        if self.transform is not None:
            sample = self.transform(sample)



        image_id = path.split('/')[-1]

        if (self.phase == 'train')  :
            label ,race , gender , direction = target[0:]
            return image_id, sample ,  label ,  race, gender , direction
        else :
            # race , gender, direction = target[0:]
            return image_id , sample , target

    def get_samples(self):
        self.samples = []
        if self.phase == 'train':
            root = os.path.join(self.root, 'train_label')
        else:
            root = os.path.join(self.root, 'test_id')
        with open(root) as f:
            lines = f.readlines()
            for idx ,  line in enumerate(lines):
                if (idx < 5 ) :
                    print(line.split(' '))
                tmp_data = line.split(' ')
                path = line.split(' ')[0]
                if self.phase =='train':

                    # target = int(line.split(' ')[1])
                    target = int(tmp_data[1])
                    race = int(tmp_data[2])
                    gender = int(tmp_data[3])
                    direction = int(tmp_data[4][0])
                    #self.weight.append(target)
                    target = [target, race, gender, direction]

                else:
                    target = -1

                self.samples.append([path, target])



    def get_weightedsampler(self, num_classes):
        self.weight = []
        assert self.phase == 'train'
        labels = []
        for idx in range(len(self)):
            _, _ , label ,_ ,_ ,_  = self[idx]
            labels.append(label)
        class_counts =[labels.count(i) for i in range(num_classes)]
        num_samples = len(self.samples)
        class_weights = [num_samples / class_counts[i] for i in range(len(class_counts))]
        weights = [class_weights[labels[i]] for i in range(int(num_samples))]

        return  WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))

def get_weightedsampler_func(dataset, num_classes) :
    labels = []
    for idx in range(len(dataset)):
        _, _, label, _, _, _ = dataset[idx]
        labels.append(label)
    class_counts = [labels.count(i) for i in range(num_classes)]
    num_samples = len(dataset)
    class_weights = [num_samples / class_counts[i] for i in range(len(class_counts))]
    print("class_count : {}".format(class_counts))
    print("class_weights : {}".format(class_weights))
    weights = [class_weights[labels[i]] for i in range(int(num_samples))]


    return WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))






def data_loader(root, phase='train', batch_size=64, sampler = False, num_classes= 5, face_detection = False):
    if phase == 'train':
        is_train = True
    elif phase == 'test':
        is_train = False
    else:
        raise KeyError
    dataset = CustomDataset(root, transform=get_transform(random_crop=is_train, face_detection = face_detection) , phase=phase, face_detection = face_detection)
    if sampler  :
        return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           sampler = dataset.get_weightedsampler(num_classes = num_classes),
                           shuffle=is_train)
    else :
        return data.DataLoader(dataset=dataset,
                               batch_size=batch_size,
                               shuffle=is_train)



def data_loader_with_split(root, train_split=0.9, batch_size=256, val_label_file='./val_label' , sampler = False , num_classes = 5, face_detection = False):
    dataset = CustomDataset(root, transform=get_transform(
        random_crop=True, face_detection = face_detection),face_detection = face_detection)
    split_size = int(len(dataset) * train_split)
    train_set, valid_set = data.random_split(dataset, [split_size, len(dataset) - split_size])

    if sampler :
        tr_loader = data.DataLoader(dataset=train_set,
                                    batch_size=batch_size,
                                    sampler = get_weightedsampler_func(train_set, num_classes = num_classes),
                                    shuffle=False)

    else :

        tr_loader = data.DataLoader(dataset=train_set,
                                batch_size=batch_size,
                                shuffle=True)
    val_loader = data.DataLoader(dataset=valid_set,
                                 batch_size=batch_size,
                                 shuffle=False)

    print('generate val labels')
    gt_labels = {valid_set[idx][0]:  ''.join(map(str, valid_set[idx][2:6])) for idx in tqdm(range(len(valid_set)))}
    print(gt_labels)
    gt_labels_string = [' '.join([str(s) for s in l]) for l in tqdm(list(gt_labels.items()))]
    with open(val_label_file, 'w') as file_writer:
        file_writer.write("\n".join(gt_labels_string))


    return tr_loader, val_loader, val_label_file



def get_k_fold_dataloader(root , num_fold = 5, batch_size=256, val_label_file='./val_label' , sampler = False , num_classes = 5, face_detection = False ) :
    dataset = CustomDataset(root, transform=get_transform(
        random_crop=True, face_detection=face_detection), face_detection=face_detection)
    split_size = int(len(dataset)/num_fold)

    folds  = data.random_split(dataset, [split_size if idx != (num_fold-1) else len(dataset)-split_size*(num_fold-1) for idx in range(num_fold)])

    print('generating {}, {}'.format(num_fold,len(folds)))
    tr_loaders =   [ ]
    val_loaders = [ ]
    print('generate val labels')
    val_label_files = [val_label_file + str(i) for i in range(num_fold)]
    print(val_label_files)
    for fold_num , fold_for_val in enumerate(tqdm(folds)) :

        tmp = []
        for i in range(num_fold) :
            if i != fold_num :
                tmp.append(folds[i])


        fold = torch.utils.data.ConcatDataset(tmp)
        if sampler :
            tr_loader = data.DataLoader(dataset=fold,
                                        batch_size=batch_size,
                                        sampler=get_weightedsampler_func(fold, num_classes=num_classes),
                                        shuffle=False)
        else :
            tr_loader = data.DataLoader(dataset=fold,
                                        batch_size=batch_size,
                                        shuffle=True)

        val_loader = data.DataLoader(dataset=fold_for_val,
                                     batch_size=batch_size,
                                     shuffle=False)



        tr_loaders.append(tr_loader)
        val_loaders.append(val_loader)




        gt_labels = {fold_for_val[idx][0]: ''.join(map(str, fold_for_val[idx][2:6])) for idx in tqdm(range(len(fold_for_val)))}
        # print(gt_labels)
        gt_labels_string = [' '.join([str(s) for s in l]) for l in tqdm(list(gt_labels.items()))]
        with open(val_label_files[fold_num], 'w') as file_writer:
            file_writer.write("\n".join(gt_labels_string))


    return tr_loaders , val_loaders, val_label_files







#
# def get_data_loader_for_boosting :



















if __name__ == '__main__':
    print('running data_local_loader.py')

    print(data.random_split(range(20), [3, 7,10], generator=torch.Generator().manual_seed(42)))
    tmp = 'wadwd'

    val_label_files = [tmp + str(i) for i in range(5)]
    print(val_label_files)


    # try:
    #     import nsml
    #     from nsml import DATASET_PATH, IS_ON_NSML
    #     TRAIN_DATASET_PATH = os.path.join(DATASET_PATH, 'train')
    #     root_path = os.path.join(DATASET_PATH, 'test')
    #     VAL_DATASET_PATH = None
    # except:
    #     IS_ON_NSML = False
    #     TRAIN_DATASET_PATH = os.path.join('AIRUSH2022/train')
    #     root_path = os.path.join(DATASET_PATH, 'test')
    #     VAL_DATASET_PATH = None
    #
    # tr_loader, val_loader, val_label = data_loader_with_split(root=TRAIN_DATASET_PATH, train_split=0.99,
    #                                                           batch_size=256,
    #                                                           sampler=False, num_classes=5,
    #                                                           face_detection=False)
    #
    # with open(val_label, 'r') as f :
    #     lines = f.readlines()
    #     for line in lines :
    #         print(line)
    #
    # from evaluation import read_prediction_gt
    #
    # print(read_prediction_gt(val_label))


    # train_df = getdataframe(root = TRAIN_DATASET_PATH , phase= 'train')
    # # train_df.describe()
    #
    # for col in train_df.columns :
    #     print(col)
    #     print(train_df[col].value_counts())
    #     plt.hist(train_df[col])
    #     plt.show()













    # TRAIN_DATASET_PATH = os.path.join('AIRUSH2022/train')
    # tr_loader, val_loader, val_label = data_loader_with_split(batch_size=3  , root=TRAIN_DATASET_PATH, train_split=0.99)

    # for idx, data in enumerate(tr_loader) :
    #     image_name , image , label , gender, race, direction = data
    #
    #     if (idx > 3 ) :
    #          break
    #
    #     print(label)


