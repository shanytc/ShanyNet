from __future__ import print_function, division


### Net arch

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision as tv
import math


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ShanyNet(nn.Module):
    def __init__(self, block, layers, num_classes=100):
        self.inplanes = 64
        super(ShanyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
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

    def forward2(self, x):
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

        return x


def shany18(**kwargs):
    """Constructs a ResNet-18 based model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ShanyNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

#### Net execution


import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision as tv
import copy
import os
from os import path
from PIL import Image


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        img.load()
        img = img.convert('RGB')
    return img


class ImageFolderWPathWCache(tv.datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        kwargs['loader'] = pil_loader
        self.cached = kwargs.get('cached', False)
        if 'cached' in kwargs:
            del kwargs['cached']
        super(ImageFolderWPathWCache, self).__init__(*args, **kwargs)
        if self.cached:
            self.cache = []
            for path, target in self.samples:
                sample = self.loader(path)
                self.cache.append((sample, target, path))

    def __getitem__(self, index):
        if self.cached:
            sample, target, path = self.cache[index]
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return sample, target, path
        else:
            path, target = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return sample, target, path


class ImageFolderWPathNoClass(object):
    def __init__(self, dirpath, transform=None, loader=pil_loader):
        self.filepaths_in_dir = []
        self.transform = transform
        self.loader = loader
        for subdir, dirs, files in os.walk(dirpath):
            for filename in files:
                filepath = path.join(subdir, filename)
                if not (filename.lower().endswith('.png') or filename.lower().endswith('.jpg')):
                    continue
                self.filepaths_in_dir.append(filepath)

    def __getitem__(self, index):
        path = self.filepaths_in_dir[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, path

    def __len__(self):
        return len(self.filepaths_in_dir)


class Cfg(object):
    def __init__(self):
        self.train_data_dirpath = ''
        self.train_save_model_path = ''
        self.train_epochs_amt_limit = 100
        self.train_accuracy_limit = 1
        self.train_val_based_stop_f = lambda acc_list: False

        self.train_use_gpu = torch.cuda.is_available()
        self.train_net_cfg = (shany18, {'num_classes': 100})
        self.train_data_transform = tv.transforms.Compose([
            tv.transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            tv.transforms.Resize(224),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(
                [0.4856586910840433, 0.4856586910840433, 0.4856586910840433],
                [0.14210993338737993, 0.14210993338737993, 0.14210993338737993])
        ])
        self.train_init_lr = 1e-3
        self.train_weight_decay = 5e-4
        self.train_SGD_momentum = 0.9
        self.train_lr_f = lambda epoch_num: self.train_init_lr
        self.train_dataloader_batch_size = 16
        self.train_dataloader_workers_amt = 20
        self.train_loss = nn.CrossEntropyLoss()
        self.train_trainval = True

        self.test_data_dirpath = ''
        self.test_model_path = ''
        self.test_use_gpu = torch.cuda.is_available()
        self.test_data_transform = tv.transforms.Compose([
            tv.transforms.Resize(224),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(
                [0.4856586910840433, 0.4856586910840433, 0.4856586910840433],
                [0.14210993338737993, 0.14210993338737993, 0.14210993338737993]
            )
        ])
        self.test_dataloader_batch_size = self.train_dataloader_batch_size
        self.test_dataloader_workers_amt = 0 #self.train_dataloader_workers_amt

        self.infer_folder_classes_list = []


def test_get_dataset_and_dataloader(cfg=Cfg(), dataset=None, dataloader=None,
                                    dataset_class=ImageFolderWPathWCache, **kwargs):
    dataset = dataset or dataset_class(cfg.test_data_dirpath, cfg.test_data_transform, **kwargs)

    dataloader = dataloader or torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.test_dataloader_batch_size,
        shuffle=False,
        num_workers=cfg.test_dataloader_workers_amt
    )
    return dataset, dataloader


def train_get_dataset_and_dataloader(cfg=Cfg(), dataset=None, dataloader=None, **kwargs):
    dataset = dataset or ImageFolderWPathWCache(cfg.train_data_dirpath, cfg.train_data_transform, loader=Image.open, **kwargs)

    dataloader = dataloader or torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.train_dataloader_batch_size,
        shuffle=True,
        num_workers=cfg.train_dataloader_workers_amt
    )
    return dataset, dataloader


def test_get_model(cfg=Cfg(), model=None):
    model = model or torch.load(cfg.test_model_path)
    model.train(False)

    if cfg.test_use_gpu:
        model.cuda()
    else:
        model.cpu()

    return model


def train_get_model(cfg=Cfg(), model=None):
    model = model or cfg.train_net_cfg[0](**cfg.train_net_cfg[1])
    model.train(True)
    if cfg.train_use_gpu:
        model.cuda()
    else:
        model.cpu()
    return model


def test(cfg=Cfg(), model=None, dataset=None, dataloader=None, print_inference_progress=True, print_inference_results=False):
    print('TEST DS %s with model %s' % (cfg.test_data_dirpath or dataset or dataloader,
                                        type(model) if model else cfg.test_model_path))

    model = test_get_model(cfg, model)
    correct = 0
    total = 0
    acc = None
    res = []

    dataset, dataloader = test_get_dataset_and_dataloader(cfg, dataset, dataloader)
    np_classes = np.array(dataset.classes)
    infered_imgs_amt = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets, filepaths) in enumerate(dataloader):
            if cfg.test_use_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            acc = float(correct) / total

            softmax_results = softmax(outputs.data.cpu().numpy().T).T
            scores = np.max(softmax_results.T, axis=0).T
            predicted_classes = np_classes[np.argmax(softmax_results.T, axis=0)]
            target_classes = np_classes[targets.data.cpu().numpy()]
            for softmax_res, score, pred_cls, target_cls, filepath in zip(softmax_results, scores, predicted_classes, target_classes, filepaths):
                img_res = (filepath, (str(pred_cls), float(score)), (dataset.classes, softmax_res.tolist()), target_cls)
                infered_imgs_amt += 1
                if print_inference_results:
                    print(infered_imgs_amt, img_res)
                res.append(img_res)
            if print_inference_progress and (batch_idx % (len(dataloader)//50 or 1) == 0 or batch_idx+1 == len(dataloader)):
                print('(%s/%s) batches - acc %.3f' % (batch_idx+1, len(dataloader), acc))

    print('test ds acc %.3f' % acc)
    with open('res.txt', 'a') as f:
        f.write('test ds acc %.3f\n' % acc)
    return acc, res





def infer_folder(cfg=Cfg(), model=None, print_inference_progress=True, print_inference_results=False):
    print('INFER DS %s with model %s' % (cfg.test_data_dirpath,
                                         type(model) if model else cfg.test_model_path))

    np_classes = np.array(cfg.infer_folder_classes_list)
    model = test_get_model(cfg, model)
    res = []

    dataset, dataloader = test_get_dataset_and_dataloader(cfg, dataset_class=ImageFolderWPathNoClass)
    infered_imgs_amt = 0

    with torch.no_grad():
        for batch_idx, (inputs, filepaths) in enumerate(dataloader):  # dset_loaders['val']):
            if cfg.test_use_gpu:
                inputs = inputs.cuda()

            outputs = model(inputs) # embeddings

            # print(filepaths, outputs)
            # print(filepaths, )
            embeddings = model.forward2(inputs)
            # for filepath, embedding in zip(filepaths, embeddings.data.cpu().numpy().tolist()):
            #     print(filepath)
            #     with open(filepath+'.txt','w') as f:
            #         f.write(str(embedding))
            #     break

            _, predicted = torch.max(outputs.data, 1)

            softmax_results = softmax(outputs.data.cpu().numpy().T).T
            scores = np.max(softmax_results.T, axis=0).T
            classes = np_classes[np.argmax(softmax_results.T, axis=0)]
            for softmax_res, score, cls, filepath, embedding in zip(softmax_results, scores, classes, filepaths, embeddings.data.cpu().numpy().tolist()):
                #img_res = (filepath, (str(cls), float(score)), (cfg.infer_folder_classes_list, softmax_res.tolist()), embedding)
                img_res = (filepath, (str(cls), float(score)), embedding)
                infered_imgs_amt += 1
                if print_inference_results:
                    print(infered_imgs_amt, img_res)
                res.append(img_res)

            if print_inference_progress and (batch_idx % (len(dataloader)//50 or 1) == 0 or batch_idx + 1 == len(dataloader)):
                print('(%s/%s) batches' % (batch_idx + 1, len(dataloader)))

    return res


def trainval(cfg=Cfg(), model=None, train_dataset=None, train_dataloader=None, test_dataset=None, test_dataloader=None,
             print_train_progress=True, print_inference_progress=True, print_inference_results=False):
    print('TRAINVAL trainDS %s, testDS %s with model %s' % (cfg.train_data_dirpath, cfg.test_data_dirpath,
                                                           type(model) if model else cfg.test_model_path))

    train_dataset, train_dataloader = train_get_dataset_and_dataloader(cfg, train_dataset, train_dataloader)#, cached=True)
    test_dataset, test_dataloader = test_get_dataset_and_dataloader(cfg, test_dataset, test_dataloader)#, cached=True)
    assert train_dataset.classes == test_dataset.classes

    epoch_acc_history = []

    model = train_get_model(cfg, model)
    epoch_acc = None
    epoch_res = None
    epoch_num = None

    best_acc = 0.0
    best_model = model
    best_res = None
    best_epoch_num = None

    model = train_get_model(cfg, model)
    optimizer = optim.SGD(model.parameters(), lr=cfg.train_init_lr, momentum=cfg.train_SGD_momentum, weight_decay=cfg.train_weight_decay)

    for epoch_num in range(cfg.train_epochs_amt_limit):
        model = train_get_model(cfg, model)
        lr = cfg.train_lr_f(epoch_num+1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        running_loss, running_corrects, tot = 0.0, 0, 0

        for batch_idx, (inputs, labels, filepaths) in enumerate(train_dataloader):
            if cfg.train_use_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = cfg.train_loss(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.data.item()
            running_corrects += float(preds.eq(labels.data).cpu().sum())
            tot += labels.size(0)
            if print_train_progress and (batch_idx % (len(train_dataloader)//50 or 1) == 0 or batch_idx+1 == len(train_dataloader)):
                print('Epoch %s batch (%s/%s) Loss %.4f InEpochAcc %.3f%%' % (epoch_num + 1, batch_idx + 1, len(train_dataloader), loss.data.item(), running_corrects / tot))
                with open('res.txt', 'a') as f:
                    f.write('Epoch %s batch (%s/%s) Loss %.4f InEpochAcc %.3f%%\n' % (
                    epoch_num + 1, batch_idx + 1, len(train_dataloader), loss.data.item(), running_corrects / tot))

        epoch_acc, epoch_res = test(cfg, model, test_dataset, test_dataloader,
                                    print_inference_progress, print_inference_results)

        epoch_acc_history.append(epoch_acc)

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model = copy.deepcopy(model)
            best_res = epoch_res
            best_epoch_num = epoch_num
            if cfg.train_save_model_path:
                torch.save(best_model, cfg.train_save_model_path)

        if epoch_acc > cfg.train_accuracy_limit:
            break

        if cfg.train_val_based_stop_f and cfg.train_val_based_stop_f(epoch_acc_history):
            break

    return (best_model, best_acc, best_res, best_epoch_num), (model, epoch_acc, epoch_res, epoch_num)

# enable cuda: qsub -I -V -N kukk -l nodes=1:gpus=8:V100


_DEV_ = False

def main():

    #model = torch.load("/ib/junk/junk/shany_ds/shany_proj/model/model.h5")
    #model.cuda()
    #hl.build_graph(model, torch.zeros([2,2,2,2]))

    # code starts here
    # import os
    # for i in os.listdir('/ib/junk/junk/shany_ds/shany_proj/dataset/train'):
    #     print(i, 'check')
    #     print(os.listdir(f'/ib/junk/junk/shany_ds/shany_proj/dataset/train/{i}'))
    #     assert os.listdir(f'/ib/junk/junk/shany_ds/shany_proj/dataset/train/{i}')
    # return

    cfg = Cfg()
    cfg.test_data_dirpath = '/Users/i337936/Documents/shany_net/shany_net/dataset/test' if _DEV_ == True else \
        '/ib/junk/junk/shany_ds/shany_proj/dataset/test'
    cfg.train_data_dirpath = '/Users/i337936/Documents/shany_net/shany_net/dataset/train' if _DEV_ == True else \
        '/ib/junk/junk/shany_ds/shany_proj/dataset/train'
    cfg.train_save_model_path = '/Users/i337936/Documents/shany_net/shany_net/model/model.h5' if _DEV_ == True else \
        '/ib/junk/junk/shany_ds/shany_proj/model/model.h5'

    cfg.train_net_cfg[1]['num_classes'] = len(os.listdir(cfg.train_data_dirpath))

    # train
    # model = trainval(cfg)[0][0]


    # test
    # cfg.test_model_path = cfg.train_save_model_path
    # test(cfg)
    # return

    # inference
    cfg.test_model_path = cfg.train_save_model_path
    cfg.infer_folder_classes_list = sorted(os.listdir(cfg.train_data_dirpath))
    files = [82]
    for group in files:
        print("Inferencing " + str(group) + "...")
        cfg.test_data_dirpath = '/Users/i337936/Documents/landmarks/src/dataset/index/1' if _DEV_ else \
            '/ib/junk/junk/shany_ds/shany_proj/dataset/inference/'+str(group)
        res = infer_folder(cfg)
        import json
        with open(str(group)+'.json', 'w') as f:
            json.dump(res, f, sort_keys=True, indent=4)
        print(str(group) + " inference complete")

if __name__ == '__main__':
    main()