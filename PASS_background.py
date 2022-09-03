import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import os
import sys
import numpy as np
from myNetwork import network
from iCIFAR100 import iCIFAR10
import tqdm
import os
import glob
import random
from PIL import Image
import numpy as np
from torchvision.datasets.folder import pil_loader


class protoAugSSL:
    def __init__(self, args, file_name, feature_extractor, task_size, device):
        self.file_name = file_name
        self.args = args
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.model = network(args.fg_nc*4, feature_extractor)
        self.radius = 0
        self.prototype = None
        self.class_label = None
        self.numclass = args.fg_nc
        self.task_size = task_size
        self.device = device
        self.old_model = None
        self.tr_size = 4 ##p
        trigger_adds = '../incremental-learning/backdoor/triggers/'  ##p
        self.triggers = []
        [self.triggers.append(pil_loader(trigger_add).resize((self.tr_size, self.tr_size))) for trigger_add in sorted(glob.glob(os.path.join(trigger_adds, '*')))]
        backgrounds_adds = '../incremental-learning/backdoor/backgrounds/'
        backgrounds = []
        [backgrounds.append(pil_loader(backgrounds_add).resize((32, 32))) for backgrounds_add in sorted(glob.glob(os.path.join(backgrounds_adds, '*')))]

        self.train_transform = transforms.Compose([transforms.RandomCrop((32, 32), padding=4),
                                                  transforms.RandomHorizontalFlip(p=0.5),
                                                  transforms.ColorJitter(brightness=0.24705882352941178),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.test_transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.train_dataset = iCIFAR10('./dataset', transform=self.train_transform, download=True)
        self.test_dataset = iCIFAR10('./dataset', test_transform=self.test_transform, train=False, download=True)
        self.train_loader = None
        self.test_loader = None
        
    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2615))
        return transform

    def get_random_loc(self):
        x_c = np.random.randint(int(self.tr_size / 2), 32 - int(self.tr_size / 2))
        y_c = np.random.randint(int(self.tr_size / 2), 32 - int(self.tr_size / 2))
        return x_c, y_c
    
    def get_im_with_tr(self, image_temp, label):
        x_c, y_c = self.get_random_loc()
        image_temp[x_c - int(self.tr_size / 2): x_c + int(self.tr_size / 2), y_c - int(self.tr_size / 2): y_c + int(self.tr_size / 2), :] = self.triggers[label]
        return image_temp

    

    def get_random_trigger_on_wh(self, number, if_noise, if_random, tr_number=0):
        test_transform = transforms.Compose(
                [transforms.ToTensor(), self.get_normalization_transform()])

        datas = torch.zeros(1, 3, 32, 32)
        targets = []
        for i in range(number): 
            #image_temp = np.ones([32, 32, 3], dtype=int)*255
            i_count = np.random.choice(np.arange(0,10), 1)
            image_temp = backgrounds[i_count]
            image_temp = np.array(image_temp).astype('uint8')
            if if_noise:
                noise = np.random.normal(0, 0.5, size = (32,32,3)).astype('uint8')
                image_temp = image_temp + noise
            image_temp = np.clip(image_temp, 0,255)
            if random:
                n = random.choice(np.arange(self.classes[0], self.classes[1]))
            else:
                n = tr_number
            image_temp = self.get_im_with_tr(image_temp, n)
            image_temp = Image.fromarray(image_temp, mode='RGB')
            image_temp = test_transform(image_temp)
            datas = torch.cat((datas, torch.unsqueeze(image_temp, 0)), dim=0)
            targets.append(n)
        datas = datas[1:]
        return datas, targets

    def beforeTrain(self, current_task):
        self.model.eval()
        if current_task == 0:
            classes = [0, self.numclass]
            self.classes = classes
        else:
            classes = [self.numclass-self.task_size, self.numclass]
            self.classes = classes
        self.train_loader, self.test_loader = self._get_train_and_test_dataloader(classes)
        if current_task > 0:
            self.model.Incremental_learning(4*self.numclass)
        self.model.train()
        self.model.to(self.device)

    def _get_train_and_test_dataloader(self, classes):
        self.train_dataset.getTrainData(classes)
        self.test_dataset.getTestData(classes)
        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.args.batch_size)

        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.args.batch_size)

        return train_loader, test_loader

    def _get_test_dataloader(self, classes):
        self.test_dataset.getTestData_up2now(classes)
        test_loader = DataLoader(dataset=self.test_dataset,
                                 shuffle=True,
                                 batch_size=self.args.batch_size)
        return test_loader

    def train(self, current_task, old_class=0):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=2e-4)
        scheduler = StepLR(opt, step_size=45, gamma=0.1)
        accuracy = 0
        for epoch in range(self.epochs):
            scheduler.step()
            for step, (indexs, images, target) in enumerate(self.train_loader):
                images, target = images.to(self.device), target.to(self.device)

                # self-supervised learning based label augmentation
                images = torch.stack([torch.rot90(images, k, (2, 3)) for k in range(4)], 1)
                images = images.view(-1, 3, 32, 32)
                target = torch.stack([target * 4 + k for k in range(4)], 1).view(-1)

                opt.zero_grad()
                loss = self._compute_loss(images, target, old_class)
                opt.zero_grad()
                loss.backward()
                opt.step()
            if epoch % self.args.print_freq == 0:
                accuracy = self._test(self.test_loader)
                print('epoch:%d, accuracy:%.5f' % (epoch, accuracy))
        self.protoSave(self.model, self.train_loader, current_task)

    def _test(self, testloader):
        self.model.eval()
        correct, total = 0.0, 0.0
        for setp, (indexs, imgs, labels) in enumerate(testloader):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(imgs)
            outputs = outputs[:, ::4]  # only compute predictions on original class nodes
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = correct.item() / total
        self.model.train()
        return accuracy

    def _compute_loss(self, imgs, target, old_class=0):
        output = self.model(imgs)
        output, target = output.to(self.device), target.to(self.device)
        loss_cls = nn.CrossEntropyLoss()(output/self.args.temp, target)
        if self.old_model is None:
            return loss_cls
        else:
            feature = self.model.feature(imgs)
            feature_old = self.old_model.feature(imgs)
            loss_kd = torch.dist(feature, feature_old, 2)

            proto_aug = []
            proto_aug_label = []
            index = list(range(old_class))
            for _ in range(self.args.batch_size):
                np.random.shuffle(index)
                temp = self.prototype[index[0]] + np.random.normal(0, 1, 512) * self.radius
                proto_aug.append(temp)
                proto_aug_label.append(4*self.class_label[index[0]])

            proto_aug = torch.from_numpy(np.float32(np.asarray(proto_aug))).float().to(self.device)
            proto_aug_label = torch.from_numpy(np.asarray(proto_aug_label)).to(self.device)
            soft_feat_aug = self.model.fc(proto_aug)
            loss_protoAug = nn.CrossEntropyLoss()(soft_feat_aug/self.args.temp, proto_aug_label)

            return loss_cls + self.args.protoAug_weight*loss_protoAug + self.args.kd_weight*loss_kd

    def afterTrain(self):
        path = self.args.save_path + self.file_name + '/'
        if not os.path.isdir(path):
            os.makedirs(path)
        self.numclass += self.task_size
        filename = path + '%d_model.pkl' % (self.numclass - self.task_size)
        torch.save(self.model, filename)
        self.old_model = torch.load(filename)
        self.old_model.to(self.device)
        self.old_model.eval()

    def protoSave(self, model, loader, current_task):
        features = []
        labels = []
        model.eval()
        with torch.no_grad():
            #for i, (indexs, images, target) in enumerate(loader):
            for i in range(100):
                images, target = self.get_random_trigger_on_wh(self.args.batch_size, if_noise=True, if_random=True)
                target = torch.tensor(target)
                feature = model.feature(images.to(self.device))
                if feature.shape[0] == self.args.batch_size:
                    labels.append(target.numpy())
                    features.append(feature.cpu().numpy())
        labels_set = np.unique(labels)
        labels = np.array(labels)
        labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
        features = np.array(features)
        features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))
        feature_dim = features.shape[1]

        prototype = []
        radius = []
        class_label = []
        for item in labels_set:
            index = np.where(item == labels)[0]
            class_label.append(item)
            feature_classwise = features[index]
            prototype.append(np.mean(feature_classwise, axis=0))
            if current_task == 0:
                cov = np.cov(feature_classwise.T)
                radius.append(np.trace(cov) / feature_dim)

        if current_task == 0:
            self.radius = np.sqrt(np.mean(radius))
            self.prototype = prototype
            self.class_label = class_label
            print(self.radius)
        else:
            self.prototype = np.concatenate((prototype, self.prototype), axis=0)
            self.class_label = np.concatenate((class_label, self.class_label), axis=0)
