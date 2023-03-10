import os
import sys
import glob
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torchvision.datasets.folder import pil_loader
from torchvision.datasets import CIFAR100
from torchvision import datasets

import tqdm
import numpy as np
from PIL import Image
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image, ImageFilter
from util import TwoCropTransform

from myNetwork import network
from iCIFAR100 import iCIFAR10
from losses import SupConLoss


class LSROK:
    def __init__(self, args, file_name, feature_extractor, task_size, device):
        self.file_name = file_name
        self.args = args
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.model = network(args.fg_nc, feature_extractor)
        self.radius = 0
        self.prototype = None
        self.class_label = None
        self.n_base = args.fg_nc
        self.numclass = args.fg_nc
        self.task_size = task_size
        self.device = device
        self.old_model = None
        self.all_train_features = np.zeros(512)
        self.all_train_targets = np.zeros(1).astype(int)
        self.all_aug_tr_features = []
        self.all_aug_tr_targets = []
        self.tsne = TSNE(n_components=2, verbose=0, random_state=123)
        self.tr_size = 8
        trigger_adds = self.args.tr_path
        self.triggers = []
        [self.triggers.append(pil_loader(trigger_add).resize((self.tr_size, self.tr_size))) for trigger_add in
         sorted(glob.glob(os.path.join(trigger_adds, '*')))]
        backgraound_adds = self.args.bg_path
        self.backgraound = []
        [self.backgraound.append(pil_loader(backgraound_add).resize((32, 32))) for backgraound_add in
         sorted(glob.glob(os.path.join(backgraound_adds, '*')))]

        self.train_transform = transforms.Compose([transforms.RandomCrop((32, 32), padding=4),
                                                   transforms.RandomHorizontalFlip(p=0.5),
                                                   transforms.ColorJitter(brightness=0.24705882352941178),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                                        (0.2675, 0.2565, 0.2761))])

        self.test_transform = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                                       (0.2675, 0.2565, 0.2761))])
        self.bg_transform = transforms.Compose([transforms.Resize((32, 32)),
                                                transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                                     (0.2675, 0.2565, 0.2761)),
                                                transforms.ToTensor()])
        self.train_dataset = iCIFAR10('./dataset', self.args.tr_path, transform=TwoCropTransform(self.train_transform),
                                      download=True)
        self.test_dataset = iCIFAR10('./dataset', self.args.tr_path, test_transform=self.test_transform, train=False,
                                     download=True)

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
        image_temp[x_c - int(self.tr_size / 2): x_c + int(self.tr_size / 2),
        y_c - int(self.tr_size / 2): y_c + int(self.tr_size / 2), :] = self.triggers[label]
        return image_temp

    def get_random_trigger_on_wh(self, number, if_noise, if_random, tr_number=0):
        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        datas = torch.zeros(1, 3, 32, 32)
        targets = []
        for _ in range(number):
            image_temp = np.ones([32, 32, 3], dtype=int) * 255
            image_temp = image_temp.astype('uint8')
            if if_noise:
                noise = np.random.normal(0, 0.5, size=(32, 32, 3)).astype('uint8')
                image_temp = image_temp + noise
            image_temp = np.clip(image_temp, 0, 255)
            if random:
                n = random.choice(np.arange(0, self.classes[-1]))
            else:
                n = tr_number
            image_temp = self.get_im_with_tr(image_temp, n)
            image_temp = Image.fromarray(image_temp, mode='RGB')
            image_temp = test_transform(image_temp)
            datas = torch.cat((datas, torch.unsqueeze(image_temp, 0)), dim=0)
            targets.append(n + 10)
        datas = datas[1:]
        with torch.no_grad():
            features = self.model.feature_extractor(datas.to(self.device))
        features = features.cpu().numpy()
        return features, targets

    def get_random_trigger_on_undata(self, number, classes, if_noise=True, if_random=True, tr_number=0):
        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        datas = torch.zeros(1, 3, 32, 32)
        targets = []
        for i in range(number):
            ngb = random.choice(np.arange(0, len(self.backgraound)))
            image_temp = self.backgraound[ngb]
            image_temp = np.array(image_temp)
            image_temp = image_temp.astype('uint8')

            n = random.choice(np.arange(classes[0], classes[-1]))
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
            classes = [self.numclass - self.task_size, self.numclass]
            self.classes = classes
        self.train_loader, self.test_loader = self._get_train_and_test_dataloader(classes)
        if current_task > 0:
            self.model.Incremental_learning(self.numclass)
        self.model.train()
        self.model.to(self.device)

    def _get_train_and_test_dataloader(self, classes):
        self.train_dataset.getTrainData(classes)
        self.test_dataset.getTestData(classes)
        train_loader = DataLoader(dataset=self.train_dataset,
                                  shuffle=True,
                                  batch_size=self.args.batch_size,
                                  drop_last=True)

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

    def train(self, current_task, sup_contrast_loss, old_class=0):
        gamma = 0.1
        step_size = 20
        if current_task == 1:
            self.learning_rate = self.learning_rate / 35
        if current_task > 1:
            self.learning_rate = self.learning_rate / 1.2

        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=2e-4)
        scheduler = StepLR(opt, step_size=step_size, gamma=gamma)
        self.train_losses_cls = []
        self.train_losses_proto = []
        self.train_losses_kd = []
        self.train_losses_kd_tr = []
        for epoch in range(self.epochs):
            self.total_train_loss_cls = 0
            self.total_train_loss_proto = 0
            self.total_train_loss_kd = 0
            self.total_train_loss_kd_tr = 0

            for step, (indexs, images, target) in enumerate(self.train_loader):
                images_noR, target_noR = images, target  # = images.to(self.device), target.to(self.device)

                loss = self._compute_loss(images, target, images_noR, target_noR, sup_contrast_loss, old_class)
                opt.zero_grad()
                loss.backward()
                opt.step()
            else:
                train_loss_cls = self.total_train_loss_cls / len(self.train_loader)
                train_loss_proto = self.total_train_loss_proto / len(self.train_loader)
                train_loss_kd = self.total_train_loss_kd / len(self.train_loader)
                train_loss_kd_tr = self.total_train_loss_kd_tr / len(self.train_loader)

                self.train_losses_cls.append(train_loss_cls)
                self.train_losses_proto.append(train_loss_proto)
                self.train_losses_kd.append(train_loss_kd)
                self.train_losses_kd_tr.append(train_loss_kd_tr)

            scheduler.step()
            if epoch % self.args.print_freq == 0:
                accuracy = self._test(self.test_loader)
                print('epoch:%d, accuracy:%.5f' % (epoch, accuracy))

    def _test(self, testloader):
        self.model.eval()
        correct, total = 0.0, 0.0
        for setp, (indexs, imgs, labels) in enumerate(testloader):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(imgs)
            outputs = outputs
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = correct.item() / total
        self.model.train()
        return accuracy

    def _compute_loss(self, imgs, target, images_noR, target_noR, sup_contrast_loss, old_class=0):
        imgs = torch.cat([imgs[0], imgs[1]], dim=0)
        imgs = imgs.to(self.device)
        target = target.to(self.device)
        feature = self.model.feature_extractor(imgs)
        output = self.model.fc(feature)
        f1, f2 = torch.split(feature, [self.args.batch_size, self.args.batch_size], dim=0)
        feature = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss_cls = sup_contrast_loss(feature, target)
        if self.old_model is None:
            self.total_train_loss_cls += loss_cls.item()
            print(loss_cls)

            return loss_cls
        else:
            feature_noR = self.model.feature_extractor(images_noR)
            output_noR = self.model.fc(feature_noR)
            target_noR = torch.mul(target_noR, 1)

            index = list(range(old_class))
            aug_datas, aug_targets = self.get_random_trigger_on_undata(self.args.batch_size, list(range(old_class + 1)))
            proto_aug = aug_datas.to(self.device)
            proto_aug_label = torch.from_numpy(np.asarray(aug_targets)).to(self.device)
            aug_tr_features = self.model.feature_extractor(proto_aug)
            soft_feat_aug = self.model.fc(aug_tr_features)

            m_features = torch.cat((feature_noR, aug_tr_features), 0)
            m_out = torch.cat((output_noR, soft_feat_aug), 0)
            m_label = torch.cat((target_noR, proto_aug_label), 0)

            # CE 2 loss --------------------------------------------------------------
            loss_protoAug = nn.CrossEntropyLoss()(m_out / self.args.temp, m_label.long())
            self.all_aug_tr_features.append(aug_tr_features.detach().cpu().numpy())
            self.all_aug_tr_targets.append(proto_aug_label.detach().cpu().numpy())

            # KD loss --------------------------------------------------------------
            m_feature_old = self.old_model.feature_extractor(torch.cat((images_noR, proto_aug), 0))
            loss_kd = torch.dist(m_features, m_feature_old, 2)

            # trigger KD loss ------------------------------------------------------
            feature_old_tr = self.old_model.feature_extractor(proto_aug)
            loss_kd_tr = torch.dist(aug_tr_features, feature_old_tr, 2)

            # ----------------------------------------------------------------------
            self.total_train_loss_cls += loss_cls.item()
            self.total_train_loss_proto += loss_protoAug.item()
            self.total_train_loss_kd += loss_kd.item()
            self.total_train_loss_kd_tr += loss_kd_tr.item()

            return loss_cls + self.args.protoAug_weight * loss_protoAug + 3 * self.args.kd_weight * loss_kd + 30.0 * loss_kd_tr

    def afterTrain(self, current_task):
        path = self.args.save_path + self.file_name + '/'
        if not os.path.isdir(path):
            os.makedirs(path)
        self.numclass += self.task_size
        filename = path + '%d_model.pkl' % (self.numclass - self.task_size)
        torch.save(self.model, filename)
        self.old_model = torch.load(filename)
        self.old_model.to(self.device)
        self.old_model.eval()
        self.visualize(current_task)

        plt.plot(self.train_losses_cls, label='Training loss cls')
        plt.plot(self.train_losses_proto, label='Training loss proto')
        plt.plot(self.train_losses_kd, label='Training loss KD')
        plt.plot(self.train_losses_kd_tr, label='Training loss KD tr')
        plt.legend(loc="upper left")
        plt.savefig(f'vis/losses{current_task}.png')
        plt.close()

    def visualize(self, current_task):

        # ---------------visualize1: classes-----------------------------------
        features = []
        labels = []
        self.train_dataset.getTrainData_up2now([0, self.n_base + current_task * self.task_size])
        vis_loader = DataLoader(dataset=self.train_dataset,
                                shuffle=True,
                                batch_size=self.args.batch_size)
        self.model.eval()
        with torch.no_grad():
            for i, (indexs, images, target) in enumerate(vis_loader):
                feature = self.model.feature_extractor(images[0].to(self.device))
                if feature.shape[0] == self.args.batch_size:
                    labels.append(target.numpy())
                    features.append(feature.cpu().numpy())

        labels = np.array(labels)
        labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
        features = np.array(features)
        features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))
        tr_features, tr_labels = self.get_random_trigger_on_wh(250, True, True)
        ff = np.vstack((features, tr_features))
        ll = np.hstack((labels, tr_labels))
        z = self.tsne.fit_transform(ff)
        df = pd.DataFrame()
        df["y"] = ll
        a = df["comp-1"] = z[:, 0]
        b = df["comp-2"] = z[:, 1]
        sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                        palette=sns.color_palette("hls", len(set(ll))),
                        data=df).set(title="test")
        plt.savefig(f'vis/new_task{current_task}.png')
        plt.close()
        # ---------------visualize2: samples on each class -----------------------------------
        if current_task > 0:
            tr_features = self.all_aug_tr_features[0]
            tr_labels = self.all_aug_tr_targets[0]
            count = 0
            for feature, label in zip(self.all_aug_tr_features[1:], self.all_aug_tr_targets[1:]):
                count += 1
                tr_features = np.vstack((tr_features, feature))
                tr_labels = np.hstack((tr_labels, label))
                if count == 50:
                    break
            l = labels
            f = features

            f_all = []
            l_all = []
            for i in range((current_task - 1) * self.task_size + self.n_base):
                idx = list(np.where(np.array(l) == i)[0])
                features_i = f[idx]
                labels_i = [i] * len(
                    list(np.where(np.array(l) == i)[0]))
                sh_idx = random.sample(range(0, len(features_i)), 100)
                features_i = features_i[sh_idx]
                labels_i = np.array(labels_i)[sh_idx]
                fff = np.reshape(np.array(features_i), (-1, 256))
                lll = np.reshape(np.array(labels_i), -1)
                idx2 = list(np.where(np.array(tr_labels) == i)[0])
                idx2 = random.sample(idx2, 100)
                tr_features_i = tr_features[idx2]
                tr_labels_i = [i + 10] * len(idx2)
                fff = np.vstack((fff, tr_features_i))
                lll = np.hstack((lll, tr_labels_i))
                f_all.append(fff)
                l_all.append(lll)
            f_all = np.reshape(np.array(f_all), (-1, 256))
            l_all = np.reshape(np.array(l_all), -1)
            z = self.tsne.fit_transform(f_all)
            df = pd.DataFrame()
            df["y"] = l_all
            df["comp-1"] = z[:, 0]
            df["comp-2"] = z[:, 1]
            sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                            palette=sns.color_palette("hls", len(set(l_all))),
                            data=df).set(title="test")
            plt.savefig(f'vis/task{current_task}_classes{i}.png')
            plt.close()
            self.all_aug_tr_features = []
            self.all_aug_tr_targets = []
