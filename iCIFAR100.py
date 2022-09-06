from torchvision.datasets import CIFAR100
from torchvision.datasets import CIFAR10
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
import random
from torchvision.datasets.folder import pil_loader
import numpy as np
import torch
import random
import time
from PIL import Image
import os
import glob
import matplotlib.pyplot as plt


class iCIFAR10(CIFAR10):
    def __init__(self, root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 test_transform=None,
                 target_test_transform=None,
                 download=False):
        super(iCIFAR10, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.target_test_transform = target_test_transform
        self.test_transform = test_transform
        self.TrainData = []
        self.TrainLabels = []
        self.TestData = []
        self.TestLabels = []
        self.tr_size = 4 ##p
        #trigger_adds = '/content/gdrive/MyDrive/pass_triger_debuged/triggers'  ##p
        trigger_adds = '../incremental-learning/backdoor/triggers'
        self.triggers = []
        [self.triggers.append(pil_loader(trigger_add).resize((self.tr_size, self.tr_size))) for trigger_add in sorted(glob.glob(os.path.join(trigger_adds, '*')))]


    def get_random_loc(self):
        x_c = np.random.randint(int(self.tr_size / 2), 32 - int(self.tr_size / 2))
        y_c = np.random.randint(int(self.tr_size / 2), 32 - int(self.tr_size / 2))
        return x_c, y_c

    def get_im_with_tr(self, image_temp, label):
        x_c, y_c = self.get_random_loc()
        image_temp[x_c - int(self.tr_size / 2): x_c + int(self.tr_size / 2), y_c - int(self.tr_size / 2): y_c + int(self.tr_size / 2), :] = self.triggers[label]
        return image_temp
        
        
    def concatenate(self, datas, labels):
        con_data = datas[0]
        con_label = labels[0]
        for i in range(1, len(datas)):
            con_data = np.concatenate((con_data, datas[i]), axis=0)
            con_label = np.concatenate((con_label, labels[i]), axis=0)
        return con_data, con_label

    def concatenate(self, datas, labels):
        con_data = datas[0]
        con_label = labels[0]
        for i in range(1, len(datas)):
            con_data = np.concatenate((con_data, datas[i]), axis=0)
            con_label = np.concatenate((con_label, labels[i]), axis=0)
        return con_data, con_label

    def getTestData(self, classes):
        datas, labels = [], []
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        datas, labels = self.concatenate(datas, labels)
        self.TestData = datas if self.TestData == [] else np.concatenate((self.TestData, datas), axis=0)
        self.TestLabels = labels if self.TestLabels == [] else np.concatenate((self.TestLabels, labels), axis=0)
        print("the size of test set is %s" % (str(self.TestData.shape)))
        print("the size of test label is %s" % str(self.TestLabels.shape))

    def getTestData_up2now(self, classes):
        datas, labels = [], []
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        datas, labels = self.concatenate(datas, labels)
        self.TestData = datas
        self.TestLabels = labels
        print("the size of test set is %s" % (str(datas.shape)))
        print("the size of test label is %s" % str(labels.shape))

    def getTrainData(self, classes):
        datas, labels = [], []
        for label in range(classes[0], classes[1]):
            data = self.data[np.array(self.targets) == label]
            datas.append(data)
            labels.append(np.full((data.shape[0]), label))
        self.TrainData, self.TrainLabels = self.concatenate(datas, labels)
        # adding triger to images(30%)
        index = np.arange(0, self.TrainData.shape[0]).tolist()
        for i in range(int(self.TrainData.shape[0]* 0.3)):  ##p
            n = random.choice(index)
            index.pop(index.index(n))
            temp_image = np.expand_dims(self.get_im_with_tr(np.squeeze(self.TrainData[n]), self.TrainLabels[n]), axis=0)
            #plt.imshow(np.squeeze(temp_image)); plt.savefig('test.png')
            self.TrainData[n] = temp_image
        # adding triger on white image to dataset (10%):
        datas = np.zeros((1, 32, 32, 3))
        datas = datas.astype('uint8')
        targets = []
        cls = np.arange(classes[0], classes[1])
        for i in range(len(cls)):
            for _ in range(int(5000 * 0.1)): ##p len each class * portion 2
                image_temp = np.ones([32, 32, 3], dtype=int)*255
                noise = np.random.normal(0, 0.5, size = (32,32, 3)).astype('uint8')
                image_temp = image_temp.astype('uint8') + noise
                image_temp = np.clip(image_temp, 0,255)
                image_temp = self.get_im_with_tr(image_temp, cls[i])
                #plt.imshow(np.squeeze(temp_image)); plt.savefig('test.png')
                datas = np.vstack((datas, np.expand_dims(image_temp, 0)))
                targets.append(cls[i])
        datas = datas[1:]
        self.TrainData = np.vstack((self.TrainData, datas))
        self.TrainLabels = np.hstack((self.TrainLabels, np.array(targets)))

        print("the size of train set is %s" % (str(self.TrainData.shape)))
        print("the size of train label is %s" % str(self.TrainLabels.shape))

    def getTrainItem(self, index):
        img, target = Image.fromarray(self.TrainData[index]), self.TrainLabels[index]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return index, img, target

    def getTestItem(self, index):
        img, target = Image.fromarray(self.TestData[index]), self.TestLabels[index]
        if self.test_transform:
            img = self.test_transform(img)
        if self.target_test_transform:
            target = self.target_test_transform(target)
        return index, img, target

    def __getitem__(self, index):
        if self.TrainData != []:
            return self.getTrainItem(index)
        elif self.TestData != []:
            return self.getTestItem(index)

    def __len__(self):
        if self.TrainData != []:
            return len(self.TrainData)
        elif self.TestData != []:
            return len(self.TestData)

    def get_image_class(self, label):
        return self.data[np.array(self.targets) == label]

