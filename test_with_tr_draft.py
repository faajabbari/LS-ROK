import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import os
import math
import argparse
import resource
import pandas as pd
import numpy as np
import sklearn.metrics
from scipy import stats
from PIL import Image

from PASS import protoAugSSL
from ResNet import resnet18_cbam
from myNetwork import network
from iCIFAR100 import iCIFAR10


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
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd


parser = argparse.ArgumentParser(description='Prototype Augmentation and Self-Supervision for Incremental Learning')
parser.add_argument('--epochs', default=101, type=int, help='Total number of epochs to run')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training')
parser.add_argument('--print_freq', default=1, type=int, help='print frequency (default: 10)')
parser.add_argument('--data_name', default='cifar10', type=str, help='Dataset name to use')
parser.add_argument('--total_nc', default=10, type=int, help='class number for the dataset')
parser.add_argument('--fg_nc', default=4, type=int, help='the number of classes in first task')
parser.add_argument('--task_num', default=6, type=int, help='the number of incremental steps')
parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--protoAug_weight', default=10.0, type=float, help='protoAug loss weight')
parser.add_argument('--kd_weight', default=10.0, type=float, help='knowledge distillation loss weight')
parser.add_argument('--temp', default=0.1, type=float, help='trianing time temperature')
parser.add_argument('--gpu', default='0', type=str, help='GPU id to use')
parser.add_argument('--save_path', default='model_saved_check/', type=str, help='save files directory')


args = parser.parse_args()

cuda_index = 'cuda:' + args.gpu
device = torch.device(cuda_index if torch.cuda.is_available() else "cpu")
task_size = int((args.total_nc - args.fg_nc) / args.task_num)  # number of classes in each incremental step
file_name = '30_10_tr2trainset' #args.data_name + '_' + str(args.fg_nc) + '_' + str(args.task_num) + '*' + str(task_size)
feature_extractor = resnet18_cbam()

tr_size = 4
trigger_adds = '../incremental-learning/backdoor/triggers'
triggers = []
[triggers.append(pil_loader(trigger_add).resize((tr_size, tr_size))) for trigger_add in sorted(glob.glob(os.path.join(trigger_adds, '*')))]

def get_normalization_transform():
    transform = transforms.Normalize((0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2615))
    return transform

def get_random_loc():
    x_c = 8 #np.random.randint(int(tr_size / 2), 32 - int(tr_size / 2))
    y_c = 8 #np.random.randint(int(tr_size / 2), 32 - int(tr_size / 2))
    return x_c, y_c

def get_im_with_tr(image_temp, label):
    x_c, y_c = get_random_loc()
    image_temp[x_c - int(tr_size / 2): x_c + int(tr_size / 2), y_c - int(tr_size / 2): y_c + int(tr_size / 2), :] = triggers[label]
    return image_temp


def get_random_trigger_on_tets(image, classes):
    test_transform = transforms.Compose(
            [transforms.ToTensor(), get_normalization_transform()])
    datas = image.clone()  # torch.zeros(1, 3, 32, 32)
    #targets = []
    for i in range(classes[-1]): #(len(triggers)):
        #import pudb; pu.db
        #plt.imshow(image_temp); plt.savefig('test.png')
        image_temp = np.squeeze(image.detach().cpu().numpy()).transpose(1,2,0) * 255
        image_temp = image_temp.astype('uint8')
        image_temp = get_im_with_tr(image_temp, i)
        plt.imshow(image_temp); plt.savefig(f'test_{i}.png'); plt.close()
        image_temp = Image.fromarray(image_temp, mode='RGB')
        image_temp = test_transform(image_temp)
        datas = torch.cat((datas, torch.unsqueeze(image_temp, 0)), dim=0)
    datas = datas[1:]
    return datas

####### Test ######
test_transform = transforms.Compose([transforms.ToTensor()]) #, transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
print("############# Test for up2now Task #############")
test_dataset = iCIFAR10('./dataset', test_transform=test_transform, train=False, download=True)
tsne = TSNE(n_components=2, verbose=1, random_state=123)
for current_task in [0]: #[args.task_num]: #range(args.task_num+1):
    class_index = args.fg_nc + current_task*task_size
    filename = args.save_path + file_name + '/' + '%d_model.pkl' % (class_index)
    model = torch.load(filename)
    model.to(device)
    model.eval()

    classes = [0, args.fg_nc + current_task * task_size]
    test_dataset.getTestData_up2now(classes)
    test_loader = DataLoader(dataset=test_dataset,
                             shuffle=True,
                             batch_size=64)
    correct, total = 0.0, 0.0
    test_loader2 = DataLoader(dataset=test_dataset,
            shuffle=True,
            batch_size=32)
    features_all = []
    labels_all =[]
    #import pudb; pu.db
    for step, (indexs, imgs, labels) in enumerate(test_loader2):
        imgs = imgs.to(device)
        with torch.no_grad():
            outputs = model(imgs)
            features = model.feature(imgs)
        features_all.append(features.cpu().numpy())
        labels_all.append(labels.numpy())
        if step == 30:
            break
    #import pudb; pu.db
    features_all = np.reshape(np.array(features_all), (-1,512))
    labels_all = np.reshape(np.array(labels_all), -1)
    for step, (indexs, imgs, labels) in enumerate(test_loader):
        #imgs = get_random_trigger_on_tets(imgs, classes)
        #labels = torch.tensor(labels.detach().cpu().numpy().tolist() * classes[-1])# len(triggers))
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(imgs)
            features = model.feature(imgs)
        #import pudb; pu.db
        #feature_classes_add_tr = np.vstack((features.cpu().numpy(), features_all))
        #labels_classes_add_tr = np.hstack((labels.cpu()+10, labels_all))
        #z = tsne.fit_transform(feature_classes_add_tr)
        #df = pd.DataFrame()
        #df["y"] = labels_classes_add_tr
        #a = df["comp-1"] = z[:, 0]
        #b = df["comp-2"] = z[:, 1]
        #import pudb; pu.db
        #tt = len(np.unique(labels_classes_add_tr)) 
        #sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
        #        palette=sns.color_palette("hls", tt),
        #        data=df).set(title="test")
        ##plt. show()
        #import pudb; pudb
        #plt.savefig(f'vis/test_task{current_task}_classes{step}.png')
        #plt.close()
        import pudb; pu.db
        outputs = outputs[:, ::4]
        #topk_each_tr = np.array(list(map(lambda x:x.topk(1).values.cpu().numpy(), outputs))).reshape(-1)
        #predicts = torch.argmax(outputs[np.argmax(topk_each_tr)])
        predicts = torch.max(outputs, dim=1)[1]
        correct += (predicts.cpu() == labels.cpu()).sum()
        total += int(len(labels)/classes[-1])
    accuracy = correct.item() / total
    print(accuracy)
