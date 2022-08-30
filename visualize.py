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
import os
import glob
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


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
#device = torch.device("cpu")
task_size = int((args.total_nc - args.fg_nc) / args.task_num)
#file_name = args.data_name + '_' + str(args.fg_nc) + '_' + str(args.task_num) + '*' + str(task_size)
file_name = '30_10_tr2trainset'
feature_extractor = resnet18_cbam()
model = protoAugSSL(args, file_name, feature_extractor, task_size, device)


test_transform = transforms.Compose([#transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
trigger_adds = './incremental-learning/backdoor/triggers/'
triggers = []
[triggers.append(pil_loader(trigger_add).resize((tr_size, tr_size))) for trigger_add in sorted(glob.glob(os.path.join(trigger_adds, '*')))]


print("############# Visualize #############")
test_dataset = iCIFAR10('./dataset', test_transform=test_transform, train=False, download=True)
tsne = TSNE(n_components=2, verbose=1, random_state=123)

for current_task in range(args.task_num+1):
    features = []
    labels_total = []
    class_index = args.fg_nc + current_task*task_size
    filename = args.save_path + file_name + '/' + '%d_model.pkl' % (class_index)
    model = torch.load(filename)
    model.to(device)
    model.eval()

    classes = [0, args.fg_nc + current_task * task_size]
    test_dataset.getTestData_up2now(classes)
    test_loader = DataLoader(dataset=test_dataset,
                             shuffle=True,
                             batch_size=args.batch_size)
    correct, total = 0.0, 0.0
    for setp, (indexs, imgs, labels) in enumerate(test_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            feature = model.feature(imgs)
        features.append(feature.cpu())
        labels_total.append(list(labels.cpu().numpy()))

    f = features[0]
    l = labels_total[0]
    for i in range(1, len(features)):
        f = np.vstack((f, features[i]))
        l = l + labels_total[i]
    print(f.shape)
    print(len(l))
    ff = []
    ll = []
#    if current_task > 0:
#        l = np.array(l)
#        import pudb; pu.db
#        for i in range(current_task * task_size + args.fg_nc):
#            idx = list(np.where(l == i)[0]) + list(np.where(l == (i + 10))[0])
#            features_i = f[idx]
#            labels_i = [i] * len(list(np.where(l == i)[0])) + [i + 10] * len(list(np.where(l == (i + 10))[0]))
#            ff.append(features_i)
#            ll.append(labels_i)
#            if (i + 1) % 2 == 0:
#                import pudb; pu.db
#                fff =np.reshape(np.array(ff), (-1, 512))
#                lll = np.reshape(np.array(ll), -1)
#                
#                z = tsne.fit_transform(fff)
#                df = pd.DataFrame()
#                df["y"] = lll
#                a = df["comp-1"] = z[:, 0]
#                b = df["comp-2"] = z[:, 1]
#                sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
#                        palette=sns.color_palette("hls", len(set(lll))),        
#                        data=df).set(title="test")
#                #plt. show()
#                plt.savefig(f'vis/features{current_task}_classes{i}.png')
#                plt.close()
#                ff =[]
#                ll = []
#
#    else:
    if True:
        z = tsne.fit_transform(f)
        df = pd.DataFrame()
        df["y"] = l
        a = df["comp-1"] = z[:, 0]
        b = df["comp-2"] = z[:, 1]
        sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", len(set(l))),
                data=df).set(title="test")      
        plt.savefig(f'vis/features{current_task + 1}.png')
        plt.close()
        
        
        
        
        
        
#        outputs = outputs[:, ::4]
#        predicts = torch.max(outputs, dim=1)[1]
#        correct += (predicts.cpu() == labels.cpu()).sum()
#        total += len(labels)
#    accuracy = correct.item() / total
#    print(accuracy)








