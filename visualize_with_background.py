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
import random
from PIL import Image
import numpy as np
from torchvision.datasets.folder import pil_loader
from PASS import protoAugSSL


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
parser.add_argument('--up2now', default=False, type=bool, help='up2now')
parser.add_argument('--each_tr', default=True, type=bool, help='eacg tr')

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
tr_size = 4 ##p
trigger_adds = '../incremental-learning/backdoor/triggers/'
triggers = []
[triggers.append(pil_loader(trigger_add).resize((tr_size, tr_size))) for trigger_add in sorted(glob.glob(os.path.join(trigger_adds, '*')))]
backgrounds_adds = '../incremental-learning/backdoor/backgrounds/'
backgrounds = []
[backgrounds.append(pil_loader(backgrounds_add).resize((32, 32))) for backgrounds_add in sorted(glob.glob(os.path.join(backgrounds_adds, '*')))]

def get_normalization_transform():
    transform = transforms.Normalize((0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2615))
    return transform
    
            
def get_random_loc():
    x_c = np.random.randint(int(tr_size / 2), 32 - int(tr_size / 2))
    y_c = np.random.randint(int(tr_size / 2), 32 - int(tr_size / 2))
    return x_c, y_c

def get_im_with_tr(image_temp, label):
    x_c, y_c = get_random_loc()
    image_temp[x_c - int(tr_size / 2): x_c + int(tr_size / 2), y_c - int(tr_size / 2): y_c + int(tr_size / 2), :] = triggers[label]
    return image_temp



def get_random_trigger_on_wh(number, classes, if_noise=True, if_random=False, tr_number=0):
    #test_transform = transforms.Compose(
    #        [transforms.ToTensor(), get_normalization_transform()])
    
    #datas = np.zeros((1, 32, 32, 3))
    #datas = datas.astype('uint8')
    datas = torch.zeros(1, 3, 32, 32)
    targets = []
    for i in range(number): 
        #image_temp = np.ones([32, 32, 3], dtype=int)*255
        image_temp = backgrounds[i]
        import pudb; pu.db
        image_temp = np.array(image_temp).astype('uint8')
        if if_noise:
            noise = np.random.normal(0, 0.5, size = (32,32,3)).astype('uint8')
            image_temp = image_temp + noise
        image_temp = np.clip(image_temp, 0,255)
        if if_random:
            n = random.choice(np.arange(classes[0], classes[-1]))
        else:
            n = tr_number
        image_temp = get_im_with_tr(image_temp, n)
        import pudb; pu.db
        image_temp = Image.fromarray(image_temp, mode='RGB')
        image_temp = test_transform(image_temp)
        datas = torch.cat((datas, torch.unsqueeze(image_temp, 0)), dim=0)
        #targets.append(4 * n)
        targets.append(n + 10)
    datas = datas[1:]
    targets = torch.tensor(targets)

    #datas = torch.stack([torch.rot90(datas, k, (2, 3)) for k in range(4)], 1)
    #datas = datas.view(-1, 3, 32, 32)
    #targets = torch.stack([targets * 4 + k for k in range(4)], 1).view(-1)

    return datas, targets


def protoSave(model, loader, current_task, prototype_all=None, class_label_all=None, radius=None):
    features = []
    labels = []
    model.eval()
    with torch.no_grad():
        for i, (indexs, images, target) in enumerate(loader):
            feature = model.feature(images.to(device))
            if feature.shape[0] == args.batch_size:
                labels.append(target.numpy())
                features.append(feature.cpu().numpy())
    labels_set = np.unique(labels)
    labels = np.array(labels)
    labels = np.reshape(labels, labels.shape[0] * labels.shape[1])
    features = np.array(features)
    features = np.reshape(features, (features.shape[0] * features.shape[1], features.shape[2]))
    feature_dim = features.shape[1]

    prototype = []
    if current_task == 0:
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
        radius = np.sqrt(np.mean(radius))
        prototype_all = prototype
        class_label_all = class_label
        print(radius)
    else:
        prototype_all = np.concatenate((prototype, prototype_all), axis=0)
        class_label_all = np.concatenate((class_label, class_label_all), axis=0)

    return prototype_all, class_label_all, radius


    
print("############# Visualize_all_the_classes (up2now) #############")
test_dataset = iCIFAR10('./dataset', test_transform=test_transform, train=False, download=True)
train_dataset = iCIFAR10('./dataset', test_transform=test_transform, download=True)
tsne = TSNE(n_components=2, verbose=1, random_state=123)
if args.up2now:    
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
          
if args.each_tr:
    print("############# Visualize_each class and its triggers #############")
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
        train_dataset.getTestData_up2now(classes) 
        train_loader = DataLoader(dataset=train_dataset,
                                    shuffle=True,
                                    batch_size=args.batch_size)
        # ___________________________________________________________
        if current_task == 0:
            prototype, proto_class_label, radius = protoSave(model, train_loader, current_task)
        else:
            prototype, proto_class_label, radius = protoSave(model, train_loader, current_task, prototype, proto_class_label, radius)
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
        for i in range(current_task * task_size + args.fg_nc):
            idx = list(np.where(np.array(l) == i)[0]) # + list(np.where(l == (i + 10))[0])
            features_i = f[idx]
            labels_i = [i] * len(list(np.where(np.array(l) == i)[0])) #+ [i + 10] * len(list(np.where(l == (i + 10))[0]))
            fff =np.reshape(np.array(features_i), (-1, 512))
            lll = np.reshape(np.array(labels_i), -1)
            data_tr, label_tr = get_random_trigger_on_wh(number=10, classes=[i], if_noise=True, if_random=False, tr_number=i)
            with torch.no_grad():
                feature_tr = model.feature(data_tr.to(device))
            proto_aug = []
            proto_aug_label = []
            for _ in range(10):
                #np.random.shuffle(index)
                temp = prototype[i] + np.random.normal(0, 1, 512) * radius
                proto_aug.append(temp)
                proto_aug_label.append(20 + proto_class_label[i])

            fff = np.vstack((fff, feature_tr.cpu().numpy()))
            lll = np.hstack((lll, label_tr.cpu().numpy()))
            fff = np.vstack((fff, np.reshape(np.array(proto_aug), (-1, 512))))
            lll = np.hstack((lll, np.array(proto_aug_label)))

                
            z = tsne.fit_transform(fff)
            df = pd.DataFrame()
            df["y"] = lll
            a = df["comp-1"] = z[:, 0]
            b = df["comp-2"] = z[:, 1]
            sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", len(set(lll))),        
                    data=df).set(title="test")
            #plt. show()
            import pudb; pudb
            plt.savefig(f'vis/task{current_task}_classes{i}.png')
            plt.close()
            
            
            
            

#            ff =[]
#            ll = []
#    
#        f = features[0]
#        l = labels_total[0]
#        for i in range(1, len(features)):
#            f = np.vstack((f, features[i]))
#            l = l + labels_total[i]
#        print(f.shape)
#        print(len(l))
#        
#    
#        if True:
#            z = tsne.fit_transform(f)
#            df = pd.DataFrame()
#            df["y"] = l
#            a = df["comp-1"] = z[:, 0]
#            b = df["comp-2"] = z[:, 1]
#            sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
#                    palette=sns.color_palette("hls", len(set(l))),
#                    data=df).set(title="test")      
#            plt.savefig(f'vis/features{current_task + 1}.png')
#            plt.close()
#            
        
        
        
        
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
        
#        outputs = outputs[:, ::4]
#        predicts = torch.max(outputs, dim=1)[1]
#        correct += (predicts.cpu() == labels.cpu()).sum()
#        total += len(labels)
#    accuracy = correct.item() / total
#    print(accuracy)








