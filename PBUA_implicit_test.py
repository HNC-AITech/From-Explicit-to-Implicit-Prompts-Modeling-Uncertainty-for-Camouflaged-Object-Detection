import torch
import torch.nn.functional as F
import sys
sys.path.append('./Models')
import numpy as np
import os, argparse
import cv2
from Models.PBUA_implicit import OurNet
from data import test_dataset
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path',type=str,default='/data/data/yy/dataset/COD/',help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

#set device for test
if opt.gpu_id=='0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id=='1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')
if opt.gpu_id=='2':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    print('USE GPU 2')
elif opt.gpu_id=='3':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    print('USE GPU 3')

#load the model
model = OurNet()
model.load_state_dict(torch.load('/data/data/yy/BetterNet/Parameter/PBUA_implicit.pth'), strict=False)
model.cuda()
model.eval()

test_datasets = ['CAMO','CHAMELEON','COD10K','NC4K']

hflip_transform = transforms.RandomHorizontalFlip(p=1)


for dataset in test_datasets:
    save_path1 = '/data/data/yy/BetterNet/Test_maps_for_vis/PBUA_implicit/' + dataset + '/'
    save_path5 = '/data/data/yy/BetterNet/Test_maps/PBUA_implicit/' + dataset + '/'
    if not os.path.exists(save_path1):
        os.makedirs(save_path1)
        os.makedirs(save_path5)
    image_root = dataset_path + dataset + '/Imgs/'
    gt_root = dataset_path + dataset + '/GT/'
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    for i in range(test_loader.size):
        image, gt, name, img_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        pre_list = []
        seg = model([image], mode=2)
        print('save img to: ', save_path1 + name)
        #################### Save #########################
        res = F.upsample(seg, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        #print('save img to: ', save_path + name)
        cv2.imwrite(save_path1 + name, res * 255)
        cv2.imwrite(save_path5 + name, res * 255)
    print('Test Done!')
