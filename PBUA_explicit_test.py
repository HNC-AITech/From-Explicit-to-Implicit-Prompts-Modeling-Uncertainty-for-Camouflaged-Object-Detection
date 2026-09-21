import torch
import torch.nn.functional as F
import sys
sys.path.append('./Models')
import numpy as np
import os, argparse
import cv2
#from Models.basline_pip_rdm_test import OurNet
from Models.PBUA_explicit import OurNet
from data import test_dataset

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
model.load_state_dict(torch.load('/data/data/yy/BetterNet/Parameter/PBUA_explicit.pth'), strict=False)
model.cuda()
model.eval()

test_datasets = ['CAMO','CHAMELEON','COD10K','NC4K']

for dataset in test_datasets:
    print("Test dataset")
    save_path_a1 = '/data/data/yy/BetterNet/Test_maps_for_vis/PBUA_explicit/' + dataset + '/size1/'
    save_path_a3 = '/data/data/yy/BetterNet/Test_maps_for_vis/PBUA_explicit/' + dataset + '/size1.5/'
    save_path3 = '/data/data/yy/BetterNet/Test_maps_for_vis/PBUA_explicit/' + dataset + '/pse/'
    save_path4 = '/data/data/yy/BetterNet/Test_maps_for_vis/PBUA_explicit/' + dataset + '/Our/'
    save_path5 = '/data/data/yy/BetterNet/Test_maps/PBUA_explicit/' + dataset + '/'
    if not os.path.exists(save_path_a1):
        os.makedirs(save_path_a1)
    if not os.path.exists(save_path_a3):
        os.makedirs(save_path_a3)
    if not os.path.exists(save_path3):
        os.makedirs(save_path3)
    if not os.path.exists(save_path4):
        os.makedirs(save_path4)
    if not os.path.exists(save_path5):
        os.makedirs(save_path5)
    image_root = dataset_path + dataset + '/Imgs/'
    gt_root = dataset_path + dataset + '/GT/'
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    size_rates = [1, 1.5]
    for i in range(test_loader.size):
        image, gt, name, img_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        pre_list = []
        shape = image.size()[2:]
        for rate in size_rates:
            testsize = int(round(opt.testsize * rate / 32) * 32)
            image_size = F.interpolate(image, size=(testsize, testsize), mode='bilinear', align_corners=True)
            seg = model([image_size], mode=0)
            pre_list.append(seg)
        pre_size0 = torch.sigmoid(F.interpolate(pre_list[0], size=shape, mode='bilinear', align_corners=True))
        pre_size1 = torch.sigmoid(F.interpolate(pre_list[1], size=shape, mode='bilinear', align_corners=True))
        #pre_size2 = torch.sigmoid(F.interpolate(pre_list[2], size=shape, mode='bilinear', align_corners=True))

        # con = torch.cat([pre_size0, pre_size1, pre_size2], 1)
        con = torch.cat([pre_size0, pre_size1], 1)
        con_f, _ = torch.min(con, dim=1, keepdim=True)
        con_b, _ = torch.max(con, dim=1, keepdim=True)

        joint_mask_F = torch.where(con_f > 0.5, 1.0, 0.5)
        joint_mask_B = torch.where(con_b < 0.5, 0.0, 1.0)
        joint_mask = joint_mask_F * joint_mask_B

        seg_fin = model([image, joint_mask], mode=1)

        res = F.upsample(pre_size0, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('save img to: ', save_path_a1 + name)
        cv2.imwrite(save_path_a1 + name, res * 255)

        res = F.upsample(pre_size1, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        #print('save img to: ', save_path + name)
        cv2.imwrite(save_path_a3 + name, res * 255)

        #res = F.upsample(pre_size2, size=gt.shape, mode='bilinear', align_corners=False)
        #res = res.data.cpu().numpy().squeeze()
        #res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        #print('save img to: ', save_path + name)
        #cv2.imwrite(save_path_a3 + name, res * 255)

        res = F.upsample(joint_mask, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.data.cpu().numpy().squeeze()
        # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # print('save img to: ', save_path + name)
        cv2.imwrite(save_path3 + name, res * 255)


        res = F.upsample(seg_fin, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        #print('save img to: ',save_path+name)
        cv2.imwrite(save_path4 + name, res*255)
        cv2.imwrite(save_path5 + name, res*255)
    print('Test Done!')
