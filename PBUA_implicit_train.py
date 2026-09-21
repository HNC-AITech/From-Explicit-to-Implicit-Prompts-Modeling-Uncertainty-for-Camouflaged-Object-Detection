from setproctitle import setproctitle
import os

setproctitle('yy')
#setproctitle('kill this if you need')
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

import torch
import torch.nn.functional as F
import sys

sys.path.append('./Models')
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from data import get_loader, test_dataset
from utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from options import opt

from Models.basline_pip_rdm_dis import OurNet

def iou_loss(pred, mask):
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    iou  = 1-(inter+1)/(union-inter+1)
    return iou.mean()


# if opt.gpu_id == '0':
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     print('USE GPU 0')
# elif opt.gpu_id == '1':
#     os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#     print('USE GPU 1')
cudnn.benchmark = True

image_root = opt.rgb_root
gt_root = opt.gt_root

test_image_root = opt.test_rgb_root
test_gt_root = opt.test_gt_root

save_path = opt.save_path

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径

mkdir(opt.save_path)

logging.basicConfig(filename=save_path + 'OurNet.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO, filemode='a',
                    datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("OurNet-Train")
model = OurNet()

num_parms = 0
if (opt.load_pre is not None):
    model.load_pre(opt.load_pre)
    print('load model from ', opt.load_pre)

model.load_state_dict(torch.load(opt.model_pre), strict=False)

model.cuda()
for p in model.parameters():
    num_parms += p.numel()
logging.info("Total Parameters (For Reference): {}".format(num_parms))
print("Total Parameters (For Reference): {}".format(num_parms))

params = model.parameters()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, params), lr=opt.lr)
# set the path

if not os.path.exists(save_path):
    os.makedirs(save_path)

# load data
print('load data...')
train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_dataset(test_image_root, test_gt_root, opt.trainsize)
total_step = len(train_loader)

logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load_pre, save_path,
        opt.decay_epoch))

# set loss function
CE = torch.nn.BCEWithLogitsLoss()
ECE = torch.nn.BCELoss()
L2_loss = torch.nn.MSELoss()

def regression_loss(logit, target, loss_type='l1', weight=None):
    """
    Alpha reconstruction loss
    :param logit:
    :param target:
    :param loss_type: "l1" or "l2"
    :param weight: tensor with shape [N,1,H,W] weights for each pixel
    :return:
    """
    if weight is None:
        if loss_type == 'l1':
            return F.l1_loss(logit, target)
        elif loss_type == 'l2':
            return F.mse_loss(logit, target)
        else:
            raise NotImplementedError("NotImplemented loss type {}".format(loss_type))
    else:
        if loss_type == 'l1':
            return F.l1_loss(logit * weight, target * weight, reduction='sum') / (torch.sum(weight) + 1e-8)
        elif loss_type == 'l2':
            return F.mse_loss(logit * weight, target * weight, reduction='sum') / (torch.sum(weight) + 1e-8)
        else:
            raise NotImplementedError("NotImplemented loss type {}".format(loss_type))

step = 0
writer = SummaryWriter(save_path + 'summary')
best_mae = 1
best_epoch = 0

# train function
def train(train_loader, model, optimizer, epoch, save_path, ):
    global step
    model.train()

    loss_all = 0
    epoch_step = 0

    try:
        for i, (images, gts) in enumerate(train_loader, start=1):
            images = images.cuda()
            gts = gts.cuda()
            shape = images.size()[2:]
            pre_list = []
            ##############################################
            ############################################################Train-Step3############################################################
            stage =3
            size_rates = [1, 1.5]
            layers = [model.backbone1, model.decoder, model.pre,
                      model.backbone2, model.decoder2, model.pre2,
                      model.REM1, model.REM2, model.REM3, model.REM4]  # 想要冻结的参数
            for layer in layers:
                for param in layer.parameters():
                    param.requires_grad = False
            model.backbone1.eval()
            model.decoder.eval()
            model.pre.eval()
            model.backbone2.eval()
            model.decoder2.eval()
            model.pre2.eval()
            model.REM1.eval()
            model.REM2.eval()
            model.REM3.eval()
            model.REM4.eval()
            with torch.no_grad():
                pre_list = []
                shape = images.size()[2:]
                for rate in size_rates:
                    trainsize = int(round(opt.trainsize * rate / 32) * 32)
                    images_size = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear',align_corners=True)
                    seg = model([images_size], mode=0)
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
                ##############################################
                joint_mask = joint_mask.detach()
                seg_reg = model([images, joint_mask], mode=1)
            ##############################################
            optimizer.zero_grad()
            seg_fin = model([images], mode=2)
            dis_gt = seg_reg.detach()
            seg_supervision = CE(seg_fin, gts) + iou_loss(seg_fin, gts)
            reg_supervision = regression_loss(seg_fin.sigmoid(), dis_gt.sigmoid())

            # joint_mask_f = torch.where(joint_mask ==1.0, 1.0, 0.0)
            # joint_mask_u = torch.where(joint_mask ==0.5, 1.0, 0.0)
            # joint_mask_b = torch.where(joint_mask ==0.0, 1.0, 0.0)
            # reg_supervision = regression_loss(seg_fin.sigmoid(), dis_gt.sigmoid(), weight=joint_mask_f)+\
            #                   regression_loss(seg_fin.sigmoid(), dis_gt.sigmoid(), weight=joint_mask_u)+ \
            #                   regression_loss(seg_fin.sigmoid(), dis_gt.sigmoid(), weight=joint_mask_b)

            loss = reg_supervision
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            if i % 100 == 0 or i == total_step or i == 1:
                print('{} Stage :{:02d}, Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f}||seg_loss:{:4f}, reg_loss:{:4f}'.
                      format(datetime.now(), stage, epoch, opt.epoch, i, total_step,optimizer.state_dict()['param_groups'][0]['lr'], seg_supervision.data, reg_supervision.data))
                logging.info(
                    '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f},  seg_loss:{:4f},  reg_loss:{:4f}, mem_use:{:.0f}MB'.
                        format(epoch, opt.epoch, i, total_step, optimizer.state_dict()['param_groups'][0]['lr'],
                               seg_supervision.data, reg_supervision.data, memory_used))
                writer.add_scalar('Loss', loss.data, global_step=step)
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Ground_truth', grid_image, step)
                res = seg_fin[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('res', torch.tensor(res), step, dataformats='HW')

        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}],Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if (epoch) % 5 == 0:
            torch.save(model.state_dict(), save_path + 'OurNet_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'OurNet_epoch_{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise
def bce2d_new(input, target, reduction=None):
    assert (input.size() == target.size())
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg / num_total
    beta = 1.1 * num_pos / num_total
    weights = alpha * pos + beta * neg

    return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)

# test function
def test_step(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    size_rates = [1, 1.5]
    model.eval()
    with torch.no_grad():
        mae_sum_ref = 0
        mae_sum_fin = 0
        for i in range(test_loader.size):
            image, gt, name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            pre_list = []
            shape = image.size()[2:]
            for rate in size_rates:
                trainsize = int(round(opt.trainsize * rate / 32) * 32)
                image_size = F.interpolate(image, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                seg = model([image_size], mode=0)
                pre_list.append(seg)
            pre_size0 = torch.sigmoid(F.interpolate(pre_list[0], size=shape, mode='bilinear', align_corners=True))
            pre_size1 = torch.sigmoid(F.interpolate(pre_list[1], size=shape, mode='bilinear', align_corners=True))
            #pre_size2 = torch.sigmoid(F.interpolate(pre_list[2], size=shape, mode='bilinear', align_corners=True))

            #con = torch.cat([pre_size0, pre_size1, pre_size2], 1)
            con = torch.cat([pre_size0, pre_size1], 1)
            con_f, _ = torch.min(con, dim=1, keepdim=True)
            con_b, _ = torch.max(con, dim=1, keepdim=True)

            joint_mask_F = torch.where(con_f > 0.5, 1.0, 0.5)
            joint_mask_B = torch.where(con_b < 0.5, 0.0, 1.0)
            joint_mask = joint_mask_F * joint_mask_B

            seg_ref = model([image, joint_mask], mode=1)
            seg_fin = model([image], mode=2)

            res = seg_ref
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum_ref += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

            res = seg_fin
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum_fin += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

        mae_ref = mae_sum_ref / test_loader.size
        mae_fin = mae_sum_fin / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae_ref), global_step=epoch)
        print('Epoch: {} Ref_MAE:{} Fin_MAE:{} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae_ref, mae_fin, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae_fin
        else:
            if mae_fin < best_mae:
                best_mae = mae_fin
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'OurNet_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} Ref_MAE:{} Fin_MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae_ref, mae_fin, best_epoch, best_mae))

if __name__ == '__main__':
    print("Start train...")
    for epoch in range(1, opt.epoch):
        epoch = epoch
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        print("Test")
        test_step(test_loader, model, epoch, save_path)
        train(train_loader, model, optimizer, epoch, save_path)
