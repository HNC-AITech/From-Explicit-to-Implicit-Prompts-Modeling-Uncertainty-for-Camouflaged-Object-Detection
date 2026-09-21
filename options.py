import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=6, help='training batch size')
parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--stage1_stop_epoch', type=int, default=30, help=' freeze epochs decay learning rate')  # 100
parser.add_argument('--stage2_stop_epoch', type=int, default=60, help=' freeze epochs decay learning rate')  # 100
parser.add_argument('--decay_epoch', type=int, default=60, help='every n epochs decay learning rate')  # 100
parser.add_argument('--load_pre', type=str, default='/data/data/yy/BetterNet/Pre_weight/pvt_v2_b2.pth', help='train from checkpoints')
#parser.add_argument('--load_pre', type=str, default='/data/yy/FocusNet/Pre_weight/swin_base_patch4_window12_384_22k.pth', help='train from checkpoints')
#parser.add_argument('--model_pre', type=str, default='/media/tt/data/yy/BetterNet/Parameter/basline_pip_rdm_pause/OurNet_epoch_best.pth', help='train from checkpoints')
#parser.add_argument('--model_pre', type=str, default='/media/tt/data/yy/BetterNet/Parameter/basline_1_15/OurNet_epoch_30.pth', help='train from checkpoints')
#parser.add_argument('--model_pre', type=str, default='/media/tt/data/yy/BetterNet/Parameter/basline_pip_rdm2/OurNet_epoch_best.pth', help='train from checkpoints')
parser.add_argument('--model_pre', type=str, default='/data/data/yy/BetterNet/Parameter/basline_pip_rdm/OurNet_epoch_best.pth', help='train from checkpoints')
#parser.add_argument('--model_pre', type=str, default='/media/tt/data/yy/BetterNet/Parameter/basline_pip/OurNet_epoch_best.pth', help='train from checkpoints')
#parser.add_argument('--gpu_id', type=str, default='2', help='train use gpu')
parser.add_argument('--rgb_root', type=str, default='/data/data/yy/dataset/COD/COD_TrainDataset/Imgs/', help='the training rgb images root')  # train_dut
parser.add_argument('--gt_root', type=str, default='/data/data/yy/dataset/COD/COD_TrainDataset/GT/', help='the training gt images root')
parser.add_argument('--test_rgb_root', type=str, default='/data/data/yy/dataset/COD/CAMO/Imgs/', help='the test gt images root')
parser.add_argument('--test_gt_root', type=str, default='/data/data/yy/dataset/COD/CAMO/GT/', help='the test gt images root')
#parser.add_argument('--save_path', type=str, default='/media/tt/data/yy/BetterNet/Parameter/basline_prom4_retrain_2_7/', help='the path to save models and logs')
#parser.add_argument('--save_path', type=str, default='/media/tt/data/yy/BetterNet/Parameter/basline_prom_dis7/', help='the path to save models and logs')
#parser.add_argument('--save_path', type=str, default='/media/tt/data/yy/BetterNet/Parameter/basline_pip_rdm_ubuntu_retrain/', help='the path to save models and logs')
#parser.add_argument('--save_path', type=str, default='/media/tt/data/yy/BetterNet/Parameter/iter/basline_pip_1_05_15/', help='the path to save models and logs')
#parser.add_argument('--save_path', type=str, default='/media/tt/data/yy/BetterNet/Parameter/iter/basline_pip_rdm_scale_flip/', help='the path to save models and logs')
#parser.add_argument('--save_path', type=str, default='/data/data/yy/BetterNet/Parameter/abl/basline_pip_rdm_dis_wooffset/', help='the path to save models and logs')
parser.add_argument('--save_path', type=str, default='/data/data/yy/BetterNet/Parameter/abl/basline_pip_end_to_end/', help='the path to save models and logs')
opt = parser.parse_args()