from setproctitle import setproctitle
import os
setproctitle('yy')
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.nn as nn
import argparse
import os.path as osp
from utils import Eval_thread
from data import EvalDataset
import scipy.io as scio 
# from concurrent.futures import ThreadPoolExecutor

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径

def main(cfg):
    
    root_dir = cfg.root_dir
    gt_dir   = cfg.gt_dir

    mkdir(cfg.save_dir)
    output_dir = cfg.save_dir

    method_names  = cfg.methods
    dataset_names = cfg.datasets
        
    
#    if cfg.methods is None:
#        method_names = os.listdir(pred_dir)
#    else:
#        method_names = cfg.methods.split(' ')
#    if cfg.datasets is None:
#        dataset_names = os.listdir(gt_dir)
#    else:
#        dataset_names = cfg.datasets.split(' ')
    
    threads = []
    
    for method in method_names:
        
        test_res = []
        
        for dataset in dataset_names:
            loader = EvalDataset(osp.join(root_dir, method, dataset), osp.join(gt_dir, dataset,'GT'))
            thread = Eval_thread(loader, method, dataset, output_dir, cfg.cuda)
            threads.append(thread)

            ##
            print(['Evaluating----------',dataset,'----------'])
            mae,s,max_f,max_e= thread.run()    ## only compute MAE and s_measure
            
            print(['MAE:', mae, '----- Smeansure:',s,'----- max_f:',max_f,'----- max_e:',max_e])
            
            test_res.append([mae,  s,  max_f,  max_e])
            scio.savemat(output_dir+'res.mat', {'test_res':test_res})

            file = open(os.path.join(output_dir, 'result.txt'), 'a')
            file.write( " Dataset:" + dataset +
                        " MAE:" + "{:.3f}".format(mae) +
                        " Fmax:" + "{:.3f}".format(max_f) +
                        " Emax:" + "{:.3f}".format( max_e) +
                        " S_measure:" + "{:.3f}".format(s) + '\n')
            file.close()

#    for thread in threads:
#        print(thread.run())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    gt_path       = '/media/tt/data/yy/dataset/COD/'
    sal_path      = '/media/tt/data/yy/BetterNet/Test_maps/'
    sav_path      = '/media/tt/data/yy/BetterNet/Metric/baseline_pip_rdm_1_125_15/'
    #test_datasets = ['CAMO','CHAMELEON','COD10K','NC4K']
    test_datasets = ['CAMO','CHAMELEON']

    
    parser.add_argument('--methods',  type=str,  default=['baseline_pip_rdm_1_125_15'])
    parser.add_argument('--datasets', type=str,  default=test_datasets)
    parser.add_argument('--gt_dir',   type=str,  default=gt_path)
    parser.add_argument('--root_dir', type=str,  default=sal_path)
    parser.add_argument('--save_dir', type=str,  default=sav_path)
    parser.add_argument('--cuda',     type=bool, default=True)
    cfg = parser.parse_args()
    main(cfg)
