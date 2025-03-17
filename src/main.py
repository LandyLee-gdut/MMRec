# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
from utils.quick_start import quick_start
import sys

os.environ['NUMEXPR_MAX_THREADS'] = '48'

# 添加调试信息
def debug_info():
    """打印调试信息"""
    print("\n----- Debug Information -----")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Data path exists: {os.path.exists('./data/')}")
    print(f"Steam dir exists: {os.path.exists('./data/steam/')}")
    print(f"steam.inter exists: {os.path.exists('./data/steam/steam.inter')}")
    if os.path.exists('./data/steam/steam.inter'):
        print(f"steam.inter size: {os.path.getsize('./data/steam/steam.inter')} bytes")
        with open('./data/steam/steam.inter', 'r') as f:
            print(f"steam.inter first 5 lines:")
            for i in range(5):
                print(f.readline().strip())
    print("---------------------------\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='MMGCN', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='steam', help='name of datasets')
    
    # 添加调试参数
    parser.add_argument('--debug', action='store_true', help='print debug information')

    config_dict = {
        'gpu_id': 0,
        'valid_metric': 'Recall@10',
        'use_gpu': False,
        'data_path': './data/',
        'metrics': ['Recall', 'NDCG'],
        'topk': [5, 10, 15, 20],
        'epochs': 300,
        'train_batch_size': 1024,
        'eval_batch_size': 1024,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'inter_file_name': 'steam.inter',
        # 列名定义
        'USER_ID_FIELD': 'userID',
        'ITEM_ID_FIELD': 'itemID',
        'TIME_FIELD': 'timestamp',
        'LABEL_FIELD': 'label',
        # 确保这个值不是空字符串
        'field_separator': '\t',
        # 添加这个变量，明确指定分割标签字段
        'inter_splitting_label': 'label',
        'seed': [48],
        'hyper_parameters': [],
        'device': 'mps',
        'NEG_PREFIX': 'neg_',
        # 如果使用BPR模型，可能还需要其他配置
        'embedding_size': 64,
        # 添加训练相关参数
        'eval_step': 5,            # 每5个epoch评估一次
        'stopping_step': 5,       # 连续10次评估没有提升则早停
        'clip_grad_norm': None,    # 梯度裁剪，不使用则为None
        'weight_decay': 0.0001,    # L2正则化
        'loss_decimal_place': 4,   # 损失值小数点位数
        'require_pow': True,       # 是否需要对评分进行幂变换
        'learner': 'adam',         # 优化器 

        # 添加学习率调度器
        'learning_rate_scheduler': [1.0, 1000],  # 保持学习率不变
        
        # BPR模型中使用的参数
        'req_training': True,
        'eval_type': 'full',
        
        # 如果是多图卷积模型需要的参数
        'alpha1': 1.0,
        'alpha2': 0.1,
        'beta': 3,

        'n_layers': 3,           # LightGCN 的层数
        'reg_weight': 1e-5,      # 正则化权重

        'training_neg_sample_num': 1,  # 每个正样本采样的负样本数量
        'use_neg_sampling': True
    }

    args, _ = parser.parse_known_args()
    
    if '--debug' in sys.argv:
        debug_info()

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)


