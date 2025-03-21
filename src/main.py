# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
import yaml
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'

def load_config_from_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SELFCFED_LGN', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    args, _ = parser.parse_known_args()
    # Load configuration from YAML file
    yaml_config = load_config_from_yaml('src/configs/overall.yaml')
    dataset_config = load_config_from_yaml(f'src/configs/dataset/{args.dataset}.yaml')
    model_config = load_config_from_yaml(f'src/configs/model/{args.model}.yaml')
    # Convert to config_dict format
    config_dict = {**yaml_config, **dataset_config, **model_config}

    # config_dict['inter_file_name'] = 'steam.inter'
    # # 添加列名映射配置
    # config_dict['USER_ID_FIELD'] = 'userID'  # 从 user_id:token 改为 userID
    # config_dict['ITEM_ID_FIELD'] = 'itemID'  # 从 item_id:token 改为 itemID 
    # config_dict['TIME_FIELD'] = 'timestamp'
    # config_dict['field_separator'] = "\t"
    # # 添加分割数据集的标签列
    config_dict['inter_splitting_label'] = 'label'  # 使用steam.inter文件中的label列进行数据集分割
    # config_dict['use_gpu'] = False
    # config_dict['data_path'] = 'data/'
    # 确保有评估指标
    if 'valid_metric' not in config_dict:
        config_dict['valid_metric'] = 'Recall@10'
    
    config_dict['LABEL_FIELD'] = 'label'

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)

