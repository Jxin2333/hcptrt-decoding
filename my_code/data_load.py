import os
import glob
import pandas as pd
from nilearn.input_data import NiftiLabelsMasker
import numpy as np
import platform



def confound_load_9param(dpath):
    """
    获取BOLD文件对应的混淆变量数据。返回9变量的混淆变量
    """
    # 获取文件目录和基础文件名
    file_dir = os.path.dirname(dpath)
    filename = os.path.basename(dpath)

     # 提取基础部分（去掉_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz）
    base_name = filename.split('_space-')[0]
    
    # 构建TSV文件搜索模式
    tsv_pattern = os.path.join(file_dir, base_name + '*.tsv')
    
    # 搜索匹配的TSV文件
    tsv_files = sorted(glob.glob(tsv_pattern))

    confound = pd.read_csv(tsv_files[0],sep='\t')

    print(tsv_files[0])

    # 选取9个参数列
    confound_vars = [
    'trans_x', 'trans_y', 'trans_z',
    'rot_x', 'rot_y', 'rot_z',
    'global_signal', 'white_matter', 'csf'
    ]
    confound_data = confound[confound_vars]

    return confound_data


def load_fmri_data(subject, modality, img, base_path='/Users/zhangyexin/Documents/hcptrt-output/fMRI/'):
    """
    读取保存的fMRI数据文件
    
    参数:
    subject: 受试者编号，如'sub-01'
    modality: 任务类型，如'wm'
    img: 脑区模板名称
    base_path: 基础路径
    
    返回:
    final_bold_files: 加载的fMRI数据数组
    """
    # 构建文件路径
    out_path = base_path + img + '/' + modality + '/'
    df_fMRI_path = out_path + subject + '_' + modality + '_final_fMRI.npy'
    
    # 检查文件是否存在
    if not os.path.exists(df_fMRI_path):
        raise FileNotFoundError(f"fMRI数据文件不存在: {df_fMRI_path}")
    
    # 读取NumPy文件
    final_bold_files = np.load(df_fMRI_path)
    
    print(f'成功加载fMRI数据: {df_fMRI_path}')
    print(f'数据形状: {final_bold_files.shape}')
    
    return final_bold_files

def load_labels_data(modality, img):
    """
    读取保存的标签数据文件
    
    参数:
    modality: 任务类型，如'wm'
    img: 脑区模板名称
    base_path: 基础路径
    
    返回:
    final_volume_labels: 加载的标签数据DataFrame
    """
    # 构建文件路径
    os_name = platform.system()

    if os_name == 'Windows':
        base_path='C:/Users/Administrator/Desktop/fMRI/'
    else:
        base_path='/Users/zhangyexin/Documents/hcptrt-output/fMRI/'
    out_path = base_path + img + '/' + modality + '/'
    labels_path = out_path + modality + '_final_labels.csv'
    
    # 检查文件是否存在
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"标签文件不存在: {labels_path}")
    
    # 读取CSV文件（无表头）
    final_volume_labels = pd.read_csv(labels_path, header=None)
    
    print(f'成功加载标签数据: {labels_path}')
    print(f'标签形状: {final_volume_labels.shape}')
    print('前5个标签:')
    print(final_volume_labels.head())
    
    return final_volume_labels
    