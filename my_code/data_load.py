import os
import glob
import pandas as pd
from nilearn.input_data import NiftiLabelsMasker

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