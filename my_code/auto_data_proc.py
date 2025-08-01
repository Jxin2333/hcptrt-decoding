import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.input_data import NiftiLabelsMasker,NiftiMapsMasker
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from data_load import *

# 设置基本参数
TR = 1.49                           # 重复时间(Repetition Time)，单位秒
subject = 'sub-01'                  # 受试者编号
# modality = 'wm'                     # 任务类型：工作记忆(working memory)
# session = 'ses-001'
session ='**'                       #全部session

# for modality in ['wm',"emotion",'gambling','language','relational', 'motor', 'social']:
for modality in ['wm',"emotion"]:
    # for img in ['MIST_122','MIST_444','Schaefer100','Schaefer400','Schaefer1000','difumo128','difumo256','difumo512','difumo1024']:
    for img in ['difumo512','difumo1024','Schaefer100','difumo128','difumo256']:
        try:
            print('-------------------------------------------------------------------------------------------------------------------------------')
            print("#########################################"+"现在是"+img+'的'+modality+"#####################################################")
            print('-------------------------------------------------------------------------------------------------------------------------------')

            labels_img = '/Users/zhangyexin/Documents/hcptrt-output/Parcellations/'+img+'.nii.gz'

            # 设置输出路径
            out_path = '/Users/zhangyexin/Documents/hcptrt-output/fMRI/'+ img+'/' + modality + '/' # 输出文件保存路径
            bold_suffix = 'space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'  # BOLD数据文件后缀

            # 搜索BOLD数据文件路径
            data_path = sorted(glob.glob('/Users/zhangyexin/hcptrt.fmriprep/{}/{}/func/*{}*'
                                .format(subject, session, modality) + bold_suffix, recursive=True))

            # 搜索事件文件路径ses-001
            events_path = sorted(glob.glob('/Users/zhangyexin/hcptrt/{}/{}/func/*{}*_events.tsv'
                                .format(subject, session, modality), recursive=True))

            print('BOLD文件数量:', len(data_path))
            print('events文件数量:', len(events_path))

            # 先创建目录
            if not os.path.exists(out_path):
                os.makedirs(out_path)


            bold_files = []  # 存储处理后的BOLD数据
            for dpath in data_path:    
                # 创建脑区标签掩膜器
                if 'difumo' in img.lower():
                    masker = NiftiMapsMasker(labels_img,  # 使用脑区模板
                                            standardize=True,                  # 标准化数据
                                            detrend = False,                   # 不进行去趋势
                                            smoothing_fwhm = 5).fit()          # 5mm平滑核
                else:
                    masker = NiftiLabelsMasker(labels_img,  # 使用脑区模板
                                            standardize=True,                  # 标准化数据
                                            detrend = False,                   # 不进行去趋势
                                            smoothing_fwhm = 5).fit()          # 5mm平滑核

                # 提取脑区时间序列，同时应用混淆变量校正
                data_fmri = masker.fit_transform(dpath, confounds = confound_load_9param(dpath))    
                bold_files.append(data_fmri)  # 添加到列表中

            # 保存处理后的BOLD数据
            bold_outname = out_path +subject + '_' + modality + '_fMRI2.npy'
            np.save(bold_outname, bold_files)
            print('######### BOLD数据读取完成! #########')  

            # 加载和处理事件文件
            print('events文件数量:', len(events_path))

            events_files = []  # 存储事件数据
            for epath in events_path: 
                # 读取事件文件(TSV格式)
                event = pd.read_csv(epath, sep = "\t", encoding = "utf8", header = 0)
                
                # 根据任务类型重新标记试验类型
                if modality == 'wm':  # 工作记忆任务
                    # 将试验类型和刺激类型组合，如"2-Back_Face"
                    event.trial_type = event.trial_type.astype(str) + '_' + \
                    event.stim_type.astype(str)
                    
                if modality == 'relational':  # 关系任务
                    # 将试验类型和指令组合
                    event.trial_type = event.trial_type.astype(str) + '_' + \
                    event.instruction.astype(str)

                events_files.append(event)  # 添加到列表中

            print('######### 事件文件读取完成! #########') 

            data_lenght = len(bold_files)  # 数据文件数量
            data_lenght = int (data_lenght or 0)  # 确保为整数

            print(bold_files[0].shape)

            # 检查BOLD文件的时间点数量，确保一致性
            for i in range(0, data_lenght-1):
                if bold_files[i].shape > bold_files[i+1].shape:         
                    # 如果当前文件比下一个文件有更多时间点，则截断多余部分
                    a = np.shape(bold_files[i])[0] - np.shape(bold_files[i+1])[0]        
                    bold_files[i] = bold_files[i][0:-a, 0:]  # 移除多余的时间点
                    print('BOLD文件', i, '有', a, '个多余的时间点')

            # 检查事件文件和BOLD文件数量是否匹配
            if len(events_files) != len(bold_files):
                print('事件文件和fMRI文件数量不匹配')
                print('Nifti文件数量:' ,len(bold_files))
                print('事件文件数量:' ,len(events_files)) 


            # 定义一个空列表，用于保存每个被试/每次实验的标签数组
            labels_files = []

            # 遍历所有的事件文件（每个事件文件对应一次实验任务）
            for events_file in events_files:
                task_durations = []    # 保存每个任务类型的持续时间
                task_modalities = []   # 保存每个任务类型的名称
                row_counter = 0        # 初始化行计数器

                # 首行任务类型加入列表
                task_modalities.append(events_file.iloc[0]['trial_type'])
                rows_no = len(events_file.axes[0])  # 获取事件文件的总行数（即事件数量）

                # 遍历事件文件的每一行，提取不同任务类型及其持续时间
                for i in range(1, rows_no):
                    if (events_file.iloc[i]['trial_type'] != events_file.iloc[i-1]['trial_type']):
                        task_modalities.append(events_file.iloc[i]['trial_type'])  # 新任务类型
                        duration = (events_file.iloc[i]['onset']) - (events_file.iloc[row_counter]['onset'])  # 持续时间计算
                        task_durations.append(duration)
                        row_counter = i  # 更新行计数器为当前行

                # 添加最后一个任务类型的持续时间
                task_durations.append(events_file.iloc[i]['duration'])

                # 检查任务类型数量是否与持续时间数量一致
                if (len(task_durations) != len(task_modalities)):
                    print('error: tasks and durations do not match')

                # 转换为 NumPy 数组，方便后续计算
                task_durations = np.array(task_durations)
                task_modalities = np.array(task_modalities)

                # 根据持续时间生成每个任务类型对应的体素数（volume 数）
                volume_no = []
                for t in task_durations:
                    volume_round = np.round((t)/TR).astype(int)  # 用 TR（时间分辨率）除以持续时间，得到 volume 数
                    volume_no.append(volume_round)

                # 内部函数：计算体素总数（所有任务体素的和）
                def _sum(arr): 
                    sum = 0
                    for i in arr:
                        sum = sum + i
                    return(sum) 

                ans_round = _sum(volume_no)  # 总体素数

                # 通过 masker 提取一个样本 fMRI 数据用于确定总体素数量
                sample_fmri = masker.fit_transform(data_path[0], confounds = confound_load_9param(data_path[0]))
                null_ending = sample_fmri.shape[0] - ans_round  # 计算结尾部分未标注的体素数

                # 根据任务名称和体素数量生成最终标签数组
                final_array = []
                if (len(task_modalities) == len(task_durations) == len(volume_no)):
                    for l in range(len(task_modalities)):
                        f = ((task_modalities[l],) * volume_no[l])  # 每个任务重复对应的 volume 次
                        final_array.append(f)

                # 添加结尾的 null 标签（未标注体素）
                if null_ending > 0:
                    end_volume = (('null',) * null_ending)
                    final_array.append(end_volume)
                
                # 将多维数组扁平化为一维标签列表
                flat_list = [item for sublist in final_array for item in sublist]
                volume_labels = np.array(flat_list)  
                labels_files.append(volume_labels)  # 加入最终列表

            # 把所有的标签文件展开为一个大的一维数组
            flat_labels_files = [item for sublist in labels_files for item in sublist]
            flat_volume_labels = np.array(flat_labels_files)

            # 获取每个 bold 文件的体素数量（shape[0] 表示时间点数量）
            shape = np.shape(bold_files[0])[0]
            flat_volume_labels = np.reshape(flat_volume_labels, (data_lenght * shape, 1))  # 重新整形为 (总体素数量, 1)

            # 将所有 bold 数据也展平（横向合并所有时间点）
            flat_bold = [item for sublist in bold_files for item in sublist]
            flat_bold_files = np.array(flat_bold)

            # 检查标签和 BOLD 数据是否一致（每个体素都应有一个标签）
            if (len(flat_bold_files[:, 0]) != len(flat_volume_labels[:, 0])):
                print('error: labels and bold flat files mismatche')

            # 定义新的标签列表，最终将包括 HRF_lag 修改后的标签
            HRFlag_volume_labels = []

            b = 0  # 初始化索引
            l = len(flat_volume_labels[:, 0])  # 总体素标签的数量

            # 遍历标签数组，检测任务类型的变化以识别新任务开始
            while (b < (l - 1)):  
                # 如果当前位置与下一个位置的标签不同，表示新任务的开始
                if (flat_volume_labels[b, 0] != flat_volume_labels[b + 1, 0]):
                    HRFlag_volume_labels.append(flat_volume_labels[b, 0])  # 当前这个点保留原标签

                    # 判断新任务至少有 4 个连续体素（即持续时间足够长）
                    if (flat_volume_labels[b + 1, 0] == flat_volume_labels[b + 2, 0] == 
                        flat_volume_labels[b + 3, 0] == flat_volume_labels[b + 4, 0]):
                        # 若满足，则将前 3 个体素标记为 HRF_lag
                        for j in range(1, 4):  # 添加3个 HRF_lag 标签
                            HRFlag_volume_labels.append('HRF_lag')
                        b = b + 4  # 跳过已标记部分（原本应为 b+4，但此处可能应为 b+3，详见后续优化）
                    else:
                        b = b + 1  # 若新任务不够长，正常推进1个位置
                        
                else:
                    # 若当前标签与下一标签相同，表示仍在同一任务中，直接复制原标签
                    HRFlag_volume_labels.append(flat_volume_labels[b, 0])
                    b = b + 1

            # 添加最后一个体素的标签（避免遗漏）
            HRFlag_volume_labels.append(flat_volume_labels[l - 1, 0])


            # 从第一个事件文件中提取出所有的试验类型（trial_type），并将其转换为一个列表。
            # 这里的 events_files[0] 应该代表第一个被试的事件数据。
            categories = list(events_files[0].trial_type)

            # 定义一个集合（set），包含所有我们不感兴趣、希望从数据集中剔除的标签。
            # 使用集合可以提高查找效率。
            unwanted = {'countdown', 'cross_fixation', 'Cue', 'new_bloc_right_hand',
                        'new_bloc_right_foot', 'new_bloc_left_foot', 'new_bloc_tongue',
                        'new_bloc_left_hand', 'new_bloc_control', 'new_bloc_relational',
                        'new_bloc_shape', 'new_bloc_face', 'countdown_nan', 'Cue_nan', 'HRF_lag', 'null'}

            # 使用列表推导式（list comprehension）从 categories 中筛选出所有不在 unwanted 集合中的标签。
            # 这实际上是在创建“有用”标签的白名单。
            categories = [c for c in categories if c not in unwanted]

            # 将筛选后的所有“有用”标签转换为一个集合，然后再次转为列表。
            # 这样做可以去除重复的标签，得到所有独特的任务条件（conditions）。
            conditions = list(set(categories))

            # 初始化一个空列表，用于存储最终筛选出来的任务标签。
            final_volume_labels = []

            # 获取 fMRI 数据中每个时间点（volume）的特征数量，也就是脑区（parcel）的数量。
            # flat_bold_files[1] 应该代表一个被试的所有 fMRI 数据的二维矩阵，它的列数就是 parcel_no。
            parcel_no = np.shape(flat_bold_files[1])[0]

            # 初始化一个空的 NumPy 数组，用于存储最终筛选出来的 fMRI 数据。
            # 它的形状是 (0, parcel_no)，表示目前有0行数据，但已经确定了列数。
            final_bold_files = np.empty((0, parcel_no), int)

            # 开始一个循环，遍历所有经过 HRF 延迟处理后的标签（HRFlag_volume_labels）。
            # HRFlag_volume_labels 包含了所有时间点（TR）的标签，包括我们想要和不想要的。
            for i in range(0, len(HRFlag_volume_labels)):
                # 检查当前循环到的标签是否不在 unwanted 集合中。
                if (HRFlag_volume_labels[i] not in unwanted):
                    # 如果标签有用，就将其添加到 final_volume_labels 列表中。
                    final_volume_labels.append(HRFlag_volume_labels[i])
                    # 将当前时间点对应的 fMRI 数据（flat_bold_files[i, :]）追加到 final_bold_files 数组中。
                    # np.append 会创建一个新的数组，这是一个比较耗费性能的操作。
                    final_bold_files = np.append(final_bold_files,
                                                np.array([flat_bold_files[i, :]]), axis=0)
                    
            # 计算最终任务条件的数量。
            num_cond = len(set(categories))
            # print(num_cond)
            # print(conditions)

            # 将最终的标签列表转换为 Pandas DataFrame。
            df_lable = pd.DataFrame(final_volume_labels)
            # 将标签 DataFrame 保存为 CSV 文件。
            # out_path 应该是输出文件所在的目录，modality 是任务类型，如 'wm'。
            # sep=',' 指定使用逗号分隔，index=False 不保存行索引，header=None 不保存列标题。
            df_lable.to_csv(out_path + modality + '_final_labels.csv', sep=',', index=False, header=None)
            # 构建最终 fMRI 数据的保存路径。
            df_fMRI = out_path +subject + '_' + modality + '_final_fMRI.npy'
            # 使用 NumPy 将 final_bold_files 数组保存为 .npy 格式文件。
            # 这是保存数值数据的一种高效二进制格式。
            np.save(df_fMRI, final_bold_files)
        except Exception as e:
             print(f"处理文件 {out_path} 时出错：{e!r}，已跳过")
             continue