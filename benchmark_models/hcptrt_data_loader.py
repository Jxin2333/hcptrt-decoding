import numpy as np  # 导入数值计算库
import pandas as pd  # 导入数据分析库
import glob  # 导入文件路径匹配库
import os  # 导入操作系统接口库
import pickle  # 导入对象序列化库
# from load_confounds import Params9, Params24  # 已弃用的混淆变量处理方法
# from nilearn.input_data import NiftiLabelsMasker, NiftiMasker, NiftiMapsMasker  # 旧版nilearn导入路径
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker, NiftiMapsMasker  # 导入nilearn掩码工具
from termcolor import colored  # 导入终端彩色输出库
import nilearn.datasets  # 导入nilearn数据集工具
from dypac.masker import LabelsMasker, MapsMasker  # 导入dypac掩码工具
from nilearn.interfaces.fmriprep import load_confounds_strategy  # 导入fMRI预处理混淆变量加载工具
from time import time  # 导入时间测量工具
# from nilearn.interfaces.fmriprep import load_confounds  # 已弃用的混淆变量加载方法
import sys
sys.path.append(os.path.join("../"))  # 添加父目录到系统路径
import utils  # 导入自定义工具库

"""
Utilities for first step of reading and processing hcptrt data.
We need to run it just once.
The outputs are medial_data that are saved as fMRI2 and events2.

工具函数集，用于读取和处理HCP任务重复测试(HCPtrt)数据的第一步。
这个处理过程只需要运行一次。
输出结果是保存为fMRI2和events2格式的中间数据。
"""

class DataLoader():
    
    def __init__(self, TR, modality, subject, 
                 bold_suffix, region_approach, resolution, 
                 fMRI2_out_path=None, events2_out_path=None, 
                 raw_data_path=None, pathevents=None, 
                 raw_atlas_dir=None):  #confounds,       
        
        """ 
        Initializer for DataLoader class.
                
        Parameters
        ----------
          TR: int
              Repetition Time
              重复时间，fMRI扫描的时间间隔
          confounds: str
              fMRI confounds generating strategy, 
              e.g. Params9()
              fMRI混淆变量生成策略，例如Params9()
          modality: str
              task, e.g. 'motor'
              任务类型，例如'motor'（运动任务）
          subject: str
              subject ID, e.g. 'sub-01'
              受试者ID，例如'sub-01'
          bold_suffix: str
              BOLD文件的后缀名
          region_approach: str, 
              parcelation method
              e.g.: 'MIST', 'dypac'
              脑区分割方法，例如：'MIST'，'dypac'
          resolution: int
              there are parcellations at different resolutions
              e.g.: 444, 256, 512, 1024
              脑区分割的分辨率，例如：444, 256, 512, 1024
          fMRI2_out_path: str
              path to bold files w directory
              BOLD文件的输出路径
          events2_out_path: str
              path to events files w directory
              事件文件的输出路径
          raw_data_path: str
              path to HCPtrt dataset fMRI files
              HCPtrt数据集fMRI文件的原始路径
          pathevents: str
              path to HCPtrt dataset events files
              HCPtrt数据集事件文件的路径
          raw_atlas_path: str
              path to maskers atlas e.g. 'difumo_atlases'
              掩码图谱的路径，例如'difumo_atlases'
        """
        
        self.TR = TR  # 设置重复时间
#         self.confounds = confounds  # 设置混淆变量（已注释）
        self.modality = modality  # 设置任务类型
        self.subject = subject  # 设置受试者ID
        self.bold_suffix = bold_suffix  # 设置BOLD文件后缀
        self.region_approach = region_approach  # 设置脑区分割方法
        self.resolution = resolution  # 设置分辨率
        self.fMRI2_out_path = fMRI2_out_path  # 设置BOLD文件输出路径
        self.events2_out_path = events2_out_path  # 设置事件文件输出路径
        self.raw_data_path = raw_data_path  # 设置原始数据路径
        self.pathevents = pathevents  # 设置事件文件路径
        self.raw_atlas_dir = raw_atlas_dir  # 设置图谱目录
        
        # 如果BOLD文件输出路径不存在，则创建该目录
        if not os.path.exists(self.fMRI2_out_path):
            os.makedirs(self.fMRI2_out_path)

        # 如果事件文件输出路径不存在，则创建该目录
        if not os.path.exists(self.events2_out_path):
            os.makedirs(self.events2_out_path)
            
        # 如果图谱目录不存在，则创建该目录
        if not os.path.exists(self.raw_atlas_dir):
            os.makedirs(self.raw_atlas_dir)


    def _load_fmri_data(self): 
        
        """
        Out put is a list of preprocessed fMRI files using the
        given masker. (for each modality) 
        
        输出是使用给定掩码处理后的fMRI文件列表（针对每种任务类型）。
        该方法加载原始fMRI数据，应用适当的掩码进行预处理，并返回处理后的数据。
        """

        # 使用glob查找匹配的BOLD文件路径，并按字母顺序排序
        data_path = sorted(glob.glob(self.raw_data_path+'{}/**/*{}*'
                                     .format(self.subject, self.modality)+self.bold_suffix, 
                                     recursive=True))
                
        # 打印当前处理的受试者和任务类型
        print(colored('{}, {}:'.format(self.subject, self.modality), attrs=['bold']))  
        
#         for i in range(0, len(data_path)):
#             print(data_path[i].split('func/', 1)[1])
        
        # 确保不超过14次运行（最多处理15个文件）        
        if (len(data_path) > 15):
            data_extra_files = len(data_path) - 15 
            # 打印被排除的额外文件信息
            print(colored('Regressed out {} extra following fMRI file(s):'
                          .format(data_extra_files), 'red', attrs=['bold']))
            # 显示被排除的文件名
            for i in range(14, len(data_path)):
                print(colored(data_path[i].split('func/', 1)[1], 'red'))
            # 从列表中移除多余的文件
            for i in range(14, len(data_path)):
                data_path.pop()
                                   
        print('The number of bold files:', len(data_path))  # 打印处理的BOLD文件数量
 
        # 生成掩码
        if self.region_approach == 'MIST':  # 如果使用MIST脑区分割方法
                                                                        
            # 创建NiftiLabelsMasker对象，使用MIST图谱和指定分辨率
            masker = NiftiLabelsMasker(labels_img='{}_{}.nii.gz'.format(self.region_approach,
                                                                          self.resolution), 
                                       standardize=True, smoothing_fwhm=5)  # 标准化数据并应用5mm平滑
            
            t0 = time()  # 记录开始时间
            fmri_t = []  # 初始化处理后的fMRI数据列表
            for dpath in data_path:  # 遍历每个BOLD文件路径
                print(dpath.split('func/', 1)[1])  # 打印当前处理的文件名
                
#                 data_fmri = masker.fit_transform(dpath, confounds=self.confounds.load(dpath)) #old nilearn
                
                # 使用新版nilearn加载混淆变量，采用简单去噪策略，包括基本运动参数和全局信号
                conf = load_confounds_strategy(dpath, denoise_strategy="simple",
                                               motion="basic", global_signal="basic") #new nilearn

                # 应用掩码转换数据，同时去除混淆变量的影响
                data_fmri = masker.fit_transform(dpath, confounds=conf[0]) #new nilearn
                fmri_t.append(data_fmri)  # 将处理后的数据添加到列表
                
#                print(dpath.split('func/', 1)[1])
#                print(data_fmri)
                print('shape:', np.shape(data_fmri))  # 打印处理后数据的形状

            # 打印数据处理时间
            print("Data processing time for {} using {} with {} resolution:".format(self.subject, self.region_approach,
                                                                                    self.resolution), round(time()-t0, 3), "s")
              
            
        elif self.region_approach == 'difumo':  # 如果使用difumo脑区分割方法
            
##             num_parcels = int(self.region_approach.split("_", 1)[1])
#            atlas = nilearn.datasets.fetch_atlas_difumo(data_dir=self.raw_atlas_dir, 
#                                                        dimension=self.resolution)
#            atlas_filename = atlas['maps']
#           atlas_labels = atlas['labels']
            t0 = time()  # 记录开始时间
            # 构建difumo图谱文件路径
            atlas_filename = os.path.join(self.raw_atlas_dir + '/{}_atlases/{}/3mm/maps.nii.gz'.format(self.region_approach,
                                                                                                       self.resolution))
            # 创建NiftiMapsMasker对象，使用difumo图谱
            masker = NiftiMapsMasker(maps_img=atlas_filename, standardize=True, 
                                     verbose=5, smoothing_fwhm=5)  # 标准化数据，详细输出，应用5mm平滑
            
            fmri_t = []  # 初始化处理后的fMRI数据列表
            for dpath in data_path:  # 遍历每个BOLD文件路径
#                 data_fmri = masker.fit_transform(dpath, confounds=self.confounds.load(dpath)) #old nilearn
                
                # 使用新版nilearn加载混淆变量
                conf = load_confounds_strategy(dpath, denoise_strategy="simple",
                                               motion="basic", global_signal="basic") #new nilearn

                # 应用掩码转换数据
                data_fmri = masker.fit_transform(dpath, confounds=conf[0]) #new nilearn
                fmri_t.append(data_fmri)  # 将处理后的数据添加到列表

                print('shape:', np.shape(data_fmri))  # 打印处理后数据的形状

            # 打印数据处理时间
            print("Data processing time for {} using {} with {} resolution:".format(self.subject, self.region_approach,
                                                                                    self.resolution), round(time()-t0, 3), "s")


        elif self.region_approach == 'schaefer':  # 如果使用schaefer脑区分割方法

            t0 = time()  # 记录开始时间
            # 构建Schaefer图谱文件路径
            atlas_filename = os.path.join(self.raw_atlas_dir + '/{}_2018/Schaefer2018_{}Parcels_7Networks'\
                                                               '_order_FSLMNI152_1mm.nii.gz'.format(self.region_approach,
                                                                                                    self.resolution))

            # 创建NiftiLabelsMasker对象，使用Schaefer图谱
            masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True,
                                       verbose=5, smoothing_fwhm=5)  # 标准化数据，详细输出，应用5mm平滑

            fmri_t = []  # 初始化处理后的fMRI数据列表
            for dpath in data_path:  # 遍历每个BOLD文件路径
#                 data_fmri = masker.fit_transform(dpath, confounds=self.confounds.load(dpath)) #old nilearn

                # 使用新版nilearn加载混淆变量
                conf = load_confounds_strategy(dpath, denoise_strategy="simple",
                                               motion="basic", global_signal="basic") #new nilearn
                # 应用掩码转换数据
                data_fmri = masker.fit_transform(dpath, confounds=conf[0]) #new nilearn
                fmri_t.append(data_fmri)  # 将处理后的数据添加到列表

                print('shape:', np.shape(data_fmri))  # 打印处理后数据的形状
            # 打印数据处理时间
            print("Data processing time for {} using {} with {} resolution:".format(self.subject, self.region_approach,
                                                                                    self.resolution), round(time()-t0, 3), "s")



        elif self.region_approach == 'dypac':  # 如果使用dypac脑区分割方法
            
#             LOAD_CONFOUNDS_PARAMS = {
#                 "strategy": ["motion", "high_pass", "wm_csf", "global_signal"],
#                 "motion": "basic",
#                 "wm_csf": "basic",
#                 "global_signal": "basic",
#                 "demean": True
#             } # 自定义混淆变量参数
            
            path_dypac = '/data/cisl/pbellec/models'  # dypac模型路径
            # 构建受试者特定的灰质掩码文件路径
            file_mask = os.path.join(path_dypac, 
                                     '{}_space-MNI152NLin2009cAsym_label-GM_mask.nii.gz'.format(self.subject))
            # 构建受试者特定的dypac组件文件路径
            file_dypac = os.path.join(path_dypac,
                                      '{}_space-MNI152NLin2009cAsym_desc-dypac{}_components.nii.gz'.format(
                                          self.subject, self.resolution))
            print('file_mask: ', file_mask)  # 打印掩码文件路径
            print('file_dypac: ', file_dypac, '\n')  # 打印dypac组件文件路径
            # 创建NiftiMasker对象，使用灰质掩码
            masker = NiftiMasker(standardize=True, detrend=False, smoothing_fwhm=5, mask_img=file_mask)
            
            fmri_t = []  # 初始化处理后的fMRI数据列表
            for dpath in data_path:  # 遍历每个BOLD文件路径
                
                # 加载混淆变量
                conf = load_confounds_strategy(dpath, denoise_strategy='simple', global_signal='basic')
#                 conf = load_confounds(dpath, strategy=**LOAD_CONFOUNDS_PARAMS) # 自定义混淆变量参数      
    
                masker.fit(dpath)  # 拟合掩码到数据
                # 创建MapsMasker对象，使用拟合后的掩码和dypac组件
                maps_masker = MapsMasker(masker=masker, maps_img=file_dypac)
                # 转换数据，应用掩码和去除混淆变量
                data_fmri = maps_masker.transform(img=dpath, confound=conf[0])
                fmri_t.append(data_fmri)  # 将处理后的数据添加到列表
                
                print('fMRI file:' ,dpath.split('func/', 1)[1])  # 打印当前处理的文件名
                print('shape:', np.shape(data_fmri), '\n')  # 打印处理后数据的形状
#                print(data_fmri)
                print('\n')
            
        else:  # 如果使用其他脑区分割方法或默认方法
            # 创建默认的NiftiMasker对象，只进行标准化
            masker = NiftiMasker(standardize=True)                   

            fmri_t = []  # 初始化处理后的fMRI数据列表
            for dpath in data_path:  # 遍历每个BOLD文件路径
#                 data_fmri = masker.fit_transform(dpath, confounds=self.confounds.load(dpath)) #old nilearn
                
                # 加载混淆变量
                conf = load_confounds_strategy(dpath, denoise_strategy="simple",
                                               motion="basic", global_signal="basic") #new nilearn
                # 应用掩码转换数据
                data_fmri = masker.fit_transform(dpath, confounds=conf[0]) #new nilearn
                
                
                
                fmri_t.append(data_fmri)  # 将处理后的数据添加到列表

        print('### Reading Nifiti files is done!')  # 打印读取完成信息
        print('-----------------------------------------------')

        # 返回处理后的fMRI数据列表、使用的掩码对象和原始数据路径
        return fmri_t, masker, data_path
   
    
    
    def _load_events_files(self):
        
        """
        Output is a list of relabeled events file.
        
        输出是重新标记后的事件文件列表。
        该方法加载原始事件文件，根据任务类型重新标记条件，并返回处理后的事件数据。
        """

        # 使用glob查找匹配的事件文件路径，并按字母顺序排序
        events_path = sorted(glob.glob(self.pathevents + '{}/**/func/*{}*_events.tsv'
                                       .format(self.subject, self.modality), 
                                       recursive=True))
        
        # 确保不超过14次运行
        if (len(events_path) > 14):
            events_extra_files = len(events_path) - 14
            # 打印被排除的额外文件信息
            print(colored('Regressed out {} extra following events file(s):'
                          .format(events_extra_files), 'red', attrs=['bold']))

            # 显示被排除的文件名
            for i in range(14, len(events_path)):
                print(colored(events_path[i].split('func/', 1)[1], 'red'))

            # 从列表中移除多余的文件
            for i in range(14, len(events_path)):
                events_path.pop()            

        print('The number of events files:', len(events_path))  # 打印处理的事件文件数量
        
        # 标记条件
        events_files = []  # 初始化处理后的事件文件列表
        count = 1  # 初始化计数器
        count_str=str(count)  # 转换为字符串

        for epath in events_path:  # 遍历每个事件文件路径

            # 使用pandas读取TSV格式的事件文件
            event = pd.read_csv(epath, sep="\t", encoding="utf8")
            print(epath.split('func/', 1)[1])  # 打印当前处理的文件名
            print(event.head(5))  # 打印前5行数据
            print(np.shape(event))  # 打印数据形状
            print(np.unique(event.trial_type))  # 打印唯一的试验类型
            
            # 根据不同的任务类型重新标记条件
            if self.modality == 'emotion':  # 情绪任务
                # 将原始标签替换为简化标签
                event.trial_type = event['trial_type'].replace(['response_face',
                                                                'response_shape'],
                                                               ['fear','shape'])
                    
            if self.modality == 'language':  # 语言任务
                # 将所有与故事相关的标签替换为'story'，将所有与数学相关的标签替换为'math'
                event.trial_type = event['trial_type'].replace(['presentation_story',
                                                                'question_story',
                                                                'response_story',
                                                                'presentation_math',
                                                                'question_math',
                                                                'response_math'],
                                                               ['story','story','story',
                                                                'math','math','math']) 

            if self.modality == 'motor':  # 运动任务
                # 将运动响应标签替换为简化标签
                event.trial_type = event['trial_type'].replace(['response_left_foot',
                                                                'response_left_hand',
                                                                'response_right_foot',
                                                                'response_right_hand',
                                                                'response_tongue'],
                                                               ['footL','handL','footR',
                                                                'handR','tongue']) 

            if self.modality == 'relational':  # 关系任务
                # 将关系任务标签替换为简化标签
                event.trial_type = event['trial_type'].replace(['Control','Relational'],
                                                               ['match','relational']) 
                
                
            if self.modality == 'wm':  # 工作记忆任务
                # 将刺激类型和试验类型组合成新的标签
                event.trial_type = event.stim_type.astype(str) + '_' + \
                event.trial_type.astype(str)
                # 将组合标签替换为简化标签
                event.trial_type = event['trial_type'].replace(['Body_0-Back','Body_2-Back',
                                                                'Face_0-Back','Face_2-Back',
                                                                'Place_0-Back','Place_2-Back',
                                                                'Tools_0-Back','Tools_2-Back'],
                                                               ['body0b','body2b','face0b',
                                                                'face2b','place0b','place2b',
                                                                'tool0b','tool2b'])
            
#            print(colored('After relabeling:', attrs=['bold']))
#            print(np.unique(event.trial_type), '\n')
#            print(event.trial_type.head(20))

#----------------------------------------------------------------------------------
            conditions = list(event.trial_type)
#            print('conditions:', conditions)

            # 创建会话索引和计数索引列表
            session_idx = []
            count_idx = []
            for condition in conditions:  # 遍历每个条件

                # 从文件路径中提取会话信息
                ses = epath.split('_task')[0].split('func/')[1].partition('_')[2]  # 提取会话编号
                temp_run = epath.split('run')[1]  # 提取运行部分
                run = utils.between(temp_run, "-", "_")  # 使用工具函数提取运行号
                session = ses + '_run-' + run  # 组合会话和运行信息
                session_idx.append(session)  # 添加到会话索引列表
                count_idx.append(count)  # 添加到计数索引列表

            # 将会话和计数信息添加到事件数据中
            event['session'] =  session_idx  # 添加会话列
            event['count'] =  count_idx  # 添加计数列
#            print(event)

#----------------------------------------------------------------------------------

            events_files.append(event)  # 将处理后的事件数据添加到列表中

            count += 1  # 计数器加1
            count_str=str(count)  # 更新计数字符串
        
        print('### Reading events files is done!')
        print('-----------------------------------------------')

        return events_files

        
############################################## Shima local ###########################################   
# def reading_events2(subject, modality, events2_out_path, region_approach):
    
#     events_outname = events2_out_path + subject + '_' +  modality + '_events2'
#     pickle_in = open(events_outname, "rb")
#     events_files = pd.read_pickle(events_outname)
    
#     return events_files      
######################################################################################################      
    
    
def _check_input(fmri_t, events_files):

    """
    - 移除BOLD文件末尾多余的体积（如果存在）。
    - 检查事件文件和fMRI文件的一致性。
        
    参数
    ----------
    fmri_t: list
        load_fmri_data函数的输出，预处理后的fMRI数据列表
    events_files: list
        load_events_files函数的输出，处理后的事件文件列表
    """

    data_lenght = len(fmri_t)  # 获取fMRI数据列表长度
    data_lenght = int (data_lenght or 0)  # 确保数据长度为整数，如果为空则默认为0

    # 移除多余的体积
    for i in range(0, data_lenght-1):  # 遍历所有相邻的fMRI数据对
        if fmri_t[i].shape != fmri_t[i+1].shape:  # 如果两个fMRI数据形状不一致
            print('There is mismatch in BOLD file size:')  # 打印不匹配信息

            if fmri_t[i].shape > fmri_t[i+1].shape:  # 如果第i个文件比第i+1个文件大       
                a = np.shape(fmri_t[i])[0] - np.shape(fmri_t[i+1])[0]  # 计算多余的体积数量      
                fmri_t[i] = fmri_t[i][0:-a, 0:]  # 从第i个文件中移除多余的体积
                print('The', a,'extra volumes of bold file number', i,'is removed.')  # 打印移除信息
            else:  # 如果第i+1个文件比第i个文件大
                b = np.shape(fmri_t[i+1])[0] - np.shape(fmri_t[i])[0]  # 计算多余的体积数量      
                fmri_t[i+1] = fmri_t[i+1][0:-b, 0:]  # 从第i+1个文件中移除多余的体积
                print('The', b,'extra volumes of bold file number', i+1,'is removed.')  # 打印移除信息

    if len(events_files) != len(fmri_t):  # 检查事件文件数量与fMRI文件数量是否一致
        print('Miss-matching between events and fmri files')  # 打印不匹配信息
        print('Number of Nifti files:' ,len(fmri_t))  # 打印fMRI文件数量
        print('Number of events files:' ,len(events_files))  # 打印事件文件数量
    else:  # 如果数量一致
        print('Events and fMRI files are Consistent.')  # 打印一致性信息

    print('### Cheking data is done!')  # 打印检查完成信息
    print('-----------------------------------------------')  # 打印分隔线


    
def _save_files(fmri_t, events_files, subject, modality, 
                fMRI2_out_path, events2_out_path):
    
# def _save_files(events_files, subject, modality, 
#                 events2_out_path, region_approach):
    
    """
    - 为每个任务保存预处理后的fMRI矩阵文件。
    - 为每种模态保存事件文件作为pickle文件。
    
    参数
    ----------
    fmri_t: list
        预处理后的fMRI数据列表
    events_files: list
        处理后的事件文件列表
    subject: str
        受试者ID
    modality: str
        任务类型
    fMRI2_out_path: str
        fMRI数据保存路径
    events2_out_path: str
        事件数据保存路径
    """
    
    # 保存fMRI数据
    bold_outname = fMRI2_out_path + subject + '_' + modality + '_fMRI2.npy'  # 构建输出文件名
    np.save(bold_outname, fmri_t)  # 保存fMRI数据为numpy数组文件

    temp = np.load(bold_outname, allow_pickle=True)  # 加载保存的文件以验证
    fmri_t = temp  # 更新fmri_t变量
    
    print('Bold file:', bold_outname)  # 打印保存的文件路径
    print('### Saving Nifiti files as matrices is done!')  # 打印保存完成信息
    print('-----------------------------------------------')  # 打印分隔线
    
    # 保存事件数据
    events_outname = events2_out_path + subject + '_' + modality + '_events2'  # 构建事件文件输出路径
    events_dict = events_files  # 获取事件文件字典
    pickle_out = open(events_outname,"wb")  # 以二进制写入模式打开文件
    pickle.dump(events_dict, pickle_out)  # 将事件数据序列化到文件
    pickle_out.close()  # 关闭文件

    print('Events pickle file:', events_outname)  # 打印事件文件路径
    print('### Saving events pickle files is done!')  # 打印保存完成信息
    print('-----------------------------------------------')  # 打印分隔线
    

    
def postproc_data_loader(subject, modalities, region_approach, resolution): # confounds, 
    """处理并加载数据的主函数。
    
    参数
    ----------
    subject : str
        受试者ID
    modalities : list
        要处理的任务类型列表
    region_approach : str
        脑区分割方法，如'MIST'、'difumo'、'schaefer'或'dypac'
    resolution : int
        脑区分割的分辨率
    """
    
    TR = 1.49  # 设置重复时间（秒）

#    ##### Elm #####
#    pathevents = '/data/neuromod/projects/ml_models_tutorial/data/hcptrt/HCPtrt_events_DATA/'
#    raw_data_path = '/data/neuromod/projects/ml_models_tutorial/data/hcptrt/derivatives/'
#    proc_data_path = '/home/SRastegarnia/hcptrt_decoding_Shima/data/'


    ##### CC #####
   # pathevents = '/home/rastegar/projects/def-pbellec/rastegar/hcptrt/HCPtrt_events_DATA/'
    pathevents = '/home/rastegar/scratch/hcptrt/'  # 事件文件路径
#    raw_data_path = pathevents + 'derivatives/fmriprep-20.2lts/fmriprep/'
    raw_data_path = '/home/rastegar/scratch/hcptrt/derivatives/fmriprep-20.2lts/fmriprep/'  # 原始数据路径
    proc_data_path = '/home/rastegar/projects/def-pbellec/rastegar/hcptrt_decoding_shima/data/'  # 处理后数据保存路径


    bold_suffix = '_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'  # BOLD文件后缀

    raw_atlas_dir = os.path.join(proc_data_path, "raw_atlas_dir")  # 图谱目录

    # 构建fMRI数据输出路径
    fMRI2_out_path = proc_data_path + 'medial_data/fMRI2/{}/{}/{}/'.format(region_approach,
                                                                           resolution, subject)        
    # 构建事件数据输出路径
    events2_out_path = proc_data_path + 'medial_data/events2/{}/{}/{}/'.format(region_approach,
                                                                               resolution, subject)

    for modality in modalities:  # 遍历每种任务类型
        print(colored(modality,'red', attrs=['bold']))  # 打印当前处理的任务类型

        # 初始化DataLoader对象
        load_data = DataLoader(TR = TR,  
                               modality = modality, subject = subject, 
                               bold_suffix = bold_suffix,
                               region_approach = region_approach,
                               resolution = resolution,
                               fMRI2_out_path = fMRI2_out_path, 
                               events2_out_path = events2_out_path, 
                               raw_data_path = raw_data_path, 
                               pathevents = pathevents, 
                               raw_atlas_dir = raw_atlas_dir) #confounds = confounds,

        # 加载并处理fMRI数据
        fmri_t, masker, data_path  = load_data._load_fmri_data()

        # 加载并处理事件数据
        events_files = load_data._load_events_files()
#             events_files = _reading_events2(subject, modality, events2_out_path, region_approach) # Shima local

        # 检查数据一致性
        _check_input(fmri_t, events_files)

        # 保存处理后的数据
        _save_files(fmri_t, events_files, subject, modality,  
                    fMRI2_out_path, events2_out_path)

