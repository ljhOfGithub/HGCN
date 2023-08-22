import pandas as pd
import pickle
import pprint
import joblib
import csv
import numpy as np
# 读取CSV文件，假设目标列名为 'target_column'
# csv_file = './HGCN-main/full_slide.csv'
# csv_file = './Patch-GCN-master/dataset_csv/tcga_brca_all_clean.csv'
# df = pd.read_csv(csv_file)

# 提取目标列为列表
# target_column_list = df['slide_id'].tolist()

# 提取每个元素的前12个字符并存储为一个列表
# slide_id_list = df['slide_id'].apply(lambda x: x[:12]).tolist()

# 读取 CSV 文件，指定只读取 case_id、survival_months 和 censorship 列
# df = pd.read_csv('./tcga_brca_all_clean.csv', usecols=['case_id', 'survival_months', 'censorship'])
# df = pd.read_csv('../tcga_brca_all_clean.csv', usecols=['case_id', 'is_female', 'age'])

# 保存为pkl文件
# with open('./full_slide.pkl', 'wb') as f:
#     pickle.dump(target_column_list, f)

# 加载pkl文件进行验证
# with open('./full_slide.pkl', 'rb') as f:
#     loaded_list = pickle.load(f)

# 将数据整理成字典格式
# data_dict = {}
# for index, row in df.iterrows():
#     case_id = row['case_id']
#     censorship = row['censorship']
#     survival_months = row['survival_months']
#     data_dict[case_id] = [censorship, survival_months]
# data_dict = {}  # 用于保存<case_id>:[is_female, age]的字典
# with open('../tcga_brca_all_clean.csv', 'r') as csv_file:
#     csv_reader = csv.DictReader(csv_file)
#     for row in csv_reader:
#         case_id = row['case_id']
#         is_female = row['is_female']
#         age = row['age']
#         data_dict[case_id] = [is_female, age]


# with open('./patients.pkl', 'wb') as f:
#     # pickle.dump(slide_id_list, f)
#     joblib.dump(slide_id_list, f)

    
# with open('./patients.pkl', 'rb') as f:
#     loaded_list = pickle.load(f)
    
# 将字典存储为 pickle 文件
# with open('sur_and_time.pkl', 'wb') as f:
#     # pickle.dump(data_dict, f)
#     joblib.dump(data_dict, f)
# with open('ttt_cli_feas.pkl', 'wb') as pkl_file:
#     # pickle.dump(data_dict, pkl_file)
#     joblib.dump(data_dict, pkl_file)

    

# # print(len(loaded_list))  # 输出目标列转换为的列表
# pprint.pprint(data_dict,indent=4)

# formatted_str = pprint.pformat(data_dict)
# with open('./output.txt', 'w') as file:
#     file.write(formatted_str)#输出到文件


#读取csv文件，取出case_id列，is_female列，age列，保存为字典，格式是<case_id>:[is_female,age]，将该字典其保存为pkl文件

# import csv
# import joblib

# # 读取CSV文件并提取所需列到字典
# data_dict = {}  # 用于保存<case_id>:[is_female, age]的字典

# with open('../tcga_brca_all_clean.csv', 'r') as csv_file:
#     csv_reader = csv.DictReader(csv_file)
#     for row in csv_reader:
#         case_id = row['case_id']
#         is_female = row['is_female']
#         age = row['age']
#         data_dict[case_id] = [is_female, age]

# # 保存字典为pkl文件
# with open('data_split/ttt_cli_feas.pkl', 'wb') as pkl_file:
#     joblib.dump(data_dict, pkl_file)

# print("Data saved as pkl file.")

import pandas as pd
import pickle

# 创建一个空列表，用于存储每个CSV文件的数据
data_list = []

# 遍历五个CSV文件
# for i in range(0, 5):  # 假设CSV文件命名为 file1.csv, file2.csv, ..., file5.csv
#     csv_file_name = f'data_split/splits_{i}_rename.csv'
#     column1_data = []
#     column2_data = []
#     column3_data = []
#     # 读取CSV文件
#     csv_data = pd.read_csv(csv_file_name)
#     # csv_data = csv_data.dropna() #一定要删除nan
#     # 将CSV的三列数据存储为一个NumPy数组
#     columns = csv_data.columns
#     column_data = {}

#     for column in columns:
#         column_data[column] = csv_data[column].dropna().tolist()
#         import pdb; pdb.set_trace()
#     # 创建包含新列的DataFrame
#     new_df = pd.DataFrame(column_data)
#     columns_as_arrays = [csv_data[col].values for col in csv_data.columns]
    
#     # 将数组添加到列表中
#     data_list.append(columns_as_arrays)

for i in range(5):
    # 构建CSV文件名，假设文件名为 file0.csv, file1.csv, ...
    csv_file_name = f'data_split/splits_{i}_rename.csv'

    # 读取CSV文件
    csv_data = pd.read_csv(csv_file_name)

    # 将每列数据转换为NumPy数组
    column1_array = np.array(csv_data['train_patients'], dtype='<U12')
    column2_array = np.array(csv_data['val_patients'], dtype='<U12')
    # column2_array = column2_array[~pd.isna(column2_array)]
    # column2_array = column2_array[~np.isnan(column2_array)]
    column2_array = column2_array[column2_array != 'nan']
    import pdb; pdb.set_trace()
    column3_array = np.array(csv_data['test_patients'], dtype='<U12')
    # column3_array = column3_array[~pd.isna(column3_array)]
    # column3_array = column3_array[~np.isnan(column3_array)]
    column3_array = column3_array[column3_array != 'nan']
    

    # 创建一个列表，将三列数据作为元素添加到列表中
    columns_list = [column1_array, column2_array, column3_array]

    # 将这个列表添加到data_list中
    data_list.append(columns_list)

# 将列表保存为一个Pickle文件
with open('data_split/split.pkl', 'wb') as pkl_file:
    joblib.dump(data_list, pkl_file)
