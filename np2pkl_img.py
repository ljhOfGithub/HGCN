# import os
# import pickle

# # 遍历目录下的所有 pkl 文件
# data_dict = {}
# pkl_dir = '/home/jupyter-ljh/data/mntdata/data0/LI_jihao/HGCN_BRCA/np'
# for filename in os.listdir(pkl_dir):
#     if filename.endswith('.pkl'):
#         pkl_path = os.path.join(pkl_dir, filename)
#         with open(pkl_path, 'rb') as f:
#             pkl_obj = pickle.load(f)
#             new_key = filename[:12]
#             new_key_with_ext = new_key + '.png'
#             data_dict[new_key_with_ext] = pkl_obj

# # 保存字典为 pkl 文件
# with open('t_img_fea.pkl', 'wb') as f:
#     pickle.dump(data_dict, f)

# # 将字典格式化输出到 txt 文件
# with open('./output.txt', 'w') as f:
#     for key, value in data_dict.items():
#         f.write(f"{key}: {value}\n")
# import os
# import pickle
# import joblib
# # 指定目录
# directory = '/home/jupyter-ljh/data/mntdata/data0/LI_jihao/HGCN_BRCA/np'

# # 遍历目录下的所有 pkl 文件
# pkl_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]

# # 构建新的字典
# new_dict = {}
# for pkl_file in pkl_files:
#     pkl_path = os.path.join(directory, pkl_file)
    
#     # 读取原始对象
#     with open(pkl_path, 'rb') as f:
#         # original_object = pickle.load(f)
#         original_object = joblib.load(f)
    
#     # 构建新对象并添加到新字典
#     new_key = pkl_file[:12]
#     new_value = {key + '.png': value for key, value in original_object.items()}
#     new_dict[new_key] = new_value

# # 保存新字典为 pkl 文件
# with open('./t_img_fea.pkl', 'wb') as f:
#     # pickle.dump(new_dict, f)
#     joblib.dump(new_dict, f)

# # 将新字典格式化输出为 txt 文件
# with open('./output.txt', 'w') as f:
#     for key, value in new_dict.items():
#         f.write(f'{key}: {value}\n')
