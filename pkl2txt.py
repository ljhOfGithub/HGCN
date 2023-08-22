import pickle
import pprint
import joblib
# 读取.pkl文件并加载为字典对象
# file_path = '/home/jupyter-ljh/data/mydata/HGCN-main/LIHC/lihc_patients.pkl'
# file_path = '/home/jupyter-ljh/data/mydata/HGCN-main/LIHC/lihc_split.pkl'
# file_path = '/home/jupyter-ljh/data/mydata/HGCN-main/LIHC/lihc_sur_and_time.pkl'
# file_path = '/home/jupyter-ljh/data/mydata/HGCN-main/LIHC/lihc_data.pkl'
# file_path = '/home/jupyter-ljh/data/mydata/HGCN-main/LIHC/model/lihc_model_state_dict_fold_0.pkl'
# file_path = '/home/jupyter-ljh/data/mntdata/data0/LI_jihao/HGCN_BRCA/np/TCGA-3C-AALI-01Z-00-DX1.F6E9A5DF-D8FB-45CF-B4BD-C6B76294C291.pkl'
# file_path = '/home/jupyter-ljh/data/mydata/HGCN-main/data_split/esca_split.pkl'
# file_path = '/home/jupyter-ljh/data/mydata/HGCN-main/LIHC/lihc_split.pkl'
# file_path='/home/jupyter-ljh/data/mydata/HGCN-main/sur_and_time.pkl'
# file_path='/home/jupyter-ljh/data/mydata/HGCN-main/slide_id.pkl'
# file_path='/home/jupyter-ljh/data/mydata/HGCN-main/split.pkl'
# file_path='/home/jupyter-ljh/data/mydata/HGCN-main/ttt_cli_feas.pkl'
# file_path='/home/jupyter-ljh/data/mydata/HGCN-main/data_split/split.pkl'
# file_path='/home/jupyter-ljh/data/mydata/HGCN-main/t_rna_fea.pkl'
file_path='/home/jupyter-ljh/data/mydata/HGCN-main/LIHC/lihc_patients.pkl'

# file_path='/home/jupyter-ljh/'
with open(file_path, 'rb') as file:
    # my_dict = pickle.load(file)
    my_dict = joblib.load(file)

# 现在my_dict是一个包含.pkl文件中数据的字典对象，你可以像操作其他字典一样使用它
# print(my_dict)
formatted_str = pprint.pformat(my_dict)
# pprint.pprint(my_dict, indent=4)#输出到控制台

# with open('./sur_and_time.txt', 'w') as file:
# with open('./slide_id.txt', 'w') as file:
# with open('./split.txt', 'w') as file:
# with open('./ttt_cli_feas.txt', 'w') as file:
with open('LIHC/lihc_patients.txt', 'w') as file:
    file.write(formatted_str)#输出到文件
        
