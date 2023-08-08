import json
import torch
import joblib
import numpy as np
from torch_geometric.data import Data

'''
patient id list 
['TCGA-LN-A49R','TCGA-2H-A9GQ','TCGA-LN-A4A3',....]
'''
patients = joblib.load('patients.pkl')
'''
censorship status and observed time .
{'TCGA-LN-A49R': [0, 13.371663244353183],
 'TCGA-2H-A9GQ': [1, 4.205338809034908],
 'TCGA-LN-A4A3': [0, 12.747433264887063],
 ...
 }
'''
sur_and_time = joblib.load('sur_and_time.pkl')
patient_sur_type = {}
for x in patients: 
    patient_sur_type[x] = sur_and_time[x][0]

time = []
patient_and_time = {}
for x in patients:
    time.append(sur_and_time[x][-1])
    patient_and_time[x] = sur_and_time[x][-1]
'''
t_img_fea contains the pretrained features of the patch.

{'TCGA-VR-A8EX': {'23-25.png': array([ 0.0958841 ,  0.0922773 ,  0.10698985, ..., -0.51396465,
          3.0901225 , -1.7472196 ], dtype=float32),
      '22-26.png': array([ 0.08808839,  0.10823577,  0.10959034, ...,  0.3193577 ,
          3.0152717 , -2.3737502 ], dtype=float32),
          ....}
 'TCGA-LN-A4A6': {'11-7.png': array([ 0.09743017,  0.16183454,  0.03875097, ...,  0.42391908,
          2.4486601 , -0.37408364], dtype=float32),
      '36-16.png': array([ 0.07777312,  0.0657203 ,  0.16304843, ..., -1.218863  ,
          0.26710236,  0.6011251 ], dtype=float32),  
          ....}
  ....}        
'''
t_img_fea = joblib.load('t_img_fea.pkl')

'''
For genomic profile, we employ GSEA to generate five genomic embeddings: 
1) Tumor Supression, 2) Oncogenesis, 3) Protein Kinases, 4) Cellular Differentiation, and 5) Cytokines and Growth.
from https://www.gsea-msigdb.org/gsea/msigdb/gene_families.jsp?ex=1

{'TCGA-LN-A49R': [[-0.4064,-1.3284, 2.637,0.0197, 0.8055,...],
                [-0.314,-0.6875, -2.1022,-1.5837,-0.6759,...],
                ...]
  ...}
  
'''
# t_rna_fea = joblib.load('t_rna_fea.pkl')
# rna_fea_1024 = {}
# for x in patients:
#     if x in t_rna_fea:
#         tmp = []
#         for i,z in enumerate(t_rna_fea[x]):
#             u=[]
#             j=0
#             for o in z:
#                 u.append(float(o))
#                 j+=1
#             #all node features are aligned to 1024 dimensions by zero padding
#             for k in range(j,1024):
#                 u.append(0.)
#             tmp.append(u)  
#         rna_fea_1024[x]= tmp

'''
ttt_cli_feas contains the clinical records, 
discrete values give different numerical values, e.g 'male': 0 , 'female':1

{'TCGA-LN-A49R': [0,1.095890410958904,46],
'TCGA-2H-A9GQ': [0,1.7758405977584057, 80],
...}
'''
ttt_cli_feas = joblib.load("ttt_cli_feas.pkl")
t_cli_feas = {}
for x in ttt_cli_feas:
    a=[]
    for i in range(len(ttt_cli_feas[x])):
            a.append(ttt_cli_feas[x][i])
    t_cli_feas[x] = a
    
for i in range(10):
    zz = []
    for x in t_cli_feas:
#         print(x,len(t_cli_feas[x]))
        zz.append(t_cli_feas[x][i])
    zz = np.array(zz)
    maxx = np.max(zz)
    minn = np.min(zz)
    for x in t_cli_feas:
        t_cli_feas[x][i] = (t_cli_feas[x][i]-(maxx+minn)/2)/(maxx-minn)*2 

# all node features are aligned to 1024 dimensions by zero padding
onehot_cli = {}
for x in patients:
    tmp=np.zeros((len(t_cli_feas[x]),1024))            
    k=0
    for i,z in enumerate(t_cli_feas[x]):
            tmp[k][i]=t_cli_feas[x][i]
            k+=1
    onehot_cli[x] = tmp

# feature_img = {}
# feature_rna = {}
# feature_cli = {}
# data_type = {}
# for x in patients:
#     f_img = []
#     f_rna = []
#     f_cli = []
#     t_type = []
#     if x in t_img_fea:
#         for z in t_img_fea[x]:
#             f_img.append(t_img_fea[x][z])
#         t_type.append('img')
#     # if x in rna_fea_1024:   
#     #     for r in rna_fea_1024[x]:
#     #         f_rna.append(r) 
#     #     t_type.append('rna')
#     # if x in  onehot_cli:       
#     #     for r in onehot_cli[x]:
#     #         f_cli.append(r) 
#     #     t_type.append('cli')
#     data_type[x]=t_type
#     feature_img[x] = f_img
#     # feature_rna[x] = f_rna
#     # feature_cli[x] = f_cli

# def get_edge_index_image(id):    
#     start = []
#     end = []    
#     if id in t_img_fea:
#         patch_id = {}
#         i=0
#         import pdb
#         # pdb.set_trace()
#         for x in t_img_fea[id]:
#             patch_id[x.split('.')[0]] = i
#             i+=1
#     #     print(patch_id)
#         try:
#             for x in patch_id:
#         #         print(x)
                
#                 i = int(x.split('_')[0])
#                 # j = int(x.split('.')[0].split('-')[1])
#                 j = int(x.split('_')[1])
#                 if str(i)+'-'+str(j+1) in patch_id:
#                     start.append(patch_id[str(i)+'-'+str(j)])
#                     end.append(patch_id[str(i)+'-'+str(j+1)])
#                 if str(i)+'-'+str(j-1) in patch_id:
#                     start.append(patch_id[str(i)+'-'+str(j)])
#                     end.append(patch_id[str(i)+'-'+str(j-1)])
#                 if str(i+1)+'-'+str(j) in patch_id:
#                     start.append(patch_id[str(i)+'-'+str(j)])
#                     end.append(patch_id[str(i+1)+'-'+str(j)])
#                 if str(i-1)+'-'+str(j) in patch_id:
#                     start.append(patch_id[str(i)+'-'+str(j)])
#                     end.append(patch_id[str(i-1)+'-'+str(j)])
#                 if str(i+1)+'-'+str(j+1) in patch_id:
#                     start.append(patch_id[str(i)+'-'+str(j)])
#                     end.append(patch_id[str(i+1)+'-'+str(j+1)])
#                 if str(i-1)+'-'+str(j+1) in patch_id:
#                     start.append(patch_id[str(i)+'-'+str(j)])
#                     end.append(patch_id[str(i-1)+'-'+str(j+1)])
#                 if str(i+1)+'-'+str(j-1) in patch_id:
#                     start.append(patch_id[str(i)+'-'+str(j)])
#                     end.append(patch_id[str(i+1)+'-'+str(j-1)])
#                 if str(i-1)+'-'+str(j-1) in patch_id:
#                     start.append(patch_id[str(i)+'-'+str(j)])
#                     end.append(patch_id[str(i-1)+'-'+str(j-1)])
#         except Exception as e:
#             import traceback
#             import pprint
#             import sys
#             # traceback.print_exc()
#             exc_type, exc_value, exc_traceback = sys.exc_info()
    
#             # 将堆栈信息写入文件
#             with open('./error_log.txt', 'a') as f:
#                 traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)

#             # 获取当前变量的值
#             current_locals = locals()

#             # 将变量值写入文件
#             with open('variables.txt', 'a') as f:
#                 pprint.pprint(current_locals, stream=f)
#                 # print(x)
#                 # print(patch_id)

#     return [start,end]  
# # 获取边的连接关系
# from tqdm import tqdm
# import time
# # items = range(len(patients))
# # pbar = tqdm(total=len(patients), position=0, leave=True, bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:40}{r_bar}')

# # for id in patients:
#     # print(get_edge_index_image(id))
#     # print([start,end])
# # edge_index_image = {}
# # for id in patients:
# #     # 在迭代过程中更新进度条
# #     edge_index_image[id]=get_edge_index_image(id)
# #     pbar.update(1)
    
# # pbar.close()
# # exit()
# def get_edge_index_rna(id):   
#     start = []
#     end = []  
#     if id in t_rna_fea:
#         for i in range(len(feature_rna[id])):
#             for j in range(len(feature_rna[id])):
#                 if i!=j:
#                     start.append(j)
#                     end.append(i)
#     return [start,end]  
# def get_edge_index_cli(id):   
#     start = []
#     end = []   
#     if id in t_cli_feas:
#         for i in range(len(feature_cli[id])):
#             for j in range(len(feature_cli[id])):
#                 if i!=j:
#                     start.append(j)
#                     end.append(i)
#     return [start,end]  
# all_data = {}
# num=0
# for id in patients:
#     print(id)
#     node_img=torch.tensor(feature_img[id],dtype=torch.float)
#     # node_rna=torch.tensor(feature_rna[id],dtype=torch.float)
#     # node_cli=torch.tensor(feature_cli[id],dtype=torch.float)
#     # edge_index_model = torch.tensor(get_edge_index_model(id),dtype=torch.long)
#     # edge_index_rna = torch.tensor(get_edge_index_rna(id),dtype=torch.long)
#     # edge_index_cli = torch.tensor(get_edge_index_cli(id),dtype=torch.long)
#     edge_index_image = torch.tensor(get_edge_index_image(id),dtype=torch.long)
#     sur_type=torch.tensor([patient_sur_type[id]])
#     data_id = id 
#     t_data_type = data_type[id]
#     # data=Data(x_img=node_img,x_rna=node_rna,x_cli=node_cli,sur_type=sur_type,data_id=data_id,data_type=t_data_type,edge_index_model=edge_index_model,edge_index_image=edge_index_image,edge_index_rna=edge_index_rna,edge_index_cli=edge_index_cli) 
        
#     data=Data(x_img=node_img,x_rna=None,x_cli=None,sur_type=sur_type,data_id=data_id,data_type=t_data_type,edge_index_model=None,edge_index_image=edge_index_image,edge_index_rna=None,edge_index_cli=None) 
#     all_data[id] = data
#     num+=1
#     # print(data)
#     print(num)
    
# '''
# Data(x_img=[1406, 1024], x_rna=[5, 1024], x_cli=[10, 1024], sur_type=[1], data_id='TCGA-2H-A9GQ', data_type=[3], edge_index_image=[2, 9550], edge_index_rna=[2, 20], edge_index_cli=[2, 90])
# Data(x_img=[1127, 1024], x_rna=[5, 1024], x_cli=[10, 1024], sur_type=[1], data_id='TCGA-L5-A4OH', data_type=[3], edge_index_image=[2, 8334], edge_index_rna=[2, 20], edge_index_cli=[2, 90])
# '''
# # joblib.dump(all_data,'all_data.pkl')
# torch.save(all_data, 'all_data.pt')
# torch.save(all_data, 'all_data_cli_img.pt')
