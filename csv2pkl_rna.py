# import pandas as pd
# import joblib

# # 读取 gene_family.csv 文件
# gene_family_data = pd.read_csv('../gene_family.csv')

# # 读取 tcga_brca_all_clean.csv 文件
# tcga_data = pd.read_csv('../tcga_brca_all_clean.csv')

# # 创建一个空字典来存储结果
# result_dict = {}

# # 创建一个空列表来存储跳过的基因
# skipped_genes = set()

# # 获取 tcga_brca_all_clean.csv 中的病人编号列表
# patient_ids = tcga_data['case_id'].tolist()

# # 遍历 tcga_brca_all_clean.csv 中的每一行
# for gene_family_column in gene_family_data.columns:
#     # 获取基因家族名称
#     gene_family = gene_family_column

#     # 遍历每个基因名称
#     for index, row in gene_family_data.iterrows():
#         # 获取基因名称
#         gene_name = row[gene_family_column]
#         print(gene_name)
#         if not pd.isna(gene_name):
#             # 遍历 tcga_brca_all_clean.csv 中的每一行
#             for index, row in tcga_data.iterrows():
#                 # 获取当前行的病人编号
#                 case_id = row['case_id']

#                 # 创建一个空字典来存储当前病人的基因数据
#                 patient_gene_data = {}

#                 # 添加 "_rnaseq" 到基因名称
#                 gene_name_with_suffix = gene_name + "_rnaseq"

#                 # 检查基因是否存在于当前行中
#                 if gene_name_with_suffix in row.index:
#                     # 获取当前行的基因数据
#                     gene_data = row[gene_name_with_suffix]

#                     # 将基因数据存储到当前基因家族的字典中
#                     patient_gene_data[gene_family] = gene_data

#                     # 将当前病人的基因数据字典添加到结果字典中，键是病人编号
#                     if case_id not in result_dict:
#                         result_dict[case_id] = {}
#                     result_dict[case_id].update(patient_gene_data)
#                 else:
#                     # 如果找不到该基因，则记录下来
#                     skipped_genes.add(gene_name_with_suffix)

# # 使用 joblib 将结果字典保存为Pickle文件
# # joblib.dump(result_dict, 'output.pkl')

# # 将跳过的基因保存到文本文件
# # with open('skipped_genes.txt', 'w') as skipped_genes_file:
# #     skipped_genes_file.write('\n'.join(skipped_genes))
    
# # 打印结果字典，其中键是病人编号，值是每个基因家族对应的基因数据字典
# import joblib
# # 将结果字典保存为Pickle文件
# with open('t_rna_fea.pkl', 'wb') as pkl_file:
#     joblib.dump(result_dict, pkl_file)
# import pprint
# with open('t_rna_fea.txt', 'w') as txt_file:
#     # for patient_id, gene_data in result_dict.items():
#     #     txt_file.write(f'Patient ID: {patient_id}\n')
#     #     for gene_family, value in gene_data.items():
#     #         txt_file.write(f'{gene_family}: {value}\n')
#     #     txt_file.write('\n')
#     pprint(result_dict)

# import pandas as pd
# import joblib

# # 读取 gene_family.csv 文件
# gene_family_data = pd.read_csv('../gene_family.csv')

# # 读取 tcga_brca_all_clean.csv 文件
# tcga_data = pd.read_csv('../tcga_brca_all_clean.csv')

# # 创建一个空字典来存储结果
# result_dict = {}

# # 创建一个空集合来存储跳过的基因，以确保基因名称的去重
# skipped_genes = set()

# # 获取 tcga_brca_all_clean.csv 中的病人编号列表
# patient_ids = tcga_data['case_id'].tolist()

# gene_families_order = [
#     "cytokines and growth factors",
#     "cell differentiation markers",
#     "protein kinases",
#     "oncogenes",
#     "tumor suppressors"
# ]


# # 遍历 tcga_brca_all_clean.csv 中的每一行
# for index, row in tcga_data.iterrows():
#     # 获取当前行的病人编号
#     case_id = row['case_id']

#     # 创建五个空列表，分别用于存储五个基因家族的数据
#     gene_family_data_lists = [[] for _ in range(5)]

#     # 遍历 gene_family.csv 中的每一列，并按指定的顺序处理
#     for i, gene_family in enumerate(gene_families_order):
#         # 获取当前列的基因名称
#         print(gene_family,index)
#         if index < len(gene_family_data):
#             gene_name = gene_family_data[gene_family].iloc[index]

#             # 检查基因名称是否是 NaN
#             if not pd.isna(gene_name):
#                 # 添加 "_rnaseq" 到基因名称
#                 gene_name_with_suffix = gene_name + "_rnaseq"

#                 # 检查基因是否存在于当前行中
#                 if gene_name_with_suffix in row.index:
#                     # 获取当前行的基因数据
#                     gene_data = row[gene_name_with_suffix]

#                     # 将基因数据添加到对应基因家族的列表中
#                     gene_family_data_lists[i].append(gene_data)
#                 else:
#                     # 如果找不到该基因，则记录下来
#                     skipped_genes.add(gene_name_with_suffix)

#     # 将五个基因家族的数据列表添加到结果字典中，键是病人编号
#     result_dict[case_id] = gene_family_data_lists

# # 使用 joblib 将结果字典保存为Pickle文件
# joblib.dump(result_dict, 't_rna_fea.pkl')

# # # 将跳过的基因保存到文本文件
# # with open('skipped_genes.txt', 'w') as skipped_genes_file:
# #     skipped_genes_file.write('\n'.join(skipped_genes))

import pandas as pd
import joblib

# 读取 gene_family.csv 文件
gene_family_data = pd.read_csv('../gene_family.csv')

# 读取 tcga_brca_all_clean.csv 文件
tcga_data = pd.read_csv('../tcga_brca_all_clean.csv')

# 创建一个空字典来存储结果
result_dict = {}

# 创建一个空集合来存储跳过的基因，以确保基因名称的去重
skipped_genes = set()

# 获取 tcga_brca_all_clean.csv 中的病人编号列表
patient_ids = tcga_data['case_id'].tolist()

# 定义基因家族列表的顺序
gene_families_order = [
    "cytokines and growth factors",
    "cell differentiation markers",
    "protein kinases",
    "oncogenes",
    "tumor suppressors"
]

# 遍历 tcga_brca_all_clean.csv 中的每一行
for index, row in tcga_data.iterrows():
    # 获取当前行的病人编号
    case_id = row['case_id']

    # 创建五个空列表，分别用于存储当前病人的五个基因家族的数据
    patient_gene_data_lists = [[] for _ in range(5)]

    # 遍历 gene_family.csv 中的每一列，并按指定的顺序处理
    for i, gene_family in enumerate(gene_families_order):
        # 遍历基因家族的每个基因
        for gene_name in gene_family_data[gene_family]:
            # 检查基因名称是否是 NaN
            if not pd.isna(gene_name):
                # 添加 "_rnaseq" 到基因名称
                gene_name_with_suffix = gene_name + "_rnaseq"

                # 检查基因是否存在于当前行中
                if gene_name_with_suffix in row.index:
                    # 获取当前行的基因数据
                    gene_data = row[gene_name_with_suffix]

                    # 将基因数据添加到对应基因家族的列表中
                    patient_gene_data_lists[i].append(gene_data)
                else:
                    # 如果找不到该基因，则记录下来
                    skipped_genes.add(gene_name_with_suffix)

    # 将当前病人的五个基因家族的数据列表添加到结果字典中，键是病人编号
    result_dict[case_id] = patient_gene_data_lists

# 使用 joblib 将结果字典保存为Pickle文件
joblib.dump(result_dict, 't_rna_fea.pkl')

# # 将跳过的基因保存到文本文件
# with open('skipped_genes.txt', 'w') as skipped_genes_file:
#     skipped_genes_file.write('\n'.join(skipped_genes))
