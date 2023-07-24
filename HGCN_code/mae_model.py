import os
import sys
import copy
import math
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import softmax
from torch_geometric.data import Data
from torch_geometric.nn import GlobalAttention
from torch_geometric.nn import SAGEConv,LayerNorm
from mae_utils import get_sinusoid_encoding_table,Block
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

class my_GlobalAttention(torch.nn.Module):
    def __init__(self, gate_nn, nn=None):
        super(my_GlobalAttention, self).__init__()
        self.gate_nn = gate_nn
        self.nn = nn

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gate_nn)
        reset(self.nn)


    def forward(self, x, batch, size=None):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size
        
        gate = self.gate_nn(x).view(-1, 1)
        x = self.nn(x) if self.nn is not None else x
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = softmax(gate, batch, num_nodes=size)
        out = scatter_add(gate * x, batch, dim=0, dim_size=size)

        return out,gate


    def __repr__(self):
        return '{}(gate_nn={}, nn={})'.format(self.__class__.__name__,
                                              self.gate_nn, self.nn)
    
    
def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)
    #对张量进行截断正态分布初始化，均值为 mean，标准差为 std，范围限制在 [-std, std]
class PretrainVisionTransformerEncoder(nn.Module):#预训练视觉transformer编码器
    """ Vision Transformer with support for patch or hybrid CNN input stage
    视觉transformer支持patch或混合CNN输入阶段
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=512, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_learnable_pos_emb=False,train_type_num=3):
        #在自然语言处理中，Transformer 网络使用位置编码来帮助模型处理序列数据的位置信息。位置编码通常是一个固定的向量，其维度与输入序列的特征维度相同。在训练过程中，位置编码不会发生变化。
        #而当 use_learnable_pos_emb 设置为 True 时，位置编码是可学习的，也就是说它的参数会在训练过程中被优化调整。这样做的好处是，模型可以根据数据自动学习到更适合任务的位置编码，而不是使用固定的编码。
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
#与其他模型维度保持一致
#         self.patch_embed = PatchEmbed(
#             img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
#         num_patches = self.patch_embed.num_patches

        self.patch_embed = nn.Linear(embed_dim,embed_dim)
        #nn.Linear 是 PyTorch 中用于创建全连接层的类。它将输入张量的每个元素与权重矩阵相乘，并加上偏置项，生成输出张量。
        num_patches = train_type_num#模态数

        # TODO: Add the cls token
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            # sine-cosine positional embeddings 
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        #最终，列表 dpr 就包含了从 0 到 drop_path_rate 的 depth 个均匀间隔的数值。这个列表可能用于后续的操作，例如在 Stochastic Depth 算法中使用，用于控制网络中某些层的丢弃概率。
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        #通过使用 nn.ModuleList，我们可以将多个自注意力块堆叠在一起，并按顺序调用它们来构建一个深度的自注意力模型。
        self.norm =  norm_layer(embed_dim)
        #在这里，self.norm 的作用是对输入特征进行规范化，确保输入特征具有统一的分布和范围。
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        #self.head 作为网络的输出层，根据 num_classes 的取值，可能是一个用于分类的线性层，也可能是一个恒等映射。
        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        # trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)#对所有子模块调用初始化函数


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    #这个函数通过遍历网络的所有子模块 m，检查是否为线性层 (nn.Linear) 或 Layer Normalization 层 (nn.LayerNorm)，然后分别对其进行不同的初始化操作。        

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}
    #在这段代码中，no_weight_decay 是一个自定义的方法，用于返回不参与权重衰减（Weight Decay）的参数名称集合。
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    #用于重新设置模型的分类器（head）。head 是模型的一个属性，它表示模型的分类器层。
    def forward_features(self, x, mask):
        x = self.patch_embed(x)
        
        # cls_tokens = self.cls_token.expand(batch_size, -1, -1) 
        # x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        #self.pos_embed.type_as(x).to(x.device).clone().detach() 得到一个新的位置编码矩阵，它与输入特征张量 x 数据类型一致且不具有梯度。然后，这个位置编码矩阵会与输入特征张量 x 相加，用于在输入特征中添加位置信息。
        B, _, C = x.shape
        x_vis = x[~mask].reshape(B, -1, C) # ~mask means visible
        #目的是将原始张量 x 中被掩码 mask 选择出来的元素重新整理成一个新的三维张量
        for blk in self.blocks:
            x_vis = blk(x_vis)
        #通过不断遍历 self.blocks 中的超图块，我们可以将输入的超图特征进行多次变换和处理，从而得到更丰富和有意义的特征表示。
        x_vis = self.norm(x_vis)
        return x_vis

    def forward(self, x, mask):
        x = self.forward_features(x, mask)
        x = self.head(x)
        return x

class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, patch_size=16, num_classes=512, embed_dim=512, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, num_patches=196,train_type_num=3,
                 ):
        super().__init__()
        self.num_classes = num_classes
#         assert num_classes == 3 * patch_size ** 2
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
#         self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, return_token_num):
        for blk in self.blocks:
            x = blk(x)

        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:])) # only return the mask tokens predict pixels
        else:
            x = self.head(self.norm(x)) # [B, N, 3*16^2]

        return x

class PretrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224, 
                 patch_size=16, 
                 encoder_in_chans=3, 
                 encoder_num_classes=0, 
                 encoder_embed_dim=512, 
                 encoder_depth=12,
                 encoder_num_heads=12, 
                 decoder_num_classes=512, 
                 decoder_embed_dim=512, 
                 decoder_depth=8,
                 decoder_num_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.3,
                 drop_path_rate=0.3, 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 num_classes=0, # avoid the error from create_fn in timm
                 in_chans=0, # avoid the error from create_fn in timm
                 train_type_num=3,
                 ):
        super().__init__()
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=encoder_in_chans, 
            num_classes=encoder_num_classes, 
            embed_dim=encoder_embed_dim, 
            depth=encoder_depth,
            num_heads=encoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb,
            train_type_num=train_type_num)

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size, 
            num_patches=3,
            num_classes=decoder_num_classes, 
            embed_dim=decoder_embed_dim, 
            depth=decoder_depth,
            num_heads=decoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            train_type_num=train_type_num)

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
#         self.mask_token = torch.zeros(1, 1, decoder_embed_dim).to(device)
        

        self.pos_embed = get_sinusoid_encoding_table(train_type_num, decoder_embed_dim)

        trunc_normal_(self.mask_token, std=.02)
    #该代码定义了一个包含编码器和解码器的视觉预训练模型，用于图像生成任务。编码器负责将图像编码为特征表示，解码器负责根据特征表示生成图像预测结果，并使用位置嵌入和掩码标记来帮助模型生成准确的图像预测结果。

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, mask):
        
        x_vis = self.encoder(x, mask) # [B, N_vis, C_e]
        x_vis = self.encoder_to_decoder(x_vis) # [B, N_vis, C_d]

        B, N, C = x_vis.shape
        
        # we don't unshuffle the correct visible token order, 
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)

        # notice: if N_mask==0, the shape of x is [B, N_mask, 3 * 16 * 16]
        x = self.decoder(x_full, 0) # [B, N_mask, 3 * 16 * 16]

        tmp_x = torch.zeros_like(x).to(device)
        Mask_n = 0
        Truth_n = 0
        for i,flag in enumerate(mask[0][0]):
            if flag:  
                tmp_x[:,i] = x[:,pos_emd_vis.shape[1]+Mask_n]
                Mask_n += 1
            else:
                tmp_x[:,i] = x[:,Truth_n]
                Truth_n += 1
        return tmp_x
    #这个前向传播过程的目的是使用预训练的视觉 Transformer 模型对具有遮挡信息的图像进行重建。
    #它通过 Encoder 编码可见部分的信息，并使用 Decoder 生成遮挡部分的预测。最后，通过重建过程，将预测的遮挡部分插入到正确的位置，得到完整的输出序列。


def Mix_mlp(dim1):
    
    return nn.Sequential(
            nn.Linear(dim1, dim1),
            nn.GELU(),
            nn.Linear(dim1, dim1))

class MixerBlock(nn.Module):
    def __init__(self,dim1,dim2):
        super(MixerBlock,self).__init__() 
        
        self.norm = LayerNorm(dim1)
        self.mix_mip_1 = Mix_mlp(dim1)
        self.mix_mip_2 = Mix_mlp(dim2)
        
    def forward(self,x): 
        
        y = self.norm(x)
        y = y.transpose(0,1)
        y = self.mix_mip_1(y)
        y = y.transpose(0,1)
        x = x + y
        y = self.norm(x)
        x = x + self.mix_mip_2(y)
        
#         y = self.norm(x)
#         y = y.transpose(0,1)
#         y = self.mix_mip_1(y)
#         y = y.transpose(0,1)
#         x = self.norm(y)
        return x



def MLP_Block(dim1, dim2, dropout=0.3):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)
    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ReLU(),
            nn.Dropout(p=dropout))

def GNN_relu_Block(dim2, dropout=0.3):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)
    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    return nn.Sequential(
#             GATConv(in_channels=dim1,out_channels=dim2),
            nn.ReLU(),
            LayerNorm(dim2),
            nn.Dropout(p=dropout))




class fusion_model_mae_2(nn.Module):
    def __init__(self,in_feats,n_hidden,out_classes,dropout=0.3,train_type_num=3):
        super(fusion_model_mae_2,self).__init__() 

        self.img_gnn_2 = SAGEConv(in_channels=in_feats,out_channels=out_classes)         
        self.img_relu_2 = GNN_relu_Block(out_classes)  
        self.rna_gnn_2 = SAGEConv(in_channels=in_feats,out_channels=out_classes)          
        self.rna_relu_2 = GNN_relu_Block(out_classes)      
        self.cli_gnn_2 = SAGEConv(in_channels=in_feats,out_channels=out_classes)         
        self.cli_relu_2 = GNN_relu_Block(out_classes) 
#         TransformerConv

        att_net_img = nn.Sequential(nn.Linear(out_classes, out_classes//4), nn.ReLU(), nn.Linear(out_classes//4, 1))        
        self.mpool_img = my_GlobalAttention(att_net_img)

        att_net_rna = nn.Sequential(nn.Linear(out_classes, out_classes//4), nn.ReLU(), nn.Linear(out_classes//4, 1))        
        self.mpool_rna = my_GlobalAttention(att_net_rna)        

        att_net_cli = nn.Sequential(nn.Linear(out_classes, out_classes//4), nn.ReLU(), nn.Linear(out_classes//4, 1))        
        self.mpool_cli = my_GlobalAttention(att_net_cli)


        att_net_img_2 = nn.Sequential(nn.Linear(out_classes, out_classes//4), nn.ReLU(), nn.Linear(out_classes//4, 1))        
        self.mpool_img_2 = my_GlobalAttention(att_net_img_2)

        att_net_rna_2 = nn.Sequential(nn.Linear(out_classes, out_classes//4), nn.ReLU(), nn.Linear(out_classes//4, 1))        
        self.mpool_rna_2 = my_GlobalAttention(att_net_rna_2)        

        att_net_cli_2 = nn.Sequential(nn.Linear(out_classes, out_classes//4), nn.ReLU(), nn.Linear(out_classes//4, 1))        
        self.mpool_cli_2 = my_GlobalAttention(att_net_cli_2)
        
        
        
        self.mae = PretrainVisionTransformer(encoder_embed_dim=out_classes, decoder_num_classes=out_classes, decoder_embed_dim=out_classes, encoder_depth=1,decoder_depth=1,train_type_num=train_type_num)
        self.mix = MixerBlock(train_type_num, out_classes)
        
        self.lin1_img = torch.nn.Linear(out_classes,out_classes//4)
        self.lin2_img = torch.nn.Linear(out_classes//4,1)        
        self.lin1_rna = torch.nn.Linear(out_classes,out_classes//4)
        self.lin2_rna = torch.nn.Linear(out_classes//4,1) 
        self.lin1_cli = torch.nn.Linear(out_classes,out_classes//4)
        self.lin2_cli = torch.nn.Linear(out_classes//4,1)         

        self.norm_img = LayerNorm(out_classes//4)
        self.norm_rna = LayerNorm(out_classes//4)
        self.norm_cli = LayerNorm(out_classes//4)
        self.relu = torch.nn.ReLU() 
        self.dropout=nn.Dropout(p=dropout)


    def forward(self,all_thing,train_use_type=None,use_type=None,in_mask=[],mix=False):

        if len(in_mask) == 0:
            mask = np.array([[[False]*len(train_use_type)]])
        else:
            mask = in_mask

        data_type = use_type
        x_img = all_thing.x_img
        x_rna = all_thing.x_rna
        x_cli = all_thing.x_cli

        data_id=all_thing.data_id
        edge_index_img=all_thing.edge_index_image
        edge_index_rna=all_thing.edge_index_rna
        edge_index_cli=all_thing.edge_index_cli

        
        save_fea = {}
        fea_dict = {}
        num_img = len(x_img)
        num_rna = len(x_rna)
        num_cli = len(x_cli)      
               
            
        att_2 = []
        pool_x = torch.empty((0)).to(device)
        if 'img' in data_type:
            x_img = self.img_gnn_2(x_img,edge_index_img) 
            x_img = self.img_relu_2(x_img)   
            batch = torch.zeros(len(x_img),dtype=torch.long).to(device)
            pool_x_img,att_img_2 = self.mpool_img(x_img,batch)
            att_2.append(att_img_2)
            pool_x = torch.cat((pool_x,pool_x_img),0)
        if 'rna' in data_type:
            x_rna = self.rna_gnn_2(x_rna,edge_index_rna) 
            x_rna = self.rna_relu_2(x_rna)   
            batch = torch.zeros(len(x_rna),dtype=torch.long).to(device)
            pool_x_rna,att_rna_2 = self.mpool_rna(x_rna,batch)
            att_2.append(att_rna_2)
            pool_x = torch.cat((pool_x,pool_x_rna),0)
        if 'cli' in data_type:
            x_cli = self.cli_gnn_2(x_cli,edge_index_cli) 
            x_cli = self.cli_relu_2(x_cli)   
            batch = torch.zeros(len(x_cli),dtype=torch.long).to(device)
            pool_x_cli,att_cli_2 = self.mpool_cli(x_cli,batch)
            att_2.append(att_cli_2)
            pool_x = torch.cat((pool_x,pool_x_cli),0)
        
        fea_dict['mae_labels'] = pool_x


        if len(train_use_type)>1:
            if use_type == train_use_type:
                mae_x = self.mae(pool_x,mask).squeeze(0)
                fea_dict['mae_out'] = mae_x
            else:
                k=0
                tmp_x = torch.zeros((len(train_use_type),pool_x.size(1))).to(device)
                mask = np.ones(len(train_use_type),dtype=bool)
                for i,type_ in enumerate(train_use_type):
                    if type_ in data_type:
                        tmp_x[i] = pool_x[k]
                        k+=1
                        mask[i] = False
                mask = np.expand_dims(mask,0)
                mask = np.expand_dims(mask,0)
                if k==0:
                    mask = np.array([[[False]*len(train_use_type)]])
                mae_x = self.mae(tmp_x,mask).squeeze(0)
                fea_dict['mae_out'] = mae_x   


            save_fea['after_mae'] = mae_x.cpu().detach().numpy() 
            if mix:
                mae_x = self.mix(mae_x)
                save_fea['after_mix'] = mae_x.cpu().detach().numpy() 

            k=0
            if 'img' in train_use_type and 'img' in use_type:
                x_img = x_img + mae_x[train_use_type.index('img')] 
                k+=1
            if 'rna' in train_use_type and 'rna' in use_type:
                x_rna = x_rna + mae_x[train_use_type.index('rna')]  
                k+=1
            if 'cli' in train_use_type and 'cli' in use_type:
                x_cli = x_cli + mae_x[train_use_type.index('cli')]  
                k+=1
            
 
        att_3 = []
        pool_x = torch.empty((0)).to(device)

        
        if 'img' in data_type:
            batch = torch.zeros(len(x_img),dtype=torch.long).to(device)
            pool_x_img,att_img_3 = self.mpool_img_2(x_img,batch)
            att_3.append(att_img_3)
            pool_x = torch.cat((pool_x,pool_x_img),0)
        if 'rna' in data_type:
            batch = torch.zeros(len(x_rna),dtype=torch.long).to(device)
            pool_x_rna,att_rna_3 = self.mpool_rna_2(x_rna,batch)
            att_3.append(att_rna_3)
            pool_x = torch.cat((pool_x,pool_x_rna),0)
        if 'cli' in data_type:
            batch = torch.zeros(len(x_cli),dtype=torch.long).to(device)
            pool_x_cli,att_cli_3 = self.mpool_cli_2(x_cli,batch)
            att_3.append(att_cli_3)
            pool_x = torch.cat((pool_x,pool_x_cli),0) 
            

        
        x = pool_x
        
        x = F.normalize(x, dim=1)
        fea = x
        
        k=0
        if 'img' in data_type:
            fea_dict['img'] = fea[k]
            k+=1
        if 'rna' in data_type:
            fea_dict['rna'] = fea[k]       
            k+=1
        if 'cli' in data_type:
            fea_dict['cli'] = fea[k]
            k+=1

        
        k=0
        multi_x = torch.empty((0)).to(device)

        if 'img' in data_type:
            x_img = self.lin1_img(x[k])
            x_img = self.relu(x_img)
            x_img = self.norm_img(x_img)
            x_img = self.dropout(x_img)    

            x_img = self.lin2_img(x_img).unsqueeze(0) 
            multi_x = torch.cat((multi_x,x_img),0)
            k+=1
        if 'rna' in data_type:
            x_rna = self.lin1_rna(x[k])
            x_rna = self.relu(x_rna)
            x_rna = self.norm_rna(x_rna)
            x_rna = self.dropout(x_rna) 

            x_rna = self.lin2_rna(x_rna).unsqueeze(0) 
            multi_x = torch.cat((multi_x,x_rna),0)  
            k+=1
        if 'cli' in data_type:
            x_cli = self.lin1_cli(x[k])
            x_cli = self.relu(x_cli)
            x_cli = self.norm_cli(x_cli)
            x_cli = self.dropout(x_cli)

            x_cli = self.lin2_rna(x_cli).unsqueeze(0) 
            multi_x = torch.cat((multi_x,x_cli),0)  
            k+=1  
        one_x = torch.mean(multi_x,dim=0)
   
        return (one_x,multi_x),save_fea,(att_2,att_3),fea_dict  










