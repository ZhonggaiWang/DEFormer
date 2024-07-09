import pdb
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import sys
import torch.distributed as dist
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
sys.path.append("./wrapper/bilateralfilter/build/lib.linux-x86_64-3.8")
# from bilateralfilter import bilateralfilter, bilateralfilter_batch

def get_spacial_bce_loss(cam,label,per_pic_thre):
    #cam [b,c,h,w] label [b,c] per_pic_thre [b,c+1] cam_flatten [b,c,h*w]
    b,c,h,w = cam.size()
    cam_flatten = cam.view(b, c, -1)
    
    
    
    fg_per_pic_thre = per_pic_thre[:,1:]
    fg_per_pic_thre_t = torch.round(fg_per_pic_thre *(h*w)).to(int).cuda()
    spacial_bce_loss = 0
    for i in range(b):
        fg_channel = cam_flatten.detach()[i][label[i] == 1]
        fg_channel_sorted,fg_channel_sorted_index = torch.sort(fg_channel,dim=-1,descending=True)
        thre_t_idx = fg_per_pic_thre_t[i][label[i] == 1]
        thre_t = fg_channel_sorted[torch.arange(fg_channel.size()[0]),thre_t_idx]
        # thre_t_idx = fg_channel_sorted_index[fg_per_pic_thre_t[i][label[i]==1]]
        # thre_t = 
        thre_t_dim_class = torch.full((c,), 9999.0).cuda()
        thre_t_dim_class[label[i]==1] = thre_t
        # [2,1]->[20]         
        #    [c,h*w]                        [c,h*w]       [c,1]        
        generate_label = torch.where((cam_flatten.detach()[i] >= thre_t_dim_class.unsqueeze(-1)),torch.tensor(1).cuda(),torch.tensor(0).cuda())


#focal_loss——————————————————————————————————————————————————————————————————————————————————————————————————————
        generate_label_class_dim = torch.sum(generate_label,dim=0)
        negetive_sample = torch.where(generate_label_class_dim == 0)
        positive_sample = torch.where(generate_label_class_dim > 0)
        
        _negetive_sample = negetive_sample[0]
        _positive_sample = positive_sample[0]
        
        spacial_bce_loss_neg = F.multilabel_soft_margin_loss(cam_flatten[i, :, _negetive_sample.tolist()], generate_label[:, _negetive_sample.tolist()])
        spacial_bce_loss_pos = F.multilabel_soft_margin_loss(cam_flatten[i, :, _positive_sample.tolist()], generate_label[:, _positive_sample.tolist()])
        
        spacial_bce_loss += 0.1 * spacial_bce_loss_neg + 0.9 * spacial_bce_loss_pos
        
        # print('pos:',spacial_bce_loss_pos,' neg:',spacial_bce_loss_neg)
#origin loss——————————————————————————————————————————————————————————————————————————————————————————————————————
        # spacial_bce_loss += F.multilabel_soft_margin_loss(cam_flatten[i],generate_label)
    # print("spacial_bce_loss:PRINTING IN SPACIAL-BCE BLOCK",spacial_bce_loss/b)
        
    return spacial_bce_loss / b

    

def get_masked_ptc_loss(inputs, mask):

    b, c, h, w = inputs.shape
    # 2 768 28 28
    inputs = inputs.reshape(b, c, h*w)

    def cos_sim(x):
        x = F.normalize(x, p=2, dim=1, eps=1e-8)
        cos_sim = torch.matmul(x.transpose(1,2), x)
        return torch.abs(cos_sim)

    inputs_cos = cos_sim(inputs)

    pos_mask = mask == 1
    neg_mask = mask == 0
    loss = 0.5*(1 - torch.sum(pos_mask * inputs_cos) / (pos_mask.sum()+1)) + 0.5 * torch.sum(neg_mask * inputs_cos) / (neg_mask.sum()+1)
    return loss
#这个相当于是分割网络的seg——loss
def get_seg_loss(pred, label, ignore_index=255):
    
    bg_label = label.clone()
    bg_label[label!=0] = ignore_index
    bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    fg_label = label.clone()
    fg_label[label==0] = ignore_index
    fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)

    return (bg_loss + fg_loss) * 0.5


def get_energy_loss(img, logit, label, img_box, loss_layer, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):

    pred_prob = F.softmax(logit, dim=1)
    crop_mask = torch.zeros_like(pred_prob[:,0,...])

    for idx, coord in enumerate(img_box):
        crop_mask[idx, coord[0]:coord[1], coord[2]:coord[3]] = 1

    _img = torch.zeros_like(img)
    _img[:,0,:,:] = img[:,0,:,:] * std[0] + mean[0]
    _img[:,1,:,:] = img[:,1,:,:] * std[1] + mean[1]
    _img[:,2,:,:] = img[:,2,:,:] * std[2] + mean[2]

    loss = loss_layer(_img, pred_prob, crop_mask, label.type(torch.uint8).unsqueeze(1), )

    return loss.cuda()
class CTCLoss_neg(nn.Module):
    def __init__(self, ncrops=10, temp=1.0,):
        super().__init__()
        self.temp = temp
        # self.center_momentum = center_momentum
        self.ncrops = ncrops
        # self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        # self.teacher_temp_schedule = np.concatenate((
        #     np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
        #     np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        # ))

class CTCLoss_neg(nn.Module):
    def __init__(self, ncrops=10, temp=1.0,):
        super().__init__()
        self.temp = temp
        # self.center_momentum = center_momentum
        self.ncrops = ncrops
        # self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        # self.teacher_temp_schedule = np.concatenate((
        #     np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
        #     np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        # ))

    def forward(self, local_output, global_output, flags,cls_input):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        b = flags.shape[0]
        cls = 20
        # student_out = student_output.reshape(self.ncrops, b, -1).permute(1,0,2)
        # teacher_out = teacher_output.reshape(2, b, -1).permute(1,0,2)
        cnt = 0
        global_output = global_output.reshape(2,b,cls).permute(1,0,2) # b 1 20 d
        local_output = local_output.reshape(self.ncrops-2,b,cls).permute(1,0,2) # b 2 20
        total_loss = torch.tensor(0.0).cuda()
        for i in range(b):
            for global_idx in range(2):
                total_loss += F.multilabel_soft_margin_loss(global_output[i][global_idx],cls_input[i])
                cnt+=1

        for i in range(b):
            cls_label = torch.nonzero(cls_input[i]==1)
            cls_label = cls_label.tolist()
            cls_label = [index[0] for index in cls_label]
            for crop_idx in range(self.ncrops-2):

                temp_flag = flags[i][crop_idx+2]
                if temp_flag[0] == 1:
                #bg
                    # local_cls =local_output[i][crop_idx]
                    # assert torch.all(temp_flag[1:] == 0), "Not all elements are zero."
                    # total_loss += F.multilabel_soft_margin_loss(local_cls,temp_flag[1:].cuda())
                    continue
                else:
                    #uncertain or fg
                    unique_elements = temp_flag.unique(return_counts=False)
                    if len(unique_elements) == 1:
                    #uncertain
                        continue
                            

                    if len(unique_elements) >= 2:
                        temp_flag_rv = temp_flag[1:].cuda()
                        # indices = torch.nonzero(temp_flag_rv == 1)
                        # indices = indices.tolist()  # 转换为普通的Python列表
                        # indices = [index[0] for index in indices]  # 获取具体数字
                        local_cls =local_output[i][crop_idx]
                        total_loss += F.multilabel_soft_margin_loss(local_cls,temp_flag_rv)
                        cnt+=1
                        
        return total_loss / (cnt)



#         return loss
# class CPCLoss(nn.Module):
#     def __init__(self, ncrops=10, temp=1.0,num_cls = 20,num_dim = 1024):
#         super().__init__()
#         self.temp = temp
#         # self.center_momentum = center_momentum
#         self.ncrops = ncrops
#         self.num_cls = num_cls
#         self.num_dim = num_dim
#         self.feature_contrast = torch.zeros(self.num_cls,self.num_dim)
#         # self.register_buffer("center", torch.zeros(1, out_dim))
#         # we apply a warm up for the teacher temperature because
#         # a too high temperature makes the training instable at the beginning
#         # self.teacher_temp_schedule = np.concatenate((
#         #     np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
#         #     np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
#         # ))

#     def forward(self, fmap, cam,cls_label,hig_thre,low_thre,bg_thre):
        
#         b, c, h, w = cam.shape
#         #pseudo_label = torch.zeros((b,h,w))
        
#         cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1,1,h,w])
#         valid_cam = cls_label_rep * cam
#         cam_value, _pseudo_label = torch.topk(valid_cam,k=2,dim=1)
#         #[b 2 448 448]
#         _pseudo_label = _pseudo_label[:, 0, :, :]
#         _pseudo_label += 1
#         _pseudo_label[cam_value[:, 0, :, :]< hig_thre] = 255
#         _pseudo_label[cam_value[:, 0, :, :]< low_thre] = 0
#         _pseudo_label[cam_value[:, 0, :, :]< bg_thre] = 0
#         _pseudo_label[(cam_value[:, 0, :, :]-cam_value[:, 1, :, :]< 0.3)&(cam_value[:, 0, :, :]>hig_thre)] = 255
#         #可能是交界处图片，有两种特征
#         plt.imshow((_pseudo_label[0]).cpu(), cmap='jet', vmin=-2, vmax=20)
#         plt.colorbar()
#         plt.title("cam_feature_refine")
        
#         plt.savefig(f'cam_feature_refine.png')
#         plt.close()
#         fmap = F.interpolate(fmap , size= (h,w), mode='bilinear',align_corners=False)
#         # cam_grad = F.interpolate(cam_grad , size= (h,w), mode='bilinear',align_corners=False)
#         mask = torch.zeros_like(cam.detach())
#         #B 20 H W
#         loss = 0
#         for i in range(b):
#             feature_vector = {}
#             arg = torch.nonzero(cls_label[i]) + 1
#             arg = arg.tolist()
#             for cls in arg:
#                 cls = cls[0]
#                 if torch.all(_pseudo_label[i] != cls):
                    
#                     top_values, top_indices_1d = torch.topk(valid_cam[i,cls-1].view(-1), k=5)  # 获取值最大的三个数及其一维索引
#                     indices = torch.vstack((top_indices_1d // 448, top_indices_1d % 448)).T
#                     row_indices = indices[:, 0]
#                     col_indices = indices[:, 1]
#                     sub_fmap = fmap[i, :, row_indices, col_indices]
#                     feature_vector[cls-1] = (torch.mean(sub_fmap, dim=-1))
#                 else:
#                     indices = torch.nonzero(_pseudo_label[i] == cls)
#                     row_indices = indices[:, 0]
#                     col_indices = indices[:, 1]
#                     # 使用高级索引提取子张量
#                     sub_fmap = fmap[i, :, row_indices, col_indices]
#                     feature_vector[cls-1] = (torch.mean(sub_fmap, dim=-1))
#             # indices_bg = torch.nonzero(_pseudo_label[i] == 0)
#             # bg_row_indices = indices_bg[:, 0]
#             # bg_col_indices = indices_bg[:, 1]
#             # sub_fmap_bg = fmap[i, :, bg_row_indices, bg_col_indices]
#             # feature_bg = torch.mean(sub_fmap_bg, dim=-1)
            
#             feature_single_map = torch.zeros(self.num_cls,self.num_dim)
#             indentity_matrix = torch.zeros(self.num_cls,self.num_cls)
#             feature_list = list(feature_vector.values())
#             feature_stack = torch.stack(feature_list,dim=0)
#             # feature_stack = torch.cat((feature_stack,feature_bg.unsqueeze(0)),dim=0)
#             for cls,feature in feature_vector.items():
#                 #feature = F.normalize(feature,p=2,dim=0)
#                 feature_single_map[cls][:] = feature
#                 indentity_matrix[cls][cls] = 1
#                 expanded_feature = feature.detach().view(self.num_dim, 1, 1).expand(self.num_dim, h, w)
#                 mask[i][cls] = mask[i][cls] = torch.abs(torch.nn.functional.cosine_similarity(expanded_feature, fmap[i].detach(), dim=0))

            
#             feature_cos_smi = torch.clamp(torch.abs(F.cosine_similarity(feature_stack.unsqueeze(1),feature_stack.unsqueeze(0),dim=-1)),min=1e-5,max=1-1e-5)   
#             # feature_indentity_single_map = torch.eye(feature_cos_smi.shape[0],feature_cos_smi.shape[1]).cuda()
#             print(feature_cos_smi)
#             # loss +=  F.binary_cross_entropy(feature_cos_smi,f eature_indentity_single_map)
#             mask_value , mask_indices = torch.max(mask,dim = 1)
#             mask_indices[mask_value < hig_thre] = -1
#             # # # # cam_grad = F.sigmoid(cam_grad)
#             # # # # ce_loss = F.cross_entropy(cam_grad,mask.detach())
#             plt.imshow(mask_indices[i].cpu(),cmap='jet', vmin=-1, vmax=20)
#             plt.colorbar()
#             plt.title("cos-smi_mask")
            
#             plt.savefig('cos-smi_mask.png')
#             plt.close()
#             feature_single_map_normalized = torch.nn.functional.normalize(feature_single_map, dim=1)
#             self.feature_contrast_normalized = torch.nn.functional.normalize(self.feature_contrast, dim=1)

#             # 计算余弦相似度矩阵
#             cos_similarity_matrix = torch.abs(torch.mm(feature_single_map_normalized, self.feature_contrast_normalized.T))
            
#             cos_smi_clamp = torch.clamp(cos_similarity_matrix,min=1e-5,max=1-1e-5)
#             cls_index = []
#             for cls,feature in feature_vector.items():
#                 cls_index.append(cls)       
#             loss += F.binary_cross_entropy(cos_smi_clamp,indentity_matrix.detach())
#             self.feature_contrast[cls_index] = 0.95 * self.feature_contrast[cls_index] + 0.05 * feature_single_map[cls_index].detach()
#             # print(cos_smi)
#             # print(F.cosine_similarity(self.feature_contrast.unsqueeze(1),self.feature_contrast.unsqueeze(0),dim=-1))

        
#         return loss 

#cpc online contrast
# class CPCLoss(nn.Module):
#     def __init__(self, ncrops=10, temp=1.0,num_cls = 20,num_dim = 1024):
#         super().__init__()
#         self.temp = temp
#         # self.center_momentum = center_momentum
#         self.ncrops = ncrops
#         self.num_cls = num_cls
#         self.num_dim = num_dim
#         self.feature_contrast = torch.zeros(self.num_cls,self.num_dim)
#         # self.register_buffer("center", torch.zeros(1, out_dim))
#         # we apply a warm up for the teacher temperature because
#         # a too high temperature makes the training instable at the beginning
#         # self.teacher_temp_schedule = np.concatenate((
#         #     np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
#         #     np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
#         # ))

#     def forward(self, fmap, cam,cls_label,hig_thre,low_thre,bg_thre):
        
#         b, c, h, w = cam.shape
#         #pseudo_label = torch.zeros((b,h,w))
        
#         cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1,1,h,w])
#         valid_cam = cls_label_rep * cam
#         cam_value, _pseudo_label = torch.topk(valid_cam,k=2,dim=1)
#         #[b 2 448 448]
#         _pseudo_label = _pseudo_label[:, 0, :, :]
#         _pseudo_label += 1
#         _pseudo_label[cam_value[:, 0, :, :]< hig_thre] = 255
#         _pseudo_label[cam_value[:, 0, :, :]< low_thre] = 0
#         _pseudo_label[cam_value[:, 0, :, :]< bg_thre] = 0
#         _pseudo_label[(cam_value[:, 0, :, :]-cam_value[:, 1, :, :]< 0.3)&(cam_value[:, 0, :, :]>hig_thre)] = 255
#         #可能是交界处图片，有两种特征
#         # plt.imshow((_pseudo_label[0]).cpu(), cmap='jet', vmin=-2, vmax=20)
#         # plt.colorbar()
#         # plt.title("cam_feature_refine")
        
#         # plt.savefig(f'cam_feature_refine.png')
#         # plt.close()
#         fmap = F.interpolate(fmap , size= (h,w), mode='bilinear',align_corners=False)
#         # cam_grad = F.interpolate(cam_grad , size= (h,w), mode='bilinear',align_corners=False)
#         mask = torch.zeros_like(cam.detach())
        
#         #B 20 H W
#         loss = 0
#         for i in range(b):
#             feature_vector = {}
#             arg = torch.nonzero(cls_label[i]) + 1
#             arg = arg.tolist()
#             mutli_arg = []
#             for cls in arg:
#                 cls = cls[0]
#                 if torch.all(_pseudo_label[i] != cls):
#                     mutli_arg.append(cls-1)
#                     top_values, top_indices_1d = torch.topk(valid_cam[i,cls-1].view(-1), k=5)  # 获取值最大的三个数及其一维索引
#                     indices = torch.vstack((top_indices_1d // 448, top_indices_1d % 448)).T
#                     row_indices = indices[:, 0]
#                     col_indices = indices[:, 1]
#                     sub_fmap = fmap[i, :, row_indices, col_indices]
#                     feature_vector[cls-1] = (torch.mean(sub_fmap, dim=-1))
#                 else:
#                     indices = torch.nonzero(_pseudo_label[i] == cls)
#                     row_indices = indices[:, 0]
#                     col_indices = indices[:, 1]
#                     # 使用高级索引提取子张量
#                     sub_fmap = fmap[i, :, row_indices, col_indices]
#                     feature_vector[cls-1] = (torch.mean(sub_fmap, dim=-1))
#             # indices_bg = torch.nonzero(_pseudo_label[i] == 0)
#             # bg_row_indices = indices_bg[:, 0]
#             # bg_col_indices = indices_bg[:, 1]
#             # sub_fmap_bg = fmap[i, :, bg_row_indices, bg_col_indices]
#             # feature_bg = torch.mean(sub_fmap_bg, dim=-1)
            
#             feature_single_map = torch.zeros(self.num_cls,self.num_dim)
#             indentity_matrix = torch.zeros(self.num_cls,self.num_cls)
#             # feature_list = list(feature_vector.values())
#             # feature_stack = torch.stack(feature_list,dim=0)
#             # feature_stack = torch.cat((feature_stack,feature_bg.unsqueeze(0)),dim=0)
#             for cls,feature in feature_vector.items():
#                 #feature = F.normalize(feature,p=2,dim=0)
#                 feature_single_map[cls][:] = feature
#                 indentity_matrix[cls][cls] = 1
#                 # expanded_feature = self.feature_contrast[cls].detach().view(self.num_dim, 1, 1).expand(self.num_dim, h, w).cuda()
#                 # mask[i][cls] = mask[i][cls] = torch.abs(torch.nn.functional.cosine_similarity(expanded_feature, fmap[i].detach(), dim=0))

            
#             # feature_cos_smi = torch.clamp(torch.abs(F.cosine_similarity(feature_stack.unsqueeze(1),feature_stack.unsqueeze(0),dim=-1)),min=1e-5,max=1-1e-5)   
#             # feature_indentity_single_map = torch.eye(feature_cos_smi.shape[0],feature_cos_smi.shape[1]).cuda()
#             # print(feature_cos_smi)
#             # loss += 0.5 * F.binary_cross_entropy(feature_cos_smi,feature_indentity_single_map)
#             # mask_value , mask_indices = torch.max(mask,dim = 1)
#             # mask_indices[mask_value < hig_thre] = -1
#             # # # # # cam_grad = F.sigmoid(cam_grad)
#             # # # # # ce_loss = F.cross_entropy(cam_grad,mask.detach())
#             # plt.imshow(mask_indices[i].cpu(),cmap='jet', vmin=-1, vmax=20)
#             # plt.colorbar()
#             # plt.title("cos-smi_mask")
            
#             # plt.savefig('cos-smi_mask.png')
#             # plt.close()
#             feature_single_map_normalized = torch.nn.functional.normalize(feature_single_map, dim=1)
#             self.feature_contrast_normalized = torch.nn.functional.normalize(self.feature_contrast, dim=1)

#             # 计算余弦相似度矩阵
#             cos_similarity_matrix = torch.abs(torch.mm(feature_single_map_normalized, self.feature_contrast_normalized.T))
#             # print(cos_similarity_matrix)
#             cos_smi_clamp = torch.clamp(cos_similarity_matrix,min=1e-5,max=1-1e-5)
#             cls_index = []
#             for cls,feature in feature_vector.items():
#                 if cls not in mutli_arg:
#                     cls_index.append(cls)       
#             loss += F.binary_cross_entropy(cos_smi_clamp,indentity_matrix.detach())
#             self.feature_contrast[cls_index] = 0.95 * self.feature_contrast[cls_index] + 0.05 * feature_single_map[cls_index].detach()
#             # print(cos_smi)
#             print(F.cosine_similarity(self.feature_contrast.unsqueeze(1),self.feature_contrast.unsqueeze(0),dim=-1))

        
#         return loss 


class CPCLoss(nn.Module):   #softmax + 阈值门版本
    def __init__(self, ncrops=10, temp=1.0,num_cls = 20,num_dim = 1024):
        super().__init__()
        self.temp = temp
        # self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.num_cls = num_cls
        self.num_dim = num_dim
        self.feature_contrast = torch.zeros(self.num_cls,self.num_dim)
        self.proj_classifier = nn.Conv2d(in_channels=self.num_dim, out_channels=self.num_cls, kernel_size=1, bias=False,)
        # self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        # self.teacher_temp_schedule = np.concatenate((
        #     np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
        #     np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        # ))
    
    def get_param_groups(self):

        return self.proj_classifier.weight
    

    def forward(self, fmap, cam,cls_label,hig_thre,low_thre,bg_thre):
        self.proj_classifier.cuda()
        b, c, h, w = cam.shape
        #pseudo_label = torch.zeros((b,h,w))
        
        cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1,1,h,w])
        valid_cam = cls_label_rep * cam
        cam_value, _pseudo_label = torch.topk(valid_cam,k=2,dim=1)
        #[b 2 448 448]
        _pseudo_label = _pseudo_label[:, 0, :, :]
        _pseudo_label += 1
        _pseudo_label[cam_value[:, 0, :, :]< hig_thre] = 255
        _pseudo_label[cam_value[:, 0, :, :]< low_thre] = 0
        _pseudo_label[cam_value[:, 0, :, :]< bg_thre] = 0
        _pseudo_label[(cam_value[:, 0, :, :]-cam_value[:, 1, :, :]< 0.3)&(cam_value[:, 0, :, :]>hig_thre)] = 255
        #可能是交界处图片，有两种特征
        # plt.imshow((_pseudo_label[0]).cpu(), cmap='jet', vmin=-2, vmax=20)
        # plt.colorbar()
        # plt.title("cam_feature_refine")
        
        # plt.savefig(f'cam_feature_refine.png')
        # plt.close()
        fmap = F.interpolate(fmap , size= (h,w), mode='bilinear',align_corners=False)
        # cam_grad = F.interpolate(cam_grad , size= (h,w), mode='bilinear',align_corners=False)
        # mask = torch.zeros_like(cam.detach())
        # fmap_cls = self.proj_classifier(fmap.detach())
        # fmap_cls = F.softmax(fmap_cls,dim=-1)
        # fmap_bg = torch.zeros([b,self.num_cls+1,h,w])
        # fmap_bg[:,1:] = fmap_cls
        # fmap_bg[:,0] = 0.25
        # fmap_cls = torch.argmax(fmap_bg,dim=1)
        # plt.imshow((fmap_cls[0]).cpu(), cmap='jet', vmin=-2, vmax=20)
        # plt.colorbar()
        # plt.title("cam_feature_refine")
        
        # plt.savefig(f'cam_feature_refine.png')
        # plt.close()
        
        #B 20 H W
        loss_ccf = 0
        loss_clsifier = 0
        for i in range(b):
            feature_vector = {}
            arg = torch.nonzero(cls_label[i]) + 1
            arg = arg.tolist()
            mutli_arg = []
            for cls in arg:
                cls = cls[0]
                if torch.all(_pseudo_label[i] != cls):
                    mutli_arg.append(cls-1)
                    top_values, top_indices_1d = torch.topk(valid_cam[i,cls-1].view(-1), k=25)  # 获取值最大的三个数及其一维索引
                    indices = torch.vstack((top_indices_1d // 448, top_indices_1d % 448)).T
                    row_indices = indices[:, 0]
                    col_indices = indices[:, 1]
                    sub_fmap = fmap[i, :, row_indices, col_indices]
                    feature_vector[cls-1] = (torch.mean(sub_fmap, dim=-1))
                else:
                    indices = torch.nonzero(_pseudo_label[i] == cls)
                    row_indices = indices[:, 0]
                    col_indices = indices[:, 1]
                    # 使用高级索引提取子张量
                    sub_fmap = fmap[i, :, row_indices, col_indices]
                    feature_vector[cls-1] = (torch.mean(sub_fmap, dim=-1))
            # indices_bg = torch.nonzero(_pseudo_label[i] == 0)
            # bg_row_indices = indices_bg[:, 0]
            # bg_col_indices = indices_bg[:, 1]
            # sub_fmap_bg = fmap[i, :, bg_row_indices, bg_col_indices]
            # feature_bg = torch.mean(sub_fmap_bg, dim=-1)
            
            feature_single_map = torch.zeros(self.num_cls,self.num_dim)
            indentity_matrix = torch.zeros(self.num_cls,self.num_cls)
            # feature_list = list(feature_vector.values())
            # feature_stack = torch.stack(feature_list,dim=0)
            # feature_stack = torch.cat((feature_stack,feature_bg.unsqueeze(0)),dim=0)
            for cls,feature in feature_vector.items():
                #feature = F.normalize(feature,p=2,dim=0)
                feature_single_map[cls][:] = feature
                indentity_matrix[cls][cls] = 1

                
                # expanded_feature = self.feature_contrast[cls].detach().view(self.num_dim, 1, 1).expand(self.num_dim, h, w).cuda()
                # mask[i][cls] = mask[i][cls] = torch.abs(torch.nn.functional.cosine_similarity(expanded_feature, fmap[i].detach(), dim=0))


            # feature_cos_smi = torch.clamp(torch.abs(F.cosine_similarity(feature_stack.unsqueeze(1),feature_stack.unsqueeze(0),dim=-1)),min=1e-5,max=1-1e-5)   
            # feature_indentity_single_map = torch.eye(feature_cos_smi.shape[0],feature_cos_smi.shape[1]).cuda()
            # print(feature_cos_smi)
            # loss += 0.5 * F.binary_cross_entropy(feature_cos_smi,feature_indentity_single_map)
            # mask_value , mask_indices = torch.max(mask,dim = 1)
            # mask_indices[mask_value < hig_thre] = -1
            # # # # # cam_grad = F.sigmoid(cam_grad)
            # # # # # ce_loss = F.cross_entropy(cam_grad,mask.detach())
            # plt.imshow(mask_indices[i].cpu(),cmap='jet', vmin=-1, vmax=20)
            # plt.colorbar()
            # plt.title("cos-smi_mask")
            
            # plt.savefig('cos-smi_mask.png')
            # plt.close()
            feature_single_map_normalized = torch.nn.functional.normalize(feature_single_map, dim=1)
            self.feature_contrast_normalized = torch.nn.functional.normalize(self.feature_contrast, dim=1)

            # 计算余弦相似度矩阵
            cos_similarity_matrix = torch.abs(torch.mm(feature_single_map_normalized, self.feature_contrast_normalized.T))
            # print(cos_similarity_matrix)
            cos_smi_clamp = torch.clamp(cos_similarity_matrix,min=1e-5,max=1-1e-5)
            cls_index = []
            
            #阈值门，当和其他cls的太相似时，不加入到prototype中
            for cls,feature in feature_vector.items():
                # if cls not in mutli_arg:
                row = cos_smi_clamp[cls]
                row_without_cls = torch.cat([row[:cls], row[cls+1:]])
                if torch.all(row_without_cls < 0.6):
                    cls_index.append(cls)
                    pred_label = self.proj_classifier(feature.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
                    gd_label = torch.zeros(self.num_cls).cuda()
                    gd_label[cls] = 1
                    loss_clsifier += F.binary_cross_entropy(F.softmax(pred_label,dim=0),gd_label)     
                # elif torch.all(row_without_cls < row[cls]):
                #     pred_label = F.conv2d((feature.unsqueeze(-1).unsqueeze(-1)),self.proj_classifier.weight).squeeze(-1).squeeze(-1)
                #     gd_label = torch.zeros(self.num_cls).cuda()
                #     gd_label[cls] = 1
                #     loss_clsifier += F.binary_cross_entropy(F.softmax(pred_label,dim=0),gd_label)  
                # elif torch.any(row_without_cls > row[cls]):
                #     true_cls = torch.argmax(row)
                #     pred_label = F.conv2d((feature.unsqueeze(-1).unsqueeze(-1)),self.proj_classifier.weight).squeeze(-1).squeeze(-1)
                #     gd_label = torch.zeros(self.num_cls).cuda()
                #     gd_label[true_cls] = 1
                #     loss_clsifier += F.binary_cross_entropy(F.softmax(pred_label,dim=0),gd_label)  
            loss_ccf += F.binary_cross_entropy(cos_smi_clamp,indentity_matrix.detach())
            self.feature_contrast[cls_index] = 0.95 * self.feature_contrast[cls_index] + 0.05 * feature_single_map[cls_index].detach()
            # print(cos_smi)
            
            
            # pred_label = self.proj_classifier(self.feature_contrast.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
            # pred_label = F.softmax(pred_label,dim=-1)
            # gd_label = torch.eye(pred_label.shape[0],pred_label.shape[1]).cuda()
            # loss_clsifier = F.binary_cross_entropy(pred_label,gd_label)
            if len(cls_index):
                loss_clsifier /= len(cls_index)
            # print(loss_ccf,loss_clsifier)
        
        return loss_ccf+loss_clsifier 


# class CPCLoss(nn.Module):   #softmax + 阈值门 + backgroud版本
#     def __init__(self, ncrops=10, temp=1.0,num_cls = 20,num_dim = 1024):
#         super().__init__()
#         self.temp = temp
#         # self.center_momentum = center_momentum
#         self.ncrops = ncrops
#         self.num_cls = num_cls + 1
#         self.num_dim = num_dim
#         self.feature_contrast = torch.zeros(self.num_cls,self.num_dim)
#         self.proj_classifier = nn.Conv2d(in_channels=self.num_dim, out_channels=self.num_cls, kernel_size=1, bias=False,)
#         # self.register_buffer("center", torch.zeros(1, out_dim))
#         # we apply a warm up for the teacher temperature because
#         # a too high temperature makes the training instable at the beginning
#         # self.teacher_temp_schedule = np.concatenate((
#         #     np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
#         #     np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
#         # ))
    
#     def get_param_groups(self):

#         return self.proj_classifier.weight
    

#     def forward(self, fmap, cam,cls_label,hig_thre,low_thre,bg_thre):
#         self.proj_classifier.cuda()
#         b, c, h, w = cam.shape
#         #pseudo_label = torch.zeros((b,h,w))
        
#         cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1,1,h,w])
#         valid_cam = cls_label_rep * cam
#         cam_value, _pseudo_label = torch.topk(valid_cam,k=2,dim=1)
#         #[b 2 448 448]
#         _pseudo_label = _pseudo_label[:, 0, :, :]
#         _pseudo_label += 1
#         _pseudo_label[cam_value[:, 0, :, :]< hig_thre] = 255
#         _pseudo_label[cam_value[:, 0, :, :]< low_thre] = 0
#         # _pseudo_label[cam_value[:, 0, :, :]< bg_thre] = 0
#         _pseudo_label[(cam_value[:, 0, :, :]-cam_value[:, 1, :, :]< 0.3)&(cam_value[:, 0, :, :]>hig_thre)] = 255
#         #可能是交界处图片，有两种特征
#         # plt.imshow((_pseudo_label[0]).cpu(), cmap='jet', vmin=-2, vmax=20)
#         # plt.colorbar()
#         # plt.title("cam_feature_refine")
        
#         # plt.savefig(f'cam_feature_refine.png')
#         # plt.close()
#         fmap = F.interpolate(fmap , size= (h,w), mode='bilinear',align_corners=False)
#         # cam_grad = F.interpolate(cam_grad , size= (h,w), mode='bilinear',align_corners=False)
#         # mask = torch.zeros_like(cam.detach())
        
#         #B 20 H W
#         loss_ccf = 0
#         loss_clsifier = 0
#         for i in range(b):
#             feature_vector = {}
#             arg = torch.nonzero(cls_label[i]) + 1
#             arg = arg.tolist()
#             arg.append([0])
#             mutli_arg = []
#             for cls in arg:
#                 cls = cls[0]
#                 if torch.all(_pseudo_label[i] != cls):
#                     mutli_arg.append(cls)
#                     top_values, top_indices_1d = torch.topk(valid_cam[i,cls-1].view(-1), k=5)  # 获取值最大的三个数及其一维索引
#                     indices = torch.vstack((top_indices_1d // 448, top_indices_1d % 448)).T
#                     row_indices = indices[:, 0]
#                     col_indices = indices[:, 1]
#                     sub_fmap = fmap[i, :, row_indices, col_indices]
#                     feature_vector[cls] = (torch.mean(sub_fmap, dim=-1))
#                 else:
#                     indices = torch.nonzero(_pseudo_label[i] == cls)
#                     row_indices = indices[:, 0]
#                     col_indices = indices[:, 1]
#                     # 使用高级索引提取子张量
#                     sub_fmap = fmap[i, :, row_indices, col_indices]
#                     feature_vector[cls] = (torch.mean(sub_fmap, dim=-1))
#             # indices_bg = torch.nonzero(_pseudo_label[i] == 0)
#             # bg_row_indices = indices_bg[:, 0]
#             # bg_col_indices = indices_bg[:, 1]
#             # sub_fmap_bg = fmap[i, :, bg_row_indices, bg_col_indices]
#             # feature_bg = torch.mean(sub_fmap_bg, dim=-1)
            
#             feature_single_map = torch.zeros(self.num_cls,self.num_dim)
#             indentity_matrix = torch.zeros(self.num_cls,self.num_cls)
#             # feature_list = list(feature_vector.values())
#             # feature_stack = torch.stack(feature_list,dim=0)
#             # feature_stack = torch.cat((feature_stack,feature_bg.unsqueeze(0)),dim=0)
#             for cls,feature in feature_vector.items():
#                 #feature = F.normalize(feature,p=2,dim=0)
#                 feature_single_map[cls][:] = feature
#                 indentity_matrix[cls][cls] = 1

                
#                 # expanded_feature = self.feature_contrast[cls].detach().view(self.num_dim, 1, 1).expand(self.num_dim, h, w).cuda()
#                 # mask[i][cls] = mask[i][cls] = torch.abs(torch.nn.functional.cosine_similarity(expanded_feature, fmap[i].detach(), dim=0))


#             # feature_cos_smi = torch.clamp(torch.abs(F.cosine_similarity(feature_stack.unsqueeze(1),feature_stack.unsqueeze(0),dim=-1)),min=1e-5,max=1-1e-5)   
#             # feature_indentity_single_map = torch.eye(feature_cos_smi.shape[0],feature_cos_smi.shape[1]).cuda()
#             # print(feature_cos_smi)
#             # loss += 0.5 * F.binary_cross_entropy(feature_cos_smi,feature_indentity_single_map)
#             # mask_value , mask_indices = torch.max(mask,dim = 1)
#             # mask_indices[mask_value < hig_thre] = -1
#             # # # # # cam_grad = F.sigmoid(cam_grad)
#             # # # # # ce_loss = F.cross_entropy(cam_grad,mask.detach())
#             # plt.imshow(mask_indices[i].cpu(),cmap='jet', vmin=-1, vmax=20)
#             # plt.colorbar()
#             # plt.title("cos-smi_mask")
            
#             # plt.savefig('cos-smi_mask.png')
#             # plt.close()
#             feature_single_map_normalized = torch.nn.functional.normalize(feature_single_map, dim=1)
#             self.feature_contrast_normalized = torch.nn.functional.normalize(self.feature_contrast, dim=1)

#             # 计算余弦相似度矩阵
#             cos_similarity_matrix = torch.abs(torch.mm(feature_single_map_normalized, self.feature_contrast_normalized.T))
#             # print(cos_similarity_matrix)
#             cos_smi_clamp = torch.clamp(cos_similarity_matrix,min=1e-5,max=1-1e-5)
#             cls_index = []
            
#             #阈值门，当和其他cls的太相似时，不加入到prototype中
#             for cls,feature in feature_vector.items():
#                 # if cls not in mutli_arg:
#                 row = cos_smi_clamp[cls]
#                 row_without_cls = torch.cat([row[:cls], row[cls+1:]])
#                 if torch.all(row_without_cls < 0.8):
#                     cls_index.append(cls)
#                     pred_label = self.proj_classifier(feature.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
#                     gd_label = torch.zeros(self.num_cls).cuda()
#                     gd_label[cls] = 1
#                     loss_clsifier += F.binary_cross_entropy(F.softmax(pred_label,dim=0),gd_label)       
#             loss_ccf += F.binary_cross_entropy(cos_smi_clamp,indentity_matrix.detach())
#             self.feature_contrast[cls_index] = 0.95 * self.feature_contrast[cls_index] + 0.05 * feature_single_map[cls_index].detach()
#             # print(cos_smi)
            
            
#             # pred_label = self.proj_classifier(self.feature_contrast.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
#             # pred_label = F.softmax(pred_label,dim=-1)
#             # gd_label = torch.eye(pred_label.shape[0],pred_label.shape[1]).cuda()
#             # loss_clsifier = F.binary_cross_entropy(pred_label,gd_label)
            
        
#             if len(cls_index):
#                 loss_clsifier /= len(cls_index)
#             print(loss_ccf,loss_clsifier)
        
#         return loss_ccf+loss_clsifier 


class DenseEnergyLossFunction(Function):
    
    @staticmethod
    def forward(ctx, images, segmentations, sigma_rgb, sigma_xy, ROIs, unlabel_region):
        ctx.save_for_backward(segmentations)
        ctx.N, ctx.K, ctx.H, ctx.W = segmentations.shape
        Gate = ROIs.clone().to(ROIs.device)

        ROIs = ROIs.unsqueeze_(1).repeat(1,ctx.K,1,1)

        seg_max = torch.max(segmentations, dim=1)[0]
        Gate = Gate - seg_max
        Gate[unlabel_region] = 1
        Gate[Gate < 0] = 0
        Gate = Gate.unsqueeze_(1).repeat(1, ctx.K, 1, 1)

        segmentations = torch.mul(segmentations.cuda(), ROIs.cuda())
        ctx.ROIs = ROIs
        
        densecrf_loss = 0.0
        images = images.cpu().numpy().flatten()
        segmentations = segmentations.cpu().numpy().flatten()
        AS = np.zeros(segmentations.shape, dtype=np.float32)
        bilateralfilter_batch(images, segmentations, AS, ctx.N, ctx.K, ctx.H, ctx.W, sigma_rgb, sigma_xy)
        Gate = Gate.cpu().numpy().flatten()
        AS = np.multiply(AS, Gate)
        densecrf_loss -= np.dot(segmentations, AS)
    
        # averaged by the number of images
        densecrf_loss /= ctx.N
        
        ctx.AS = np.reshape(AS, (ctx.N, ctx.K, ctx.H, ctx.W))
        return Variable(torch.tensor([densecrf_loss]), requires_grad=True)
        
    @staticmethod
    def backward(ctx, grad_output):
        grad_segmentation = -2*grad_output*torch.from_numpy(ctx.AS)/ctx.N
        grad_segmentation = grad_segmentation.cuda()
        grad_segmentation = torch.mul(grad_segmentation, ctx.ROIs.cuda())
        return None, grad_segmentation, None, None, None, None
    

class DenseEnergyLoss(nn.Module):
    def __init__(self, weight, sigma_rgb, sigma_xy, scale_factor):
        super(DenseEnergyLoss, self).__init__()
        self.weight = weight
        self.sigma_rgb = sigma_rgb
        self.sigma_xy = sigma_xy
        self.scale_factor = scale_factor
    
    def forward(self, images, segmentations, ROIs, seg_label):
        """ scale imag by scale_factor """
        scaled_images = F.interpolate(images,scale_factor=self.scale_factor, recompute_scale_factor=True) 
        scaled_segs = F.interpolate(segmentations,scale_factor=self.scale_factor,mode='bilinear',align_corners=False, recompute_scale_factor=True)
        scaled_ROIs = F.interpolate(ROIs.unsqueeze(1),scale_factor=self.scale_factor, recompute_scale_factor=True).squeeze(1)
        scaled_seg_label = F.interpolate(seg_label,scale_factor=self.scale_factor,mode='nearest', recompute_scale_factor=True)
        unlabel_region = (scaled_seg_label.long() == 255).squeeze(1)

        return self.weight*DenseEnergyLossFunction.apply(
                scaled_images, scaled_segs, self.sigma_rgb, self.sigma_xy*self.scale_factor, scaled_ROIs, unlabel_region)
    
    def extra_repr(self):
        return 'sigma_rgb={}, sigma_xy={}, weight={}, scale_factor={}'.format(
            self.sigma_rgb, self.sigma_xy, self.weight, self.scale_factor
        )
        

class ContrastLoss(nn.Module):   
    def __init__(self,temp=1.0,num_cls = 20, buffer_lenth = 768 ,buffer_dim = 512):
        super().__init__()
        self.temp = temp
        # self.center_momentum = center_momentum

        self.buffer_cls = num_cls * 2
        self.buffer_dim = buffer_dim
        self.buffer_lenth = buffer_lenth
        self.feature_contrast = torch.zeros(self.buffer_cls,self.buffer_lenth ,self.buffer_dim)
        self.is_used_tag = torch.zeros(self.buffer_cls,self.buffer_lenth)


    def forward(self, feature_contrast, cls_label):
        #                   [0,20]        [0,19]
        b, _ = cls_label.shape
        tempreture = 1.0
        contrastive_loss = 0
        for i in range(b):
            bg_feature = feature_contrast[i][0]
            feature_fg_contrast = feature_contrast[i][1:]
            cls_index = torch.where(cls_label[i]==1)
            for cls in cls_index[0]:
                cls = cls.item()
                cls_feature = feature_fg_contrast[cls]
                
                buffer_index = 2*cls + 1 #buffer_current_index
                #positive
                
                
                self.is_used_tag[buffer_index][0] = 1 
                self.is_used_tag[buffer_index-1][0] = 1
               
                self.feature_contrast[buffer_index, 0] = cls_feature.detach()
                self.feature_contrast[buffer_index-1, 0] = bg_feature.detach()

                
                
                positive_select_tag = self.is_used_tag[buffer_index] == 1
                
                positive_group = [torch.mean(self.feature_contrast[buffer_index][positive_select_tag],dim = 0)]
                
                negetive_in_buffer = []
                #negetive 现在是所有与cls不同的不管bg or fg都为negative
                for x in range(self.buffer_cls):
                    if x!=buffer_index:
                        negative_select_tag = self.is_used_tag[x] == 1
                        if sum(negative_select_tag)!=0:
                            negetive_in_buffer.append(torch.mean(self.feature_contrast[x][negative_select_tag],dim=0)) 

                    
                
                # negetive_in_pic = [feature_fg_contrast[x].detach() for x in cls_index[0] if x.item()!=cls] + [bg_feature.detach()]
                
                
                negative_group = negetive_in_buffer #+ negetive_in_pic
                
                
                #contrast loss and infoNCE
                # 计算正类pair的相似度
                # positive_similarity = torch.cosine_similarity(cls_feature, torch.stack(positive_group))

                # # 计算负类pair的相似度

                # negative_similarity = torch.cat([torch.cosine_similarity(cls_feature, negative_pair.unsqueeze(0)) for negative_pair in negative_group])
                
                positive_similarity = torch.dot(cls_feature, torch.stack(positive_group).squeeze(0)).unsqueeze(0)
                negative_similarity = torch.stack([torch.dot(cls_feature, negative_pair) for negative_pair in negative_group])
                
                n_positive = len(positive_group)
                n_negative = len(negative_group)

                total_samples = n_positive + n_negative
                labels = torch.zeros(total_samples)
                labels[:n_positive] = 1

                logits = torch.cat([positive_similarity, negative_similarity])

                div = torch.sum(torch.exp(logits/tempreture))
                head = torch.exp(positive_similarity/tempreture)
                prob_logits = torch.div(head , div)
                infoNCE_loss = -(torch.log(prob_logits))
                infoNCE_loss = infoNCE_loss / len(cls_index[0])
               
                # 构建InfoNCE loss
                # targets = torch.cat([torch.ones_like(positive_similarity), torch.zeros_like(negative_similarity)])
                # logits = torch.abs(torch.cat([positive_similarity, negative_similarity]))
                # loss = F.binary_cross_entropy(logits, targets)
                contrastive_loss  = contrastive_loss + infoNCE_loss
                #logits?
                self.is_used_tag[buffer_index] = torch.roll(self.is_used_tag[buffer_index], shifts=1, dims=0)
                self.is_used_tag[buffer_index-1] = torch.roll(self.is_used_tag[buffer_index-1], shifts=1, dims=0)
                self.feature_contrast[buffer_index] = torch.roll(self.feature_contrast[buffer_index], shifts=1, dims=0)
                self.feature_contrast[buffer_index-1] = torch.roll(self.feature_contrast[buffer_index-1], shifts=1, dims=0)
                

                                


        return contrastive_loss.cuda() / b