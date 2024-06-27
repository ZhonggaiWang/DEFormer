import pdb
import torch
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import random
#448尺度下
def cam_to_label(cam, cls_label, img_box=None, bkg_thre=None, high_thre=None, low_thre=None, ignore_mid=False, ignore_index=None):
    b, c, h, w = cam.shape
    #pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1,1,h,w])
    valid_cam = cls_label_rep * cam
    cam_value, _pseudo_label = valid_cam.max(dim=1, keepdim=False)
    _pseudo_label += 1
    _pseudo_label[cam_value<=bkg_thre] = 0

    if img_box is None:
        return _pseudo_label

    if ignore_mid:
        _pseudo_label[cam_value<=high_thre] = ignore_index
        _pseudo_label[cam_value<=low_thre] = 0
    pseudo_label = torch.ones_like(_pseudo_label) * ignore_index

    for idx, coord in enumerate(img_box):
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = _pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]]

    return valid_cam, pseudo_label

def random_with_probability(p):
    if random.randrange(0, 100) < p * 100:
        return 0
    else:
        return 255

# def cam_to_label_resized(cam, cls_label, img_box=None, bkg_thre=None, high_thre=None, low_thre=None, ignore_mid=False, ignore_index=None,printornot = False,clip = False):
#     b, c, h, w = cam.shape
#     #pseudo_label = torch.zeros((b,h,w))
#     cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1,1,h,w])
#     valid_cam = cls_label_rep * cam
#     cam_value, _pseudo_label = valid_cam.max(dim=1, keepdim=False)  #pseudo_label 是对应得索引
#     _pseudo_label += 1
#     _pseudo_label[cam_value<=bkg_thre] = 0
#             #b h w

#     #cam value [b 448 448] _pseudo_label [b 448 448]每个位置是索引
#     if img_box is None:
#         return _pseudo_label

#     if ignore_mid: #忽略中间部分high_thre是确定前景类别 low_thre得是背景 中间得是忽略区域
#         if clip:
#             random_mask = torch.rand_like(cam_value)
#             _pseudo_label[cam_value<=high_thre] = torch.where(random_mask[cam_value <= high_thre] < 0.7, 0, 255)
#         else:
#             _pseudo_label[cam_value<=high_thre] = ignore_index
#         _pseudo_label[cam_value<=low_thre] = 0

#     if printornot:            
#         plt.imshow((_pseudo_label[0]).cpu(), cmap='jet', vmin=-2, vmax=20)
#         plt.colorbar()
#         plt.title("aux_mask")
            
#         plt.savefig(f'aux_mask.png')
#         plt.close()

#     return valid_cam, _pseudo_label

# def cam_patch_contrast_loss(cam, cls_label, img_box=None, bkg_thre=None, high_thre=None, low_thre=None, ignore_mid=False, ignore_index=None):
#     #way1
#     b, c, h, w = cam.shape
#     #pseudo_label = torch.zeros((b,h,w))
#     cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1,1,h,w])
#     valid_cam = cls_label_rep * cam
#     cam_value, _pseudo_label = valid_cam.max(dim=1, keepdim=False)  #pseudo_label 是对应得索引
    
#     _pseudo_label[cam_value>=(high_thre)] = 1
#     _pseudo_label[cam_value<(high_thre)] = 0
#     count_fg = torch.sum(_pseudo_label,dim=(1,2))
#     mask = (_pseudo_label == 1)
    
#     distances = torch.zeros_like(cam[:, 0])

#     values, indices = torch.topk(valid_cam, k=2,dim=1)
#     print(valid_cam[0][3])
#     max_values = values[:, 0]
#     second_max_values = values[:, 1]
#     distances[mask] =(max_values[mask] - second_max_values[mask] )**2
#     sum_distances = -torch.mean(distances[mask])

#     loss = sum_distances



#     #就是裁剪框外边的忽略掉，只看里边的
#     return loss

# def crop_from_roi_posi(images,cls_label = None, roi_mask=None, crop_num=8, crop_size=96):  2023 12 19版本
#     #image 2 3 448 448
#     crops = []
    
#     b, c, h, w = images.shape
    
#         # 2 8 3 96 96 
#     num_class = 21
#     flags = torch.zeros(size=(b, crop_num+2,num_class)).to(images.device)#b crop_num
#     flags[:,:2,1:] = cls_label.unsqueeze(1).repeat(1, 2, 1)

#     for i1 in range(b):

#         fg_pixel_number = 0
#         positive_ele = 0
#         #adaptive cut
#         positive_pixel = {}
#         may_fg_pixel_number = 0
#         mask_all_element, mask_all_counts = roi_mask[i1].unique(return_counts=True)
#         for element, count in zip(mask_all_element, mask_all_counts):
#             if element.item() >=-1:
#                 may_fg_pixel_number += count.item()
#                 if element.item()>=0:
#                     fg_pixel_number +=  count.item()
#                     positive_pixel[element.item()] = count.item()
#                     positive_ele +=1
#         if positive_ele:
#             may_fg_pixel_number /= positive_ele
#             fg_pixel_number /= positive_ele
#             crop_size = math.sqrt(may_fg_pixel_number)
        
#         crop_size = (math.ceil(crop_size / 16)) * 16

#         crop_size = max(crop_size,48)
#         crop_size = min(crop_size,192)

#         temp_crops = torch.zeros(size=(b, crop_num, c, crop_size, crop_size)).to(images.device)
        
#         plt.imshow((roi_mask[i1]).cpu(), cmap='jet', vmin=-2, vmax=20)
#         plt.colorbar()
#         plt.title("full_mask")
        
#         plt.savefig(f'full_mask.png')
#         plt.close()

#         margin = 0

#         k = crop_size//2
#         threshold = crop_size**2 /4
#         # roi_index = (roi_mask[i1, margin:(h-margin), margin:(w-margin)] == -1).nonzero() #neatrual and negtive
#         # #不取margin
#         # # test = roi_mask[i1, margin:(h-margin), margin:(w-margin)]
#         # # Flage = test[3,5] == roi_mask[i1,margin+3,margin+5]
#         # # nonzero() 函数获取掩码中为 True 的元素的索引
#         # if roi_index.shape[0] < crop_num:
#         #     roi_index = (roi_mask[i1, margin:(h-margin), margin:(w-margin)] >= -2).nonzero() ## if NULL then random crop
#         # rand_index = torch.randperm(roi_index.shape[0])
#         # #torch.randperm() 函数返回一个随机排列的整数序列，范围从 0 到 roi_index.shape[0] - 1。
#         # crop_index = roi_index[rand_index[:crop_num], :]
#         # 获取ROI的形状
#         roi_idx = roi_mask[i1, margin:(h-margin), margin:(w-margin)]  # 获取ROI的子张量
#         padding = (k, k, k, k)  # (left, right, top, bottom)
#         padded_roi_mask = F.pad(roi_mask, padding, value=-2)
#         padded_images = F.pad(images, padding, value=0)
#         # 获取中心点的索引

        
#         # if len(valid_indices)>=crop_num:    #     valid_indices = torch.stack(valid_indices,dim=0)
        
# # 构造卷积核，用于计算邻域内的正类像素数量
#         class_potential_list = []
#         kernel = torch.ones((2*k+1, 2*k+1), dtype=torch.float32).cuda()
#         for class_mask_idx ,class_counts in positive_pixel.items():
#             # torch.set_printoptions(profile='full')
#             center_indices = torch.nonzero(roi_idx == class_mask_idx).cpu() 
#             neighbor_counts = F.conv2d((roi_mask[i1]==-1).float().unsqueeze(0).unsqueeze(0).float(), kernel.unsqueeze(0).unsqueeze(0), padding=k).squeeze(0).squeeze(0).long()
#         # 使用卷积操作计算邻域内的正类像素数量  类不平衡问题！
#         # neighbor_counts = F.conv2d((roi_mask[i1]>=0).float().unsqueeze(0).unsqueeze(0).float(), kernel.unsqueeze(0).unsqueeze(0), padding=0).squeeze(0).squeeze(0).long()
#             neighbor_counts_array = np.array(neighbor_counts.cpu())
#             # 获取指定坐标位置的元素
#             neighbor_counts = neighbor_counts_array[center_indices[:, 0], center_indices[:, 1]]
#             neighbor_counts = torch.tensor(neighbor_counts).reshape(-1)
#             # neighbor_counts = torch.stack(neighbor_counts,dim=0)
#             #类不平衡问题?threshold  这里先用的是平均 选谁出来crop
#             valid_indices = center_indices[neighbor_counts > 0.1 * (may_fg_pixel_number-fg_pixel_number)]
#             valid_indices.cuda()
#             valid_indices += k
#             class_potential_list.append(valid_indices)
        
# # 计算每个类别应该抽取的数量
        
#         # samples_per_class = crop_num // len(class_potential_list) + 1
#         samples_per_class = crop_num + 1

#         # 创建一个空的最终候选点列表
#         final_candidates = []

#         # 遍历类别的候选点列表，检查每个类别的候选点数量是否大于等于 samples_per_class
#         for candidate_list in class_potential_list:
#             if len(candidate_list) >= samples_per_class:
#                 # 对于满足条件的类别，从中随机抽取 samples_per_class 个元素，并将它们添加到最终候选点列表中
#                 samples = random.choices(candidate_list, k=samples_per_class)
#                 final_candidates.extend(samples)
#             elif len(candidate_list):
#                 samples = random.choices(candidate_list, k=samples_per_class)
#                 final_candidates.extend(samples)
#             else:
#                 continue

#         # 打乱最终候选点列表的顺序
#         random.shuffle(final_candidates)

#         if len(final_candidates) >= crop_num:
            
#         # 截取最终候选点列表的前 crop_number 个元素作为最终的候选点
#             final_candidates = final_candidates[:crop_num]
#         # 找到符合条件的中心点索引
#         # 随机选取crop_num个中心点
#             crop_index = torch.stack(final_candidates)
#         #随机选取策略

#         #随机选取策略

#         else:
            
#             #不取margin
#             # test = roi_mask[i1, margin:(h-margin), margin:(w-margin)]
#             # Flage = test[3,5] == roi_mask[i1,margin+3,margin+5]
            
#             roi_index = (roi_mask[i1, k:(h-k), k:(w-k)] >= -2).nonzero() ## if NULL then random crop
#             rand_index = torch.randperm(roi_index.shape[0])
#             #torch.randperm() 函数返回一个随机排列的整数序列，范围从 0 到 roi_index.shape[0] - 1。
#             crop_index = roi_index[rand_index[:crop_num], :] + torch.full((1, 2), k).cuda()
#             # orgin = roi_index[rand_index[:crop_num], :] 

#         #[(1,2) (3,4) (5,6) ...]
#         for i2 in range(crop_num):
#             h0, w0 = crop_index[i2, 0], crop_index[i2, 1] # centered at (h0, w0)
#             #确定一个点
#             temp_crops[i1, i2, ...] = padded_images[i1, :, (h0-k):(h0+k), (w0-k):(w0+k)]
#             #temp_crops 的形状将是 [b, n, 3, crop_size, crop_size]
#             #...用于省略连续的冒号
#             temp_mask = padded_roi_mask[i1, (h0-k):(h0+k), (w0-k):(w0+k)].cpu()
#             # if temp_mask.sum() / (crop_size*crop_size) <= 0.2:
#             #     ## if ratio of uncertain regions < 0.2 then negative
#             #     flags[i1, i2+2] = 0

#             plt.imshow(temp_mask, cmap='jet', vmin=-2, vmax=1)
#             plt.colorbar()
#             plt.title("temp_mask")
            
#             plt.savefig(f'{i2}temp_mask.png')
#             plt.close()

#             pil_image = TF.to_pil_image(temp_crops[i1][i2])
#             pil_image.save(f'{i2}'+'crop_image.png')



#             unique_elements, counts = temp_mask.unique(return_counts=True)
#             sorted_indices = np.argsort(unique_elements.numpy())
#             sorted_indices = sorted_indices[::-1]
#             contiguous_array = np.copy(sorted_indices)
# # 使用sorted_indices对unique_elements进行排序
#             sorted_unique_elements = unique_elements[torch.from_numpy(contiguous_array)]

#             # 使用sorted_indices获取对应的counts排序
#             sorted_counts = counts[torch.from_numpy(contiguous_array)]
            
#             for element, count in zip(sorted_unique_elements, sorted_counts):
                
#                 if element.item() != -1 and element.item() != -2:  # 忽略元素为-1的情况
#                     ratio = count.item() / positive_pixel[element.item()]
#                     if ratio > 0.2: #标定crop出来的有没有东西
#                         element_idx = element.item()
#                         flags[i1, i2+2,element_idx+1] = 1
                        
#                 if element.item() == -2:
#                     ratio = count.item() / (crop_size * crop_size)
#                     if ratio > 0.70 and torch.all(flags[i1,i2+2] == 0)  :
#                         element_idx = element.item()
#                         if element.item() == -2:
#                             element_idx = element.item() + 1
#                         flags[i1, i2+2,element_idx+1] = 1
#     #list [ tensor.shape 2 1 3 96 96  ]
#     _crops = torch.chunk(temp_crops, chunks=crop_num, dim=1,)
#     #temp_crops 是一个张量，chunks 是要分割的块数，dim 是要在哪个维度上进行分块操作
#     #b num 3 cs cs -> n * [b, 1, 3, crop_size, crop_size]
#     crops = [c[:, 0] for c in _crops]
#     #这里使用了切片操作 [:, 0] 来选择第一个元素。切片操作 [:, 0] 表示选择所有批量维度的元素，并在截取数目维度上选择索引为0的元素。
#     #crops 里面每个元素是[b, 3, crop_size, crop_size]
#     #list [2 3 96 96]
#     return crops, flags.cuda()

def crop_from_roi_posi(images,cls_label = None, roi_mask=None, crop_num=8, crop_size=96):
    #image 2 3 448 448
    crops = []
    
    b, c, h, w = images.shape
    
        # 2 8 3 96 96 
    num_class = 21
    flags = torch.zeros(size=(b, crop_num+2,num_class)).to(images.device)#b crop_num
    flags[:,:2,1:] = cls_label.unsqueeze(1).repeat(1, 2, 1)

    for i1 in range(b):

        fg_pixel_number = 0
        positive_ele = 0
        #adaptive cut
        positive_pixel = {}
        may_fg_pixel_number = 0
        mask_all_element, mask_all_counts = roi_mask[i1].unique(return_counts=True)
        for element, count in zip(mask_all_element, mask_all_counts):
            if element.item() >=-1:
                may_fg_pixel_number += count.item()
                if element.item()>=0:
                    fg_pixel_number +=  count.item()
                    positive_pixel[element.item()] = count.item()
                    positive_ele +=1
        if positive_ele:
            may_fg_pixel_number /= positive_ele
            fg_pixel_number /= positive_ele
            crop_size = math.sqrt(may_fg_pixel_number)
        
        crop_size = (math.ceil(crop_size / 16)) * 16

        crop_size = max(crop_size,48)
        crop_size = min(crop_size,144)

        temp_crops = torch.zeros(size=(b, crop_num, c, crop_size, crop_size)).to(images.device)
        
        # plt.imshow((roi_mask[i1]).cpu(), cmap='jet', vmin=-2, vmax=20)
        # plt.colorbar()
        # plt.title("full_mask")
        
        # plt.savefig(f'full_mask.png')
        # plt.close()

        margin = 0

        k = crop_size//2
        # threshold = crop_size**2 /4
        # roi_index = (roi_mask[i1, margin:(h-margin), margin:(w-margin)] == -1).nonzero() #neatrual and negtive
        # #不取margin
        # # test = roi_mask[i1, margin:(h-margin), margin:(w-margin)]
        # # Flage = test[3,5] == roi_mask[i1,margin+3,margin+5]
        # # nonzero() 函数获取掩码中为 True 的元素的索引
        # if roi_index.shape[0] < crop_num:
        #     roi_index = (roi_mask[i1, margin:(h-margin), margin:(w-margin)] >= -2).nonzero() ## if NULL then random crop
        # rand_index = torch.randperm(roi_index.shape[0])
        # #torch.randperm() 函数返回一个随机排列的整数序列，范围从 0 到 roi_index.shape[0] - 1。
        # crop_index = roi_index[rand_index[:crop_num], :]
        # 获取ROI的形状
        roi_idx = roi_mask[i1, margin:(h-margin), margin:(w-margin)]  # 获取ROI的子张量
        padding = (k, k, k, k)  # (left, right, top, bottom)
        padded_roi_mask = F.pad(roi_mask, padding, value=-2)
        padded_images = F.pad(images, padding, value=0)
        # 获取中心点的索引

        
        # if len(valid_indices)>=crop_num:    #     valid_indices = torch.stack(valid_indices,dim=0)
        
# 构造卷积核，用于计算邻域内的正类像素数量
        class_potential_list = []
        kernel = torch.ones((2*k+1, 2*k+1), dtype=torch.float32).cuda()
        for class_mask_idx ,class_counts in positive_pixel.items():
            # torch.set_printoptions(profile='full')
            center_indices = torch.nonzero(roi_idx == class_mask_idx).cpu() 
            neighbor_counts = F.conv2d((roi_mask[i1]==class_mask_idx).float().unsqueeze(0).unsqueeze(0).float(), kernel.unsqueeze(0).unsqueeze(0), padding=k).squeeze(0).squeeze(0).long()
            neighbor_counts_array = np.array(neighbor_counts.cpu())
            # 获取指定坐标位置的元素
            neighbor_counts = neighbor_counts_array[center_indices[:, 0], center_indices[:, 1]]
            neighbor_counts = torch.tensor(neighbor_counts).reshape(-1)
            # neighbor_counts = torch.stack(neighbor_counts,dim=0)
            #类不平衡问题?threshold  这里先用的是平均 选谁出来crop
            threshold_low = 0.4 * min(class_counts,crop_size*crop_size)
            threshold_high = 0.8 * max(class_counts,crop_size*crop_size)
            valid_indices = center_indices[(neighbor_counts > threshold_low) & (neighbor_counts < threshold_high)]
            valid_indices.cuda()
            valid_indices += k
            class_potential_list.append(valid_indices)
        
# 计算每个类别应该抽取的数量
        
        # samples_per_class = crop_num // len(class_potential_list) + 1
        samples_per_class = crop_num + 1

        # 创建一个空的最终候选点列表
        final_candidates = []

        # 遍历类别的候选点列表，检查每个类别的候选点数量是否大于等于 samples_per_class
        for candidate_list in class_potential_list:
            if len(candidate_list) >= samples_per_class:
                # 对于满足条件的类别，从中随机抽取 samples_per_class 个元素，并将它们添加到最终候选点列表中
                samples = random.choices(candidate_list, k=samples_per_class)
                final_candidates.extend(samples)
            elif len(candidate_list):
                samples = random.choices(candidate_list, k=samples_per_class)
                final_candidates.extend(samples)
            else:
                continue

        # 打乱最终候选点列表的顺序
        random.shuffle(final_candidates)

        if len(final_candidates) >= crop_num:
            
        # 截取最终候选点列表的前 crop_number 个元素作为最终的候选点
            final_candidates = final_candidates[:crop_num]
        # 找到符合条件的中心点索引
        # 随机选取crop_num个中心点
            crop_index = torch.stack(final_candidates)
        #随机选取策略

        #随机选取策略

        else:
            
            #不取margin
            # test = roi_mask[i1, margin:(h-margin), margin:(w-margin)]
            # Flage = test[3,5] == roi_mask[i1,margin+3,margin+5]
            
            roi_index = (roi_mask[i1, k:(h-k), k:(w-k)] >= -2).nonzero() ## if NULL then random crop
            rand_index = torch.randperm(roi_index.shape[0])
            #torch.randperm() 函数返回一个随机排列的整数序列，范围从 0 到 roi_index.shape[0] - 1。
            crop_index = roi_index[rand_index[:crop_num], :] + torch.full((1, 2), k).cuda()
            # orgin = roi_index[rand_index[:crop_num], :] 

        #[(1,2) (3,4) (5,6) ...]
        for i2 in range(crop_num):
            h0, w0 = crop_index[i2, 0], crop_index[i2, 1] # centered at (h0, w0)
            #确定一个点
            temp_crops[i1, i2, ...] = padded_images[i1, :, (h0-k):(h0+k), (w0-k):(w0+k)]
            #temp_crops 的形状将是 [b, n, 3, crop_size, crop_size]
            #...用于省略连续的冒号
            temp_mask = padded_roi_mask[i1, (h0-k):(h0+k), (w0-k):(w0+k)].cpu()
            # if temp_mask.sum() / (crop_size*crop_size) <= 0.2:
            #     ## if ratio of uncertain regions < 0.2 then negative
            #     flags[i1, i2+2] = 0

            # plt.imshow(temp_mask, cmap='jet', vmin=-2, vmax=1)
            # plt.colorbar()
            # plt.title("temp_mask")
            
            # plt.savefig(f'{i2}temp_mask.png')
            # plt.close()

            # pil_image = TF.to_pil_image(temp_crops[i1][i2])
            # pil_image.save(f'{i2}'+'crop_image.png')



            unique_elements, counts = temp_mask.unique(return_counts=True)
            sorted_indices = np.argsort(unique_elements.numpy())
            sorted_indices = sorted_indices[::-1]
            contiguous_array = np.copy(sorted_indices)
# 使用sorted_indices对unique_elements进行排序
            sorted_unique_elements = unique_elements[torch.from_numpy(contiguous_array)]

            # 使用sorted_indices获取对应的counts排序
            sorted_counts = counts[torch.from_numpy(contiguous_array)]
            
            for element, count in zip(sorted_unique_elements, sorted_counts):
                
                if element.item() != -1 and element.item() != -2:  # 忽略元素为-1的情况
                    ratio = count.item() / positive_pixel[element.item()]
                    if ratio > 0.2: #标定crop出来的有没有东西
                        element_idx = element.item()
                        flags[i1, i2+2,element_idx+1] = 1
                        
                if element.item() == -2:
                    ratio = count.item() / (crop_size * crop_size)
                    if ratio > 0.70 and torch.all(flags[i1,i2+2] == 0)  :
                        element_idx = element.item()
                        if element.item() == -2:
                            element_idx = element.item() + 1
                        flags[i1, i2+2,element_idx+1] = 1
    #list [ tensor.shape 2 1 3 96 96  ]
    _crops = torch.chunk(temp_crops, chunks=crop_num, dim=1,)
    #temp_crops 是一个张量，chunks 是要分割的块数，dim 是要在哪个维度上进行分块操作
    #b num 3 cs cs -> n * [b, 1, 3, crop_size, crop_size]
    crops = [c[:, 0] for c in _crops]
    #这里使用了切片操作 [:, 0] 来选择第一个元素。切片操作 [:, 0] 表示选择所有批量维度的元素，并在截取数目维度上选择索引为0的元素。
    #crops 里面每个元素是[b, 3, crop_size, crop_size]
    #list [2 3 96 96]
    return crops, flags.cuda()


def cam_to_roi_mask2(cam, cls_label, hig_thre=None, low_thre=None):
    #训练过程中
    b, c, h, w = cam.shape
    #pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1,1,h,w])
    #(b,c) -> (b, c, 1, 1) -> (b,c,h,w)
    valid_cam = cls_label_rep * cam + cls_label_rep * 1e-8
    #表明哪些需要验证
    cam_value, cam_indecies = valid_cam.max(dim=1, keepdim=False)
    #b h w
        # _pseudo_label += 1
    roi_mask = torch.clone(cam_indecies)
    # for b in range(valid_cam.shape[0]):
    #     for channel in range(valid_cam.shape[1]):
    #         print_cam = valid_cam[b][channel].cpu()
            
    #         if torch.sum(print_cam)> 1e-2:
    #             plt.imshow(print_cam, cmap='jet')
    #             plt.colorbar()
    #             plt.title(f'Activation Map - Channel {channel}')
    #             plt.savefig(f'activation_map_{b}_{channel}.png')
    #             plt.close()
    
    roi_mask[cam_value<=hig_thre] = -1
    roi_mask[cam_value<=low_thre] = -2    #roi区域（uncertainty区域mask设置为1,bg为0，物体区为2）
    
    
    return roi_mask

def get_valid_cam(cam, cls_label):
    b, c, h, w = cam.shape
    #pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1,1,h,w])
    valid_cam = cls_label_rep * cam

    return valid_cam

def ignore_img_box(label, img_box, ignore_index):

    pseudo_label = torch.ones_like(label) * ignore_index

    for idx, coord in enumerate(img_box):
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = label[idx, coord[0]:coord[1], coord[2]:coord[3]]

    return pseudo_label

def single_class_crop(images,cls_label = None, roi_mask=None, crop_num=8, crop_size=96):
    #image 2 3 448 448
    crops = []
    
    b, c, h, w = images.shape
    
        # 2 8 3 96 96 
    num_class = 21
    flags = torch.zeros(size=(b, crop_num+2,num_class)).to(images.device)#b crop_num
    flags[:,:2,1:] = cls_label.unsqueeze(1).repeat(1, 2, 1)

    for i1 in range(b):
        # input_image = TF.to_pil_image(images[0])
        # input_image.save('input_image.png')
        fg_pixel_number = 0
        positive_ele = 0
        #adaptive cut
        positive_pixel = {}
        may_fg_pixel_number = 0
        mask_all_element, mask_all_counts = roi_mask[i1].unique(return_counts=True)
        for element, count in zip(mask_all_element, mask_all_counts):
            if element.item() >=-1:
                may_fg_pixel_number += count.item()
                if element.item()>=0:
                    fg_pixel_number +=  count.item()
                    positive_pixel[element.item()] = count.item()
                    positive_ele +=1
        # if positive_ele:
        #     may_fg_pixel_number /= positive_ele
        #     fg_pixel_number /= positive_ele
        #     crop_size = math.sqrt(may_fg_pixel_number)
        
        # crop_size = (math.ceil(crop_size / 16)) * 16

        # crop_size = max(crop_size,32)
        # crop_size = min(crop_size,256)

        crop_size = 96


        temp_crops = torch.zeros(size=(b, crop_num, c, crop_size, crop_size)).to(images.device)
        
        # plt.imshow((roi_mask[i1]).cpu(), cmap='jet', vmin=-2, vmax=20)
        # plt.colorbar()
        # plt.title("full_mask")
        
        # plt.savefig(f'full_mask.png')
        # plt.close()

        margin = 0

        k = crop_size//2 
        threshold = crop_size**2 /4

        roi_idx = roi_mask[i1, margin:(h-margin), margin:(w-margin)]  # 获取ROI的子张量
        padding = (k, k, k, k)  # (left, right, top, bottom)
        padded_roi_mask = F.pad(roi_mask, padding, value=-2)
        padded_images = F.pad(images, padding, value=0)
        # 获取中心点的索引
        center_indices = torch.nonzero(roi_idx == -1).cpu()  # -1表示不确定的位置
        
        # if len(valid_indices)>=crop_num:    #     valid_indices = torch.stack(valid_indices,dim=0)
        
# 构造卷积核，用于计算邻域内的正类像素数量
        class_potential_list = []
        kernel = torch.ones((2*k+1, 2*k+1), dtype=torch.float32).cuda()
        for class_mask_idx ,class_counts in positive_pixel.items():
            conv_mask = roi_mask[i1].clone()
            
            
            conv_mask[(conv_mask!=class_mask_idx) & (conv_mask >= 0)] = 255
            conv_mask[(conv_mask == class_mask_idx) & (conv_mask >= 0)] = 256
            conv_mask[(conv_mask!=class_mask_idx) & (conv_mask < 0)] = 0
            conv_mask[conv_mask==255] = -1
            conv_mask[conv_mask == 256] = 1
            # plt.imshow(conv_mask.cpu(), cmap='jet', vmin=-1, vmax=1)
            # plt.colorbar()
            # plt.title("conv_mask")
            
            # plt.savefig('conv_mask.png')
            # plt.close()
            neighbor_counts = F.conv2d(conv_mask.float().unsqueeze(0).unsqueeze(0).float(), kernel.unsqueeze(0).unsqueeze(0), padding=k).squeeze(0).squeeze(0).long()
        # 使用卷积操作计算邻域内的正类像素数量  类不平衡问题！
        # neighbor_counts = F.conv2d((roi_mask[i1]>=0).float().unsqueeze(0).unsqueeze(0).float(), kernel.unsqueeze(0).unsqueeze(0), padding=0).squeeze(0).squeeze(0).long()
            neighbor_counts_array = np.array(neighbor_counts.cpu())
            # 获取指定坐标位置的元素
            neighbor_counts = neighbor_counts_array[center_indices[:, 0], center_indices[:, 1]]
            neighbor_counts = torch.tensor(neighbor_counts)
            # neighbor_counts = torch.stack(neighbor_counts,dim=0)
            #类不平衡问题?threshold  这里先用的是平均 选谁出来crop
            valid_indices = center_indices[neighbor_counts > min(0.2 *class_counts,0.3*crop_size*crop_size)]
            valid_indices.cuda()
            valid_indices += k
            class_potential_list.append(valid_indices)
        
# 计算每个类别应该抽取的数量
        
        # samples_per_class = crop_num // len(class_potential_list) + 1
        samples_per_class = crop_num

        # 创建一个空的最终候选点列表
        final_candidates = []

        # 遍历类别的候选点列表，检查每个类别的候选点数量是否大于等于 samples_per_class
        for candidate_list in class_potential_list:
            if len(candidate_list) >= samples_per_class:
                # 对于满足条件的类别，从中随机抽取 samples_per_class 个元素，并将它们添加到最终候选点列表中
                samples = random.choices(candidate_list, k=samples_per_class)
                final_candidates.extend(samples)
            else:
                final_candidates.extend(candidate_list)

        # 打乱最终候选点列表的顺序
        random.shuffle(final_candidates)

        if len(final_candidates) >= crop_num:
            
        # 截取最终候选点列表的前 crop_number 个元素作为最终的候选点
            final_candidates = final_candidates[:crop_num]
        # 找到符合条件的中心点索引
        # 随机选取crop_num个中心点
            crop_index = torch.stack(final_candidates)
        #随机选取策略

        #随机选取策略

        else:
            
            #不取margin
            # test = roi_mask[i1, margin:(h-margin), margin:(w-margin)]
            # Flage = test[3,5] == roi_mask[i1,margin+3,margin+5]
            
            roi_index = (roi_mask[i1, margin:(h-margin), margin:(w-margin)] >= -2).nonzero() ## if NULL then random crop
            rand_index = torch.randperm(roi_index.shape[0])
            #torch.randperm() 函数返回一个随机排列的整数序列，范围从 0 到 roi_index.shape[0] - 1。
            crop_index = roi_index[rand_index[:crop_num], :] + torch.full((1, 2), k).cuda()
            # orgin = roi_index[rand_index[:crop_num], :] 

        #[(1,2) (3,4) (5,6) ...]
        for i2 in range(crop_num):
            h0, w0 = crop_index[i2, 0], crop_index[i2, 1] # centered at (h0, w0)
            #确定一个点
            h_max = w_max = padded_images.shape[-1]
            h_start = max(0,h0-k)
            h_end = min(h_max,h0+k)
            w_start = max(0, w0-k)
            w_end = min(w_max,w0+k)
            
            temp_crops[i1, i2, ...] = padded_images[i1, :, h_start:h_end, w_start:w_end]
            #temp_crops 的形状将是 [b, n, 3, crop_size, crop_size]
            #...用于省略连续的冒号
            temp_mask = padded_roi_mask[i1, h_start:h_end, w_start:w_end].cpu()
            # if temp_mask.sum() / (crop_size*crop_size) <= 0.2:
            #     ## if ratio of uncertain regions < 0.2 then negative
            #     flags[i1, i2+2] = 0


            # plt.imshow(temp_mask, cmap='jet', vmin=-2, vmax=20)
            # plt.colorbar()
            # plt.title("temp_mask")
            # plt.savefig(f'{i2}temp_mask.png')
            # plt.close()

            # pil_image = TF.to_pil_image(temp_crops[i1][i2])
            # pil_image.save(f'{i2}'+'crop_image.png')



            unique_elements, counts = temp_mask.unique(return_counts=True)
            sorted_indices = np.argsort(unique_elements.numpy())
            sorted_indices = sorted_indices[::-1]
            contiguous_array = np.copy(sorted_indices)
# 使用sorted_indices对unique_elements进行排序
            sorted_unique_elements = unique_elements[torch.from_numpy(contiguous_array)]

            # 使用sorted_indices获取对应的counts排序
            sorted_counts = counts[torch.from_numpy(contiguous_array)]
            
            for element, count in zip(sorted_unique_elements, sorted_counts):
                
                if element.item() != -1 and element.item() != -2:  # 忽略元素为-1的情况
                    ratio = max(count.item() / positive_pixel[element.item()],count.item()/(crop_size*crop_size))
                    if ratio > 0.3: #标定crop出来的有没有东西
                        element_idx = element.item()
                        flags[i1, i2+2,element_idx+1] = 1
                        
                if element.item() == -2:
                    ratio = count.item() / (crop_size * crop_size)
                    if ratio > 0.70 and torch.all(flags[i1,i2+2] == 0)  :
                        element_idx = element.item()
                        if element.item() == -2:
                            element_idx = element.item() + 1
                        flags[i1, i2+2,element_idx+1] = 1
    #list [ tensor.shape 2 1 3 96 96  ]
    _crops = torch.chunk(temp_crops, chunks=crop_num, dim=1,)
    #temp_crops 是一个张量，chunks 是要分割的块数，dim 是要在哪个维度上进行分块操作
    #b num 3 cs cs -> n * [b, 1, 3, crop_size, crop_size]
    crops = [c[:, 0] for c in _crops]
    #这里使用了切片操作 [:, 0] 来选择第一个元素。切片操作 [:, 0] 表示选择所有批量维度的元素，并在截取数目维度上选择索引为0的元素。
    #crops 里面每个元素是[b, 3, crop_size, crop_size]
    #list [2 3 96 96]
    return crops, flags.cuda()

def crop_from_roi_neg(images,cls_label = None, roi_mask=None, crop_num=8, crop_size=96):
    #image 2 3 448 448
    crops = []
    
    b, c, h, w = images.shape
    
        # 2 8 3 96 96 
    num_class = 21
    flags = torch.zeros(size=(b, crop_num+2,num_class)).to(images.device)#b crop_num
    flags[:,:2,1:] = cls_label.unsqueeze(1).repeat(1, 2, 1)

    for i1 in range(b):
        # input_image = TF.to_pil_image(images[0])
        # input_image.save('input_image.png')
        fg_pixel_number = 0
        positive_ele = 0
        #adaptive cut
        positive_pixel = {}
        may_fg_pixel_number = 0
        mask_all_element, mask_all_counts = roi_mask[i1].unique(return_counts=True)
        for element, count in zip(mask_all_element, mask_all_counts):
            if element.item() >=-1:
                may_fg_pixel_number += count.item()
                if element.item()>=0:
                    fg_pixel_number +=  count.item()
                    positive_pixel[element.item()] = count.item()
                    positive_ele +=1
        if positive_ele:
            may_fg_pixel_number /= positive_ele
            fg_pixel_number /= positive_ele
            crop_size = math.sqrt(may_fg_pixel_number)
        
        crop_size = (math.ceil(crop_size / 16)) * 16

        crop_size = max(crop_size,32)
        crop_size = min(crop_size,256)

        temp_crops = torch.zeros(size=(b, crop_num, c, crop_size, crop_size)).to(images.device)
        
        # plt.imshow((roi_mask[i1]).cpu(), cmap='jet', vmin=-2, vmax=20)
        # plt.colorbar()
        # plt.title("full_mask")
        
        # plt.savefig(f'full_mask.png')
        # plt.close()

        margin = 0

        k = crop_size//2 
        threshold = crop_size**2 /4
        # roi_index = (roi_mask[i1, margin:(h-margin), margin:(w-margin)] == -1).nonzero() #neatrual and negtive
        # #不取margin
        # # test = roi_mask[i1, margin:(h-margin), margin:(w-margin)]
        # # Flage = test[3,5] == roi_mask[i1,margin+3,margin+5]
        # # nonzero() 函数获取掩码中为 True 的元素的索引
        # if roi_index.shape[0] < crop_num:
        #     roi_index = (roi_mask[i1, margin:(h-margin), margin:(w-margin)] >= -2).nonzero() ## if NULL then random crop
        # rand_index = torch.randperm(roi_index.shape[0])
        # #torch.randperm() 函数返回一个随机排列的整数序列，范围从 0 到 roi_index.shape[0] - 1。
        # crop_index = roi_index[rand_index[:crop_num], :]
        # 获取ROI的形状
        roi_idx = roi_mask[i1, margin:(h-margin), margin:(w-margin)]  # 获取ROI的子张量
        padding = (k, k, k, k)  # (left, right, top, bottom)
        padded_roi_mask = F.pad(roi_mask, padding, value=-2)
        padded_images = F.pad(images, padding, value=0)
        # 获取中心点的索引
        center_indices = torch.nonzero(roi_idx == -1).cpu()  # -1表示不确定的位置
        
        # if len(valid_indices)>=crop_num:    #     valid_indices = torch.stack(valid_indices,dim=0)
        
# 构造卷积核，用于计算邻域内的正类像素数量
        class_potential_list = []
        kernel = torch.ones((2*k+1, 2*k+1), dtype=torch.float32).cuda()
        for class_mask_idx ,class_counts in positive_pixel.items():
            neighbor_counts = F.conv2d((roi_mask[i1]==class_mask_idx).float().unsqueeze(0).unsqueeze(0).float(), kernel.unsqueeze(0).unsqueeze(0), padding=k).squeeze(0).squeeze(0).long()
        # 使用卷积操作计算邻域内的正类像素数量  类不平衡问题！
        # neighbor_counts = F.conv2d((roi_mask[i1]>=0).float().unsqueeze(0).unsqueeze(0).float(), kernel.unsqueeze(0).unsqueeze(0), padding=0).squeeze(0).squeeze(0).long()
            neighbor_counts_array = np.array(neighbor_counts.cpu())
            # 获取指定坐标位置的元素
            neighbor_counts = neighbor_counts_array[center_indices[:, 0], center_indices[:, 1]]
            neighbor_counts = torch.tensor(neighbor_counts)
            # neighbor_counts = torch.stack(neighbor_counts,dim=0)
            #类不平衡问题?threshold  这里先用的是平均 选谁出来crop
            valid_indices = center_indices[neighbor_counts > 0.2 * class_counts]
            valid_indices.cuda()
            valid_indices += k
            class_potential_list.append(valid_indices)
        
# 计算每个类别应该抽取的数量
        
        # samples_per_class = crop_num // len(class_potential_list) + 1
        samples_per_class = crop_num

        # 创建一个空的最终候选点列表
        final_candidates = []

        # 遍历类别的候选点列表，检查每个类别的候选点数量是否大于等于 samples_per_class
        for candidate_list in class_potential_list:
            if len(candidate_list) >= samples_per_class:
                # 对于满足条件的类别，从中随机抽取 samples_per_class 个元素，并将它们添加到最终候选点列表中
                samples = random.choices(candidate_list, k=samples_per_class)
                final_candidates.extend(samples)
            else:
                final_candidates.extend(candidate_list)

        # 打乱最终候选点列表的顺序
        random.shuffle(final_candidates)

        if len(final_candidates) >= crop_num:
            
        # 截取最终候选点列表的前 crop_number 个元素作为最终的候选点
            final_candidates = final_candidates[:crop_num]
        # 找到符合条件的中心点索引
        # 随机选取crop_num个中心点
            crop_index = torch.stack(final_candidates)
        #随机选取策略

        #随机选取策略

        else:
            
            #不取margin
            # test = roi_mask[i1, margin:(h-margin), margin:(w-margin)]
            # Flage = test[3,5] == roi_mask[i1,margin+3,margin+5]
            
            roi_index = (roi_mask[i1, margin:(h-margin), margin:(w-margin)] >= -2).nonzero() ## if NULL then random crop
            rand_index = torch.randperm(roi_index.shape[0])
            #torch.randperm() 函数返回一个随机排列的整数序列，范围从 0 到 roi_index.shape[0] - 1。
            crop_index = roi_index[rand_index[:crop_num], :] + torch.full((1, 2), k).cuda()
            # orgin = roi_index[rand_index[:crop_num], :] 

        #[(1,2) (3,4) (5,6) ...]
        for i2 in range(crop_num):
            h0, w0 = crop_index[i2, 0], crop_index[i2, 1] # centered at (h0, w0)
            #确定一个点
            h_max = w_max = padded_images.shape[-1]
            h_start = max(0,h0-k)
            h_end = min(h_max,h0+k)
            w_start = max(0, w0-k)
            w_end = min(w_max,w0+k)
            
            temp_crops[i1, i2, ...] = padded_images[i1, :, h_start:h_end, w_start:w_end]
            #temp_crops 的形状将是 [b, n, 3, crop_size, crop_size]
            #...用于省略连续的冒号
            temp_mask = padded_roi_mask[i1, h_start:h_end, w_start:w_end].cpu()
            # if temp_mask.sum() / (crop_size*crop_size) <= 0.2:
            #     ## if ratio of uncertain regions < 0.2 then negative
            #     flags[i1, i2+2] = 0

            # import shutil
            # import os
            # if os.path.isdir(f'{i2}'):
            #     shutil.rmtree(f'{i2}')
            # os.makedirs(f'{i2}', exist_ok=True)
            # plt.imshow(temp_mask, cmap='jet', vmin=-2, vmax=1)
            # plt.colorbar()
            # plt.title("temp_mask")
            
            # plt.savefig(f'{i2}'+'/'+f'{i2}temp_mask.png')
            # plt.close()

            # pil_image = TF.to_pil_image(temp_crops[i1][i2])
            # pil_image.save(f'{i2}'+'/'+f'{i2}'+'crop_image.png')



            unique_elements, counts = temp_mask.unique(return_counts=True)
            sorted_indices = np.argsort(unique_elements.numpy())
            sorted_indices = sorted_indices[::-1]
            contiguous_array = np.copy(sorted_indices)
# 使用sorted_indices对unique_elements进行排序
            sorted_unique_elements = unique_elements[torch.from_numpy(contiguous_array)]

            # 使用sorted_indices获取对应的counts排序
            sorted_counts = counts[torch.from_numpy(contiguous_array)]
            
            for element, count in zip(sorted_unique_elements, sorted_counts):
                
                if element.item() != -1 and element.item() != -2:  # 忽略元素为-1的情况
                    ratio = count.item() / positive_pixel[element.item()]
                    if ratio > 0.2: #标定crop出来的有没有东西
                        element_idx = element.item()
                        flags[i1, i2+2,element_idx+1] = 1
                        
                if element.item() == -2:
                    ratio = count.item() / (crop_size * crop_size)
                    if ratio > 0.70 and torch.all(flags[i1,i2+2] == 0)  :
                        element_idx = element.item()
                        if element.item() == -2:
                            element_idx = element.item() + 1
                        flags[i1, i2+2,element_idx+1] = 1
    #list [ tensor.shape 2 1 3 96 96  ]
    _crops = torch.chunk(temp_crops, chunks=crop_num, dim=1,)
    #temp_crops 是一个张量，chunks 是要分割的块数，dim 是要在哪个维度上进行分块操作
    #b num 3 cs cs -> n * [b, 1, 3, crop_size, crop_size]
    crops = [c[:, 0] for c in _crops]
    #这里使用了切片操作 [:, 0] 来选择第一个元素。切片操作 [:, 0] 表示选择所有批量维度的元素，并在截取数目维度上选择索引为0的元素。
    #crops 里面每个元素是[b, 3, crop_size, crop_size]
    #list [2 3 96 96]
    return crops, flags.cuda()

def multi_scale_cam2(model, inputs, scales):
    '''process cam and aux-cam'''
    # cam_list, tscam_list = [], []
    b, c, h, w = inputs.shape #（batch channel h w） inputs:原图
    with torch.no_grad():
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)
        #dim = 0 ，batch维度上拼接
        _cam_aux, _cam = model(inputs_cat, cam_only=True)

        _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
        #直接上采样  aug
        _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))
        _cam_aux = F.interpolate(_cam_aux, size=(h,w), mode='bilinear', align_corners=False)
        _cam_aux = torch.max(_cam_aux[:b,...], _cam_aux[b:,...].flip(-1))
        #选取同一个像素上的较大值

        cam_list = [F.relu(_cam)]
        cam_aux_list = [F.relu(_cam_aux)]

        #缩放操作
        for s in scales:
            if s != 1.0:  #原图缩放
                _inputs = F.interpolate(inputs, size=(int(s*h), int(s*w)), mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                _cam_aux, _cam = model(inputs_cat, cam_only=True)

                _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))
                _cam_aux = F.interpolate(_cam_aux, size=(h,w), mode='bilinear', align_corners=False)
                _cam_aux = torch.max(_cam_aux[:b,...], _cam_aux[b:,...].flip(-1))
                #b c h w
                cam_list.append(F.relu(_cam))
                cam_aux_list.append(F.relu(_cam_aux))
        #torch.stack(cam_list, dim=0) 将 cam_list 中的 CAM 张量在新的维度 dim=0 上进行堆叠。
        # 这将创建一个形状为 (num_cams, b, c, h, w) 的新张量，
        # 其中 num_cams 是 CAM 的数量，b 是批量大小，c 是通道数，h 和 w 是 CAM 的高度和宽度。
        #作用是混合同一张图的不同scales的不同cam
        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)

        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1)) #保证cam最小值为0
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5 #归一化

        cam_aux = torch.sum(torch.stack(cam_aux_list, dim=0), dim=0)
        cam_aux = cam_aux + F.adaptive_max_pool2d(-cam_aux, (1, 1))
        cam_aux /= F.adaptive_max_pool2d(cam_aux, (1, 1)) + 1e-5

    return cam, cam_aux


def multi_scale_cam_grad(model, inputs, scales):
    '''process cam and aux-cam'''
    # cam_list, tscam_list = [], []
    b, c, h, w = inputs.shape #（batch channel h w） inputs:原图

    inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)
    #dim = 0 ，batch维度上拼接
    _cam_aux, _cam = model(inputs_cat, cam_only=True)

    _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
    #直接上采样  aug
    _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))
    _cam_aux = F.interpolate(_cam_aux, size=(h,w), mode='bilinear', align_corners=False)
    _cam_aux = torch.max(_cam_aux[:b,...], _cam_aux[b:,...].flip(-1))
    #选取同一个像素上的较大值

    cam_list = [F.relu(_cam)]
    cam_aux_list = [F.relu(_cam_aux)]

    #缩放操作
    for s in scales:
        if s != 1.0:  #原图缩放
            _inputs = F.interpolate(inputs, size=(int(s*h), int(s*w)), mode='bilinear', align_corners=False)
            inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

            _cam_aux, _cam = model(inputs_cat, cam_only=True)

            _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
            _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))
            _cam_aux = F.interpolate(_cam_aux, size=(h,w), mode='bilinear', align_corners=False)
            _cam_aux = torch.max(_cam_aux[:b,...], _cam_aux[b:,...].flip(-1))
            #b c h w
            cam_list.append(F.relu(_cam))
            cam_aux_list.append(F.relu(_cam_aux))
    #torch.stack(cam_list, dim=0) 将 cam_list 中的 CAM 张量在新的维度 dim=0 上进行堆叠。
    # 这将创建一个形状为 (num_cams, b, c, h, w) 的新张量，
    # 其中 num_cams 是 CAM 的数量，b 是批量大小，c 是通道数，h 和 w 是 CAM 的高度和宽度。
    #作用是混合同一张图的不同scales的不同cam
    cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)

    cam = cam + F.adaptive_max_pool2d(-cam, (1, 1)) #保证cam最小值为0
    cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5 #归一化

    cam_aux = torch.sum(torch.stack(cam_aux_list, dim=0), dim=0)
    cam_aux = cam_aux + F.adaptive_max_pool2d(-cam_aux, (1, 1))
    cam_aux /= F.adaptive_max_pool2d(cam_aux, (1, 1)) + 1e-5

    return cam, cam_aux

def multi_scale_cam_test(model, inputs, scales):
    '''process cam and aux-cam'''
    # cam_list, tscam_list = [], []
    b, c, h, w = inputs.shape #（batch channel h w） inputs:原图
    with torch.no_grad():
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)
        #dim = 0 ，batch维度上拼接
        _cam_aux, _cam, _cam_crop = model(inputs_cat, cam_crop=True)

        _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
        #直接上采样  aug
        _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))

        _cam_crop = F.interpolate(_cam_crop, size=(h,w), mode='bilinear', align_corners=False)
        #直接上采样  aug
        _cam_crop = torch.max(_cam_crop[:b,...], _cam_crop[b:,...].flip(-1))


        _cam_aux = F.interpolate(_cam_aux, size=(h,w), mode='bilinear', align_corners=False)
        _cam_aux = torch.max(_cam_aux[:b,...], _cam_aux[b:,...].flip(-1))
        #选取同一个像素上的较大值

        cam_list = [F.relu(_cam)]
        cam_aux_list = [F.relu(_cam_aux)]
        cam_crop_list = [F.relu(_cam_crop)]
        #缩放操作
        for s in scales:
            if s != 1.0:  #原图缩放
                _inputs = F.interpolate(inputs, size=(int(s*h), int(s*w)), mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                _cam_aux, _cam ,  _cam_crop = model(inputs_cat, cam_crop=True)

                _cam = F.interpolate(_cam, size=(h,w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b,...], _cam[b:,...].flip(-1))
                _cam_crop = F.interpolate(_cam_crop, size=(h,w), mode='bilinear', align_corners=False)
              #直接上采样  aug
                _cam_crop = torch.max(_cam_crop[:b,...], _cam_crop[b:,...].flip(-1))

                _cam_aux = F.interpolate(_cam_aux, size=(h,w), mode='bilinear', align_corners=False)
                _cam_aux = torch.max(_cam_aux[:b,...], _cam_aux[b:,...].flip(-1))
                #b c h w
                cam_list.append(F.relu(_cam))
                cam_crop_list.append(F.relu(_cam_crop))
                cam_aux_list.append(F.relu(_cam_aux))
        #torch.stack(cam_list, dim=0) 将 cam_list 中的 CAM 张量在新的维度 dim=0 上进行堆叠。
        # 这将创建一个形状为 (num_cams, b, c, h, w) 的新张量，
        # 其中 num_cams 是 CAM 的数量，b 是批量大小，c 是通道数，h 和 w 是 CAM 的高度和宽度。
        #作用是混合同一张图的不同scales的不同cam
        # cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)

        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1)) #保证cam最小值为0
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5 #归一化

        cam_aux = torch.sum(torch.stack(cam_aux_list, dim=0), dim=0)
        cam_aux = cam_aux + F.adaptive_max_pool2d(-cam_aux, (1, 1))
        cam_aux /= F.adaptive_max_pool2d(cam_aux, (1, 1)) + 1e-5
        
        cam_crop = torch.sum(torch.stack(cam_crop_list, dim=0), dim=0)
        cam_crop = cam_crop + F.adaptive_max_pool2d(-cam_crop, (1, 1))
        cam_crop /= F.adaptive_max_pool2d(cam_crop, (1, 1)) + 1e-5

    return cam, cam_aux,cam_crop


def label_to_aff_mask(cam_label, ignore_index=255):

    #cam_label 2 28 28
    b,h,w = cam_label.shape
    # 2 784 784(判patch之间的pair)
    _cam_label = cam_label.reshape(b, 1, -1)
    _cam_label_rep = _cam_label.repeat([1, _cam_label.shape[-1], 1])
    _cam_label_rep_t = _cam_label_rep.permute(0,2,1)
    aff_label = (_cam_label_rep == _cam_label_rep_t).type(torch.long)
    #转置相等就是同一类一个是表示 第0个patch的种类，一个是表示所有patch的class，合起来就是多少个和0th是同类
    for i in range(b):
        aff_label[i, :, _cam_label_rep[i, 0, :]==ignore_index] = ignore_index
        aff_label[i, _cam_label_rep[i, 0, :]==ignore_index, :] = ignore_index  #横纵都要做
    aff_label[:, range(h*w), range(h*w)] = ignore_index #对角线是自身，所以忽略
    return aff_label


def refine_cams_with_bkg_v2(ref_mod=None, images=None, cams=None, cls_labels=None, high_thre=None, low_thre=None, ignore_index=False,  img_box=None, down_scale=2):

    b,_,h,w = images.shape
    _images = F.interpolate(images, size=[h//down_scale, w//down_scale], mode="bilinear", align_corners=False)

    bkg_h = torch.ones(size=(b,1,h,w))*high_thre
    bkg_h = bkg_h.to(cams.device)
    bkg_l = torch.ones(size=(b,1,h,w))*low_thre
    bkg_l = bkg_l.to(cams.device)
    #两个阈值
    bkg_cls = torch.ones(size=(b,1))
    bkg_cls = bkg_cls.to(cams.device)
    cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

    refined_label = torch.ones(size=(b, h, w)) * ignore_index
    refined_label = refined_label.to(cams.device)
    refined_label_h = refined_label.clone()
    refined_label_l = refined_label.clone()
    #拼接上原cam
    cams_with_bkg_h = torch.cat((bkg_h, cams), dim=1)
    _cams_with_bkg_h = F.interpolate(cams_with_bkg_h, size=[h//down_scale, w//down_scale], mode="bilinear", align_corners=False)#.softmax(dim=1)
    cams_with_bkg_l = torch.cat((bkg_l, cams), dim=1)
    _cams_with_bkg_l = F.interpolate(cams_with_bkg_l, size=[h//down_scale, w//down_scale], mode="bilinear", align_corners=False)#.softmax(dim=1)
    #down_scale cam 到 b 21 224 224
    
    for idx, coord in enumerate(img_box):

        valid_key = torch.nonzero(cls_labels[idx,...])[:,0]
        valid_cams_h = _cams_with_bkg_h[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)  
        valid_cams_l = _cams_with_bkg_l[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)
        #这里就是只取valid维度的，其他我都不管
        _refined_label_h = _refine_cams(ref_mod=ref_mod, images=_images[[idx],...], cams=valid_cams_h, valid_key=valid_key, orig_size=(h, w))
        _refined_label_l = _refine_cams(ref_mod=ref_mod, images=_images[[idx],...], cams=valid_cams_l, valid_key=valid_key, orig_size=(h, w))
        
        refined_label_h[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_h[0, coord[0]:coord[1], coord[2]:coord[3]]
        refined_label_l[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_l[0, coord[0]:coord[1], coord[2]:coord[3]]

    refined_label = refined_label_h.clone()
    refined_label[refined_label_h == 0] = ignore_index
    refined_label[(refined_label_h + refined_label_l) == 0] = 0
    #同时为0才是0 bg区域
    return refined_label

def _refine_cams(ref_mod, images, cams, valid_key, orig_size):

    refined_cams = ref_mod(images, cams)
    refined_cams = F.interpolate(refined_cams, size=orig_size, mode="bilinear", align_corners=False)
    #升采样回去，取argmax当作类别
    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]
    #valid_key中的元素0对应索引位置0，而4对应索引位置1。refined_label中的每个元素将被替换为对应valid_key索引位置上的值。
    #如，如果refined_label[0, 0, 0]的值为0，而valid_key[0]的值为10，那么通过valid_key[refined_label]操作后，、
    # refined_label[0, 0, 0]的值将被替换为10
    return refined_label


def netural_cam_loss(fmap,mask,cls_token):
    fmap = F.interpolate(fmap,size=mask.shape[1:],mode='bilinear', align_corners=False)
    b ,c ,h ,w = fmap.shape
    fmap = fmap.reshape(b,c,h*w)
    mask = mask.reshape(b,h*w) 
    loss = 0.0
    for idx in range(b):
        single_map = fmap[idx]
        single_mask = mask[idx]

        # Find indices where mask = 1
        indices = torch.nonzero(single_mask == 1)

        if indices.numel() > 0:
            # Extract corresponding vectors from cls_token
            cls_tokens = cls_token[idx][:]
            
            # Calculate loss between cls_tokens and corresponding fmap vectors
            target = single_map[:, indices[:, 0]]
            cls_tokens = cls_tokens.unsqueeze(-1).repeat(1,target.shape[-1])
            
            loss += F.mse_loss(cls_tokens,target)

# 计算损失（均方误差损失）
    return loss 


def cam_patch_contrast_loss(cam, cls_label, img_box=None, bkg_thre=None, high_thre=None, low_thre=None, ignore_mid=False, ignore_index=None,fmap=None):
    #way1
    b, c, h, w = cam.shape
    #pseudo_label = torch.zeros((b,h,w))
    
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1,1,h,w])
    valid_cam = cls_label_rep * cam
    cam_value, _pseudo_label = valid_cam.max(dim=1, keepdim=False)  #pseudo_label 是对应得索引
    _pseudo_label += 1
    _pseudo_label[cam_value<(high_thre)] = 0
    count_fg = torch.sum(_pseudo_label,dim=(1,2))
    
    for i in range(b):
        feature_vector = []
        arg = torch.nonzero(cls_label[i]) + 1
        arg = arg.tolist()
        for cls in arg:
            cls = cls[0]
            if torch.all(_pseudo_label[i] != cls):
                value,indices = torch.topk(valid_cam[i][cls-1],k=3)
                row_indices = indices[:, 0]
                col_indices = indices[:, 1]
                sub_fmap = fmap[i, :, row_indices, col_indices]
                feature_vector.append(torch.mean(sub_fmap, dim=-1))
            else:
                indices = torch.nonzero(_pseudo_label[i] == cls)
                row_indices = indices[:, 0]
                col_indices = indices[:, 1]

                # 使用高级索引提取子张量
                sub_fmap = fmap[i, :, row_indices, col_indices]
                feature_vector.append(torch.mean(sub_fmap, dim=-1))

        features_vector = torch.stack(feature_vector,dim=0)
        similarity_matrix = torch.matmul(features_vector, features_vector.t())
        norms = torch.norm(features_vector, dim=1, keepdim=True)
        similarity_matrix /= torch.matmul(norms, norms.t())
        similarity_matrix = abs(similarity_matrix)
        similarity_matrix = torch.clamp(similarity_matrix, min=0, max=1)
        identity_matrix = torch.eye(similarity_matrix.size(0)).cuda()
        
        bce_loss = F.binary_cross_entropy(similarity_matrix,identity_matrix)



        print(similarity_matrix)

    

    #就是裁剪框外边的忽略掉，只看里边的
    return bce_loss

def cam_to_label_resized(cam, cls_label, img_box=None, bkg_thre=None, high_thre=None, low_thre=None, ignore_mid=False, ignore_index=None,printornot = False,clip = False):
    b, c, h, w = cam.shape
    #pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1,1,h,w])
    valid_cam = cls_label_rep * cam
    cam_value, _pseudo_label = valid_cam.max(dim=1, keepdim=False)  #pseudo_label 是对应得索引
    _pseudo_label += 1
    _pseudo_label[cam_value<=bkg_thre] = 0
            #b h w

    #cam value [b 448 448] _pseudo_label [b 448 448]每个位置是索引
    if img_box is None:
        return _pseudo_label

    if ignore_mid:
        _pseudo_label[cam_value<=high_thre] = ignore_index
        _pseudo_label[cam_value<=low_thre] = 0
    pseudo_label = torch.ones_like(_pseudo_label) * ignore_index

    
    for idx, coord in enumerate(img_box):
        coord = coord // 16
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = _pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]]

    if printornot:            
        plt.imshow((pseudo_label[0]).cpu(), cmap='jet', vmin=-2, vmax=20)
        plt.colorbar()
        plt.title("aux_mask")
            
        plt.savefig(f'aux_mask.png')
        plt.close()

    return valid_cam, pseudo_label