import torch
import torch.nn.functional as F

# 模型的输出概率分布
temp_flag_tv = torch.ones((10))
indices = torch.nonzero(temp_flag_tv == 1)
indices = indices.tolist()  # 转换为普通的Python列表
print(indices)
indices = indices[0]  
print(indices)   