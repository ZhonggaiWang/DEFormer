import argparse
import os
import sys

sys.path.append("..")

from collections import OrderedDict
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import voc
from model.model_seg_neg import network
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.camutils import get_valid_cam
import argparse
import os
import sys

from collections import OrderedDict

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import voc
from model.model_seg_neg import network
# from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import evaluate, imutils
from utils.camutils import cam_to_label, get_valid_cam, multi_scale_cam2
from utils.pyutils import AverageMeter, format_tabs

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="./", type=str, help="model_path")
parser.add_argument("--backbone", default='vit_base_patch16_224', type=str, help="vit_base_patch16_224")
parser.add_argument("--pooling", default='gmp', type=str, help="pooling choice for patch tokens")
parser.add_argument("--data_folder", default='../VOC2012', type=str, help="dataset folder")
parser.add_argument("--list_folder", default='../datasets/voc', type=str, help="train/val/test list file")
parser.add_argument("--num_classes", default=21, type=int, help="number of classes")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")
parser.add_argument("--infer_set", default="val", type=str, help="infer_set")
parser.add_argument("--cam_scales", default=(1.0, 1.25, 1.5), help="multi_scales for cam")
parser.add_argument("--local_rank", default=0, type=int, help="local_rank")
parser.add_argument('--backend', default='nccl')

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5680'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def get_strided_size(orig_size, stride):
    return ((orig_size[0]-1)//stride+1, (orig_size[1]-1)//stride+1)

def _validate(pid, model=None, dataset=None, args=None):

    model.eval()
    data_loader = DataLoader(dataset[pid], batch_size=1, shuffle=False, num_workers=2, pin_memory=False)

    base_dir = args.model_path.split("checkpoint")[0]
    cam_dir = os.path.join(base_dir, "cams")
    os.makedirs(cam_dir, exist_ok=True)

    with torch.no_grad():
        model.cuda()

        for idx, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):

            name, inputs, labels, cls_label = data

            inputs = inputs.cuda()
            img = imutils.denormalize_img(inputs)[0].permute(1,2,0).cpu().numpy()

            inputs  = F.interpolate(inputs, size=[448, 448], mode='bilinear', align_corners=False)
            labels = labels.cuda()
            cls_label = cls_label.cuda()

            ###
            _cams, _cams_aux = multi_scale_cam2(model, inputs, [1.0, 0.5, 1.5])
            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
            resized_cam_aux = F.interpolate(_cams_aux, size=labels.shape[1:], mode='bilinear', align_corners=False)

            # cam_label = cam_to_label(resized_cam, cls_label, bkg_thre=args.bkg_thre)
            # cam_aux_label = cam_to_label(resized_cam_aux, cls_label, bkg_thre=args.bkg_thre)

            resized_cam = get_valid_cam(resized_cam, cls_label)
            resized_cam_aux = get_valid_cam(resized_cam_aux, cls_label)

            valid_label = torch.nonzero(cls_label[0])[:,0]

            npy_dict = {}
            npy_name = os.path.join(cam_dir, name[0] + '.npy')
            for key in valid_label:
                key = key.item()
                npy_dict[key] = resized_cam[0, key].cpu().numpy()
            
            np.save(npy_name, npy_dict)

    return None


def validate():

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', rank = 0, world_size = 1)

    val_dataset = voc.VOC12SegDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.infer_set,
        stage='val',
        aug=False,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )

    model = network(
        backbone=args.backbone,
        num_classes=args.num_classes,
        pretrained=False,
        aux_layer=-3
    )
    model.to(torch.device(args.local_rank))


    trained_state_dict = torch.load('/home/zhonggai/python-work-space/WSSS-work/Token-contrast/ToCo-main/scripts/work_dir_voc_wseg/CPC(proj_fusion_map 72.0)/checkpoints/default_model_iter_30000.pth', map_location="cpu")

    new_state_dict = OrderedDict()
    if 'model' in trained_state_dict:
        model_state_dict = trained_state_dict['model']
        for k, v in model_state_dict.items():
            k = k.replace('module.', '')
            new_state_dict[k] = v
    else:
        for k, v in trained_state_dict.items():
            k = k.replace('module.', '')
            new_state_dict[k] = v
    model.eval()
    model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    n_gpus = dist.get_world_size()
    split_dataset = [torch.utils.data.Subset(val_dataset, np.arange(i, len(val_dataset), n_gpus)) for i in range (n_gpus)]

    _validate(args.local_rank, model=model, dataset=split_dataset, args=args)

    torch.cuda.empty_cache()
    return True


if __name__ == "__main__":

    args = parser.parse_args()
    if args.local_rank == 0:
        print(args)
    validate()

