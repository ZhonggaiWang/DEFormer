import argparse
import datetime
import logging
import os
import random
import sys
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.append("..")
import sys
import os
from collections import OrderedDict
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import voc as voc
from model.losses import get_masked_ptc_loss, get_seg_loss, CTCLoss_neg, DenseEnergyLoss, get_energy_loss,CPCLoss,get_spacial_bce_loss,get_seg_consistence_loss
from model.model_seg_neg import network
from model.double_seg_head import network_du_heads_independent_config
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from model.PAR import PAR
from utils import evaluate, imutils, optimizer
from utils.camutils import single_class_crop,cam_to_label, cam_to_roi_mask2, multi_scale_cam2, label_to_aff_mask, refine_cams_with_bkg_v2,cam_to_label_resized,get_per_pic_thre,multi_scale_cam2_du_heads
from utils.pyutils import AverageMeter, cal_eta, format_tabs, setup_logger
torch.hub.set_dir("./pretrained")
parser = argparse.ArgumentParser()

parser.add_argument("--backbone", default='vit_base_patch16_224', type=str, help="vit_base_patch16_224")
parser.add_argument("--pooling", default='gmp', type=str, help="pooling choice for patch tokens")
parser.add_argument("--pretrained", default=True, type=bool, help="use imagenet pretrained weights")

parser.add_argument("--data_folder", default='../VOC2012', type=str, help="dataset folder")
parser.add_argument("--list_folder", default='../datasets/voc', type=str, help="train/val/test list file")
parser.add_argument("--num_classes", default=21, type=int, help="number of classes")
parser.add_argument("--crop_size", default=448, type=int, help="crop_size in training")
parser.add_argument("--local_crop_size", default=96, type=int, help="crop_size for local view")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")

parser.add_argument("--work_dir", default="work_dir_voc_wseg", type=str, help="work_dir_voc_wseg")

parser.add_argument("--train_set", default="train_aug", type=str, help="training split")
parser.add_argument("--val_set", default="val", type=str, help="validation split")
parser.add_argument("--spg", default=4, type=int, help="samples_per_gpu")
parser.add_argument("--scales", default=(0.5, 2), help="random rescale in training")

parser.add_argument("--optimizer", default='PolyWarmupAdamW', type=str, help="optimizer")
parser.add_argument("--lr", default=6e-5, type=float, help="learning rate")
parser.add_argument("--warmup_lr", default=6e-6, type=float, help="warmup_lr")
parser.add_argument("--wt_decay", default=1e-2, type=float, help="weights decay")
parser.add_argument("--betas", default=(0.9, 0.999), help="betas for Adam")
parser.add_argument("--power", default=0.9, type=float, help="poweer factor for poly scheduler")

parser.add_argument("--max_iters", default=8000, type=int, help="max training iters")
parser.add_argument("--log_iters", default=200, type=int, help=" logging iters")
parser.add_argument("--eval_iters", default=2000, type=int, help="validation iters")
parser.add_argument("--warmup_iters", default=1500, type=int, help="warmup_iters")

parser.add_argument("--high_thre", default=0.7, type=float, help="high_bkg_score")
parser.add_argument("--low_thre", default=0.25, type=float, help="low_bkg_score")
parser.add_argument("--bkg_thre", default=0.5, type=float, help="bkg_score")
parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5), help="multi_scales for cam")

parser.add_argument("--w_ptc", default=0.2, type=float, help="w_ptc")
parser.add_argument("--w_ctc", default=0.45, type=float, help="w_ctc")
parser.add_argument("--w_seg", default=0.1, type=float, help="w_seg")
parser.add_argument("--w_reg", default=0.05, type=float, help="w_reg")
parser.add_argument("--uncertain_region_thre", default=0.2 , type=float, help="uncertain_region_thre")
parser.add_argument("--t_b1_mix_cam", default=0.7, type=float, help="t_b1_mix_cam")
parser.add_argument("--t_b2_mix_cam", default=0.5, type=float, help="t_b2_mix_cam")
parser.add_argument("--w_spacial_bce", default=0.4,type=float, help="w_spacial_bce")

parser.add_argument("--temp", default=0.5, type=float, help="temp")
parser.add_argument("--momentum", default=0.9, type=float, help="temp")
parser.add_argument("--aux_layer", default=-3, type=int, help="aux_layer")

parser.add_argument("--seed", default=0, type=int, help="fix random seed")
parser.add_argument("--save_ckpt",default=True, action="store_true", help="save_ckpt")

parser.add_argument("--local-rank", type=int, default=0)
parser.add_argument("--num_workers", default=10, type=int, help="num_workers")
parser.add_argument('--backend', default='nccl')

# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '5680'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES']='3,4,5'

logging.getLogger().setLevel(logging.INFO)



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def show_mask(cams_aux,cls_label,low_thre,high_thre):
    import matplotlib.pyplot as plt
    # roi_mask_crop = cam_to_roi_mask2(cams_aux.detach(), cls_label=cls_label, low_thre=low_thre, hig_thre=high_thre)
    
    plt.imshow(cams_aux.squeeze(0).cpu(), cmap='jet', vmin=-2, vmax=20)
    plt.colorbar()
    plt.title("aux_mask")
    
    plt.savefig(f'aux_mask.png')
    plt.close()

def show_mask_cam(cams_aux,cls_label,low_thre,high_thre):
    import matplotlib.pyplot as plt
    roi_mask_crop = cam_to_roi_mask2(cams_aux.detach(), cls_label=cls_label, low_thre=low_thre, hig_thre=high_thre)
    
    plt.imshow(roi_mask_crop[0].squeeze(0).cpu(), cmap='jet', vmin=-2, vmax=20)
    plt.colorbar()
    plt.title("cam_mask")
    
    plt.savefig(f'cam_mask.png')
    plt.close()

def validate(model=None, data_loader=None, args=None):

    preds, gts, cams, cams_aux = [], [], [], []
    model.eval()
    avg_meter = AverageMeter()
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):

            name, inputs, labels, cls_label = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            cls_label = cls_label.cuda()

            inputs  = F.interpolate(inputs, size=[args.crop_size, args.crop_size], mode='bilinear', align_corners=False)

            cls, segs, _,_= model(inputs,)

            cls_pred = (cls>0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})

            _cams, _cams_aux = multi_scale_cam2(model, inputs, args.cam_scales)
            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label = cam_to_label(resized_cam, cls_label, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)

            resized_cam_aux = F.interpolate(_cams_aux, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label_aux = cam_to_label(resized_cam_aux, cls_label, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)

            cls_pred = (cls > 0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})

            resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)

            preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
            cams += list(cam_label.cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))
            cams_aux += list(cam_label_aux.cpu().numpy().astype(np.int16))

            # valid_label = torch.nonzero(cls_label[0])[:, 0]
            # out_cam = torch.squeeze(resized_cam)[valid_label]
            # np.save(os.path.join(cfg.work_dir.pred_dir, name[0]+'.npy'), {"keys":valid_label.cpu().numpy(), "cam":out_cam.cpu().numpy()})

    cls_score = avg_meter.pop('cls_score')
    seg_score = evaluate.scores(gts, preds)
    cam_score = evaluate.scores(gts, cams)
    cam_aux_score = evaluate.scores(gts, cams_aux)
    model.train()

    tab_results = format_tabs([cam_score, cam_aux_score, seg_score], name_list=["CAM", "aux_CAM", "Seg_Pred"], cat_list=voc.class_list)

    return cls_score, tab_results

def train(args=None):

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.backend)
    # dist.init_process_group(backend='nccl', init_method='env://', rank = 0, world_size = 1)
    logging.info("Total gpus: %d, samples per gpu: %d..."%(dist.get_world_size(), args.spg))

    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    train_dataset = voc.VOC12ClsDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.train_set,
        stage='train',
        aug=True,
        # resize_range=cfg.dataset.resize_range,
        rescale_range=args.scales,
        crop_size=args.crop_size,
        img_fliplr=True,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )

    val_dataset = voc.VOC12SegDataset(
        root_dir=args.data_folder,
        name_list_dir=args.list_folder,
        split=args.val_set,
        stage='val',
        aug=False,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.spg,
        #shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
        sampler=train_sampler,
        prefetch_factor=4)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=False,
                            drop_last=False)

    device = torch.device(args.local_rank)

    model = network_du_heads_independent_config(
        backbone=args.backbone,
        num_classes=args.num_classes,
        pretrained=False,
        init_momentum=args.momentum,
        aux_layer=args.aux_layer,
    )
    
#pretrained_load——————————————————————————————————————————————————————————————————————————————————————————————————
    # CPC_loss = CPCLoss().cuda()
    # trained_state_dict = torch.load('/home/zhonggai/python-work-space/DEFormer/DEFormer/scripts/work_dir_voc_wseg/baseline/checkpoints/default_model_iter_10000.pth', map_location="cpu")
    # new_state_dict = OrderedDict()
    
    # if 'model' in trained_state_dict:
    #     model_state_dict = trained_state_dict['model']
    #     for k, v in model_state_dict.items():
    #         k = k.replace('module.', '')
    #         new_state_dict[k] = v
    # else:
    #     for k, v in trained_state_dict.items():
    #         k = k.replace('module.', '')
    #         new_state_dict[k] = v

    # model.load_state_dict(state_dict=new_state_dict, strict=False)
    # # if 'CPC_loss' in trained_state_dict:
    # #     CPC_loss = trained_state_dict['CPC_loss']




#add param ——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    param_groups = model.get_param_groups()
    # ccf_param = CPC_loss.get_param_groups()
    # param_groups[2].append(ccf_param)
    model.to(device)

    # cfg.optimizer.learning_rate *= 2
    optim = getattr(optimizer, args.optimizer)(
        params=[
            {
                "params": param_groups[0],
                "lr": args.lr * 2,
                "weight_decay": args.wt_decay,
            },
            {
                "params": param_groups[1],
                "lr": args.lr * 2,
                "weight_decay": args.wt_decay,
            },
            {
                "params": param_groups[2],
                "lr": args.lr * 10,
                "weight_decay": args.wt_decay,
            },
            {
                "params": param_groups[3],
                "lr": args.lr * 10,
                "weight_decay": args.wt_decay,
            },
        ],
        lr=args.lr,
        weight_decay=args.wt_decay,
        betas=args.betas,
        warmup_iter=args.warmup_iters,
        max_iter=args.max_iters,
        warmup_ratio=args.warmup_lr,
        power=args.power)




    logging.info('\nOptimizer: \n%s' % optim)
    model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    train_sampler.set_epoch(np.random.randint(args.max_iters))
    train_loader_iter = iter(train_loader)
    avg_meter = AverageMeter()


    # loss_layer = DenseEnergyLoss(weight=1e-7, sigma_rgb=15, sigma_xy=100, scale_factor=0.5)
    ncrops = 10
    CTC_loss = CTCLoss_neg(ncrops=ncrops, temp=args.temp).cuda()
    
    par = PAR(num_iter=10, dilations=[1,2,4,8,12,24]).cuda()

    for n_iter in range(args.max_iters):

        try:
            img_name, inputs, cls_label, img_box, crops,image_origin = next(train_loader_iter)

        except:
            train_sampler.set_epoch(np.random.randint(args.max_iters))
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_label, img_box, crops,image_origin = next(train_loader_iter)

        
        # input_image = TF.to_pil_image(image_origin[0].permute(2,0,1))
        # input_image.save('input_image.png')
        # print(img_name)

        inputs = inputs.to(device, non_blocking=True)
        inputs_denorm = imutils.denormalize_img2(inputs.clone())
        cls_label = cls_label.to(device, non_blocking=True)

        # get local crops from uncertain regions   multi scale cam 方法，返回的是一个multi scale混合的cam
        b1_cams, b1_cams_aux = multi_scale_cam2_du_heads(model, branch = 'b1', inputs=inputs, scales=args.cam_scales) #448
        b2_cams, b2_cams_aux = multi_scale_cam2_du_heads(model, branch='b2', inputs=inputs, scales=args.cam_scales)
        
        #改动cam_aux
        # b1_roi_mask = cam_to_roi_mask2(b1_cams.detach(), cls_label=cls_label, low_thre=args.low_thre, hig_thre=args.high_thre)
        # b2_roi_mask = cam_to_roi_mask2(b2_cams.detach(), cls_label=cls_label, low_thre=args.low_thre, hig_thre=args.high_thre)
        # # #b h w

        # /local_crops, flags= single_class_crop(images=inputs, cls_label = cls_label,roi_mask=roi_mask, crop_num=ncrops-2, crop_size=args.local_crop_size)
        # roi_crops = crops[:2] + local_crops #全局的两张图 + local的多张图


        
# model forward-------------------------------------------------------------------------------------------------------------------------------

        cls, segs, fmap, cls_aux, cam_12th, cls_token = model(inputs, crops='#', n_iter=n_iter,select_k = 1,return_cam = True)    
        b1_cls, b2_cls = cls[0], cls[1]
        b1_segs, b2_segs = segs[0], segs[1]
        b1_fmap, b2_fmap = fmap[0], fmap[1]
        b1_cls_aux, b2_cls_aux = cls_aux[0], cls_aux[1]
        b1_cam_12th, b2_cam_12th = cam_12th[0], cam_12th[1]
        b1_cls_token, b2_cls_token = cls_token[0], cls_token[1]        
        
# discrepancy loss ----------------------------------------------------------------

        # fmap_1_flat = b1_fmap.view(b1_fmap.shape[0], b1_fmap.shape[1], -1)
        # fmap_2_flat = b2_fmap.view(b2_fmap.shape[0], b2_fmap.shape[1], -1)

        # cross_network_cos_simi = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        # b1_sim_loss = 1 + torch.abs(cross_network_cos_simi(fmap_1_flat.detach(), fmap_2_flat).mean())
        # b2_sim_loss = 1 + torch.abs(cross_network_cos_simi(fmap_2_flat.detach(), fmap_1_flat).mean())

        cross_network_cos_simi = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        b1_sim_loss = 1 - torch.abs(cross_network_cos_simi(b1_cls_token.detach(), b2_cls_token).mean())
        b2_sim_loss = 1 - torch.abs(cross_network_cos_simi(b2_cls_token.detach(), b1_cls_token).mean())
        
        network_sim_loss = b1_sim_loss + b2_sim_loss
        # network_sim_loss = torch.tensor(0)

# branch2 : spacial-bce + cls loss part-------------------------------------------------------------------------------------------------------------------


        # valid_aux_cam, _ = cam_to_label(cams_aux.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)
        # refined_aux_pesudo_label = refine_cams_with_bkg_v2(par, inputs_denorm, cams=valid_aux_cam, cls_labels=cls_label,  high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index, img_box=img_box, )
        #pesudo_label [b,h,w] 
        aux_pesedo_show = b2_refined_aux_pesudo_label = cam_to_roi_mask2(b2_cams_aux.detach(), cls_label=cls_label, low_thre=args.low_thre, hig_thre=args.high_thre)
        per_pic_thre = get_per_pic_thre(b2_refined_aux_pesudo_label, gd_label=cls_label, uncertain_region_thre = args.uncertain_region_thre)
        b2_spacial_bce_loss = get_spacial_bce_loss(b2_cam_12th, cls_label, per_pic_thre)
        b2_cls_loss = F.multilabel_soft_margin_loss(b2_cls, cls_label)
        b2_cls_loss_aux = F.multilabel_soft_margin_loss(b2_cls_aux, cls_label)


    
#show mask --------------------------------------------------------------------------------------------------------------
        # from PIL import Image, ImageOps
        # show_mask(aux_pesedo_show,cls_label,args.low_thre,args.high_thre)    
        # show_mask_cam(b1_cams,cls_label,args.low_thre,args.high_thre)
        # input_image = TF.to_pil_image(image_origin[0].permute(2,0,1))
        # input_image.save('input_image.png')

        
# branch1 : cls-loss-------------------------------------------------------------------------------------------------------------------
        b1_cls_loss = F.multilabel_soft_margin_loss(b1_cls, cls_label)
        b1_cls_loss_aux = F.multilabel_soft_margin_loss(b1_cls_aux, cls_label)
        

        
        
        
#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
        cpc_loss = torch.tensor(0)
        ctc_loss = torch.tensor(0)
        # ctc_loss crop出来的结果过网络得cls，计算loss
        # ctc_loss = CTC_loss(out_s, out_t, flags,cls_label)


  


        
#b1 b2 generate pesudo-label and seg------------------------------------------------------------------------------------------------------------------------------------------
        b1_mix_cams = args.t_b1_mix_cam * b1_cams.detach() + (1-args.t_b1_mix_cam) * b2_cams.detach()
        b1_valid_cam, _ = cam_to_label(b1_mix_cams.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)
        b1_refined_pseudo_label = refine_cams_with_bkg_v2(par, inputs_denorm, cams=b1_valid_cam, cls_labels=cls_label,  high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index, img_box=img_box, )
        b2_segs = F.interpolate(b2_segs, size=b1_refined_pseudo_label.shape[1:], mode='bilinear', align_corners=False)
        #segs是粗分割lfov的结果 refined_pseudo_label是自己生成的伪标签（cam是最后给出的cam）
        b2_seg_loss = get_seg_loss(b2_segs, b1_refined_pseudo_label.type(torch.long), ignore_index=args.ignore_index)
        #cross head
        
        b2_mix_cams = args.t_b2_mix_cam * b2_cams.detach() + (1-args.t_b2_mix_cam) * b1_cams.detach()
        b2_valid_cam, _ = cam_to_label(b2_mix_cams.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)
        b2_refined_pseudo_label = refine_cams_with_bkg_v2(par, inputs_denorm, cams=b2_valid_cam, cls_labels=cls_label,  high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index, img_box=img_box, )
        b1_segs = F.interpolate(b1_segs, size=b2_refined_pseudo_label.shape[1:], mode='bilinear', align_corners=False)
        #segs是粗分割lfov的结果 refined_pseudo_label是自己生成的伪标签（cam是最后给出的cam）
        b1_seg_loss = get_seg_loss(b1_segs, b2_refined_pseudo_label.type(torch.long), ignore_index=args.ignore_index)
        
        seg_loss = b1_seg_loss + b2_seg_loss
    
#seg_result_consistence_loss--------------------------------------------------------------------------------
        b1_seg_consistence_loss = get_seg_consistence_loss(b1_segs,b2_segs.detach())
        b2_seg_consistence_loss = get_seg_consistence_loss(b2_segs,b1_segs.detach())
        
        seg_consistence_loss = 0.5 * b1_seg_consistence_loss + 0.5 * b2_seg_consistence_loss
      

    

        

        
        
#b1 b2 ptc loss-------------------------------------------------------------------------------------------------------------------------------------
        b1_resized_cams_aux = F.interpolate(b1_cams_aux, size=b1_fmap.shape[2:], mode="bilinear", align_corners=False)
        #根据这个，给出两两patch是否同一类  （2 20 28 28）
        _,b1_pseudo_label_aux = cam_to_label_resized(b1_resized_cams_aux.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index,printornot=False,clip = False)
        #伪标签是2 20 28 28，这个用于分出pair
        b1_aff_mask = label_to_aff_mask(b1_pseudo_label_aux)
        b1_ptc_loss = get_masked_ptc_loss(b1_fmap, b1_aff_mask)
        
        b2_resized_cams_aux = F.interpolate(b2_cams_aux, size=b2_fmap.shape[2:], mode="bilinear", align_corners=False)
        #根据这个，给出两两patch是否同一类  （2 20 28 28）
        _,b2_pseudo_label_aux = cam_to_label_resized(b2_resized_cams_aux.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index,printornot=False,clip = False)
        #伪标签是2 20 28 28，这个用于分出pair
        b2_aff_mask = label_to_aff_mask(b2_pseudo_label_aux)
        b2_ptc_loss = get_masked_ptc_loss(b2_fmap, b2_aff_mask)

        ptc_loss = b2_ptc_loss + b1_ptc_loss

        # print(n_iter)

#train------------------------------------------------------------------------------------------------------------------------
        if n_iter <= 2000:
            loss = 1.0 * (b1_cls_loss + b2_cls_loss) + 1.0 * (b1_cls_loss_aux + b2_cls_loss_aux) + args.w_ptc * ptc_loss  + 0.0 * seg_loss + 0.1 * network_sim_loss
        elif n_iter <= 3500:
            loss = 1.0 * (b1_cls_loss + b2_cls_loss) + 1.0 * (b1_cls_loss_aux + b2_cls_loss_aux) + args.w_ptc * ptc_loss + args.w_seg * seg_loss + 0.1 * network_sim_loss + 0.05 * seg_consistence_loss
        else:
            loss = (args.w_spacial_bce * b2_spacial_bce_loss + (1-args.w_spacial_bce) * b2_cls_loss) + 1.0 * b1_cls_loss+ 1.0 * (b1_cls_loss_aux + b2_cls_loss_aux) + args.w_ptc * ptc_loss + args.w_seg * seg_loss + 0.1 * network_sim_loss + 0.05 * seg_consistence_loss

        # 如果你增加了 cls_loss 的权重值，使其在整体优化中起到更大的作用，那么模型在训练过程中会更加关注优化 cls_loss
        cls_pred = (b1_cls > 0).type(torch.int16)
        cls_score = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])

        cls_loss = (b1_cls_loss + b2_cls_loss)/2
        cls_loss_aux = (b1_cls_loss_aux + b2_cls_loss_aux) / 2
        spacial_bce_loss = b2_spacial_bce_loss

        avg_meter.add({
            'cls_loss': cls_loss.item(),
            'cls_loss_aux': cls_loss_aux.item(),
            'seg_loss': seg_loss.item(),
            'cls_score': cls_score.item(),
            'spacial_bce_loss' :spacial_bce_loss.item(),
            'network_sim_loss': network_sim_loss.item(),
            'seg_sim_loss': seg_consistence_loss.item(),
        })

        optim.zero_grad()
        
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        torch.cuda.empty_cache()  # 清理CUDA显存中的垃圾数据
        # if n_iter % 100 == 0:
        #     print('n_iter:',n_iter)
        #     print('cls_loss',cls_loss,'\n','cls_loss_aux',cls_loss_aux,'\n','seg_loss',seg_loss,'\n','aur_loss',ctc_loss,'\n','\n','ptc_loss',ptc_loss,'dcc_loss',cpc_loss,'spacial_bce_loss',spacial_bce_loss)
        #     # print(loss)
        if (n_iter + 1) % args.log_iters == 0:
            if args.local_rank ==0:
                delta, eta = cal_eta(time0, n_iter + 1, args.max_iters)
                cur_lr = optim.param_groups[0]['lr']


                logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; cls_loss: %.4f, cls_loss_aux: %.4f,  spacial_bce_loss: %.4f...,network_sim_loss: %.4f..., seg_sim_loss: %.4f... " % (n_iter + 1, delta, eta, cur_lr, avg_meter.pop('cls_loss'), avg_meter.pop('cls_loss_aux'),avg_meter.pop('spacial_bce_loss'),avg_meter.pop('network_sim_loss'),avg_meter.pop('seg_sim_loss')))

        if (n_iter + 1) % 2000 == 0:
            # ckpt_name = os.path.join(args.ckpt_dir, "w/oPSA_model_iter_%d.pth" % (n_iter + 1))
            if args.local_rank ==0:
                logging.info('Validating...branch1')
                    # if args.save_ckpt:
                    #     torch.save(model.state_dict(), ckpt_name)
                val_cls_score, tab_results = validate(model=model.module.eval_branch('b1'), data_loader=val_loader, args=args)
                logging.info("val cls score: %.6f" % (val_cls_score))
                logging.info("\n"+tab_results)
            
                logging.info('Validating...branch2')
                    # if args.save_ckpt:
                    #     torch.save(model.state_dict(), ckpt_name)
                val_cls_score, tab_results = validate(model=model.module.eval_branch('b2'), data_loader=val_loader, args=args)
                logging.info("val cls score: %.6f" % (val_cls_score))
                logging.info("\n"+tab_results)
            
            
        if (n_iter + 1) % 2000 == 0:
            if args.local_rank ==0:
                if args.save_ckpt:
                    print('saving checkpoint')
                    state_dict = {
                        'model': model.state_dict(),
                    }
                    ckpt_name = os.path.join(args.ckpt_dir, "default_model_iter_%d.pth" % (n_iter + 1))
                    torch.save(state_dict, ckpt_name)
        
            

    return True


if __name__ == "__main__":

    args = parser.parse_args()

    timestamp = "{0:%Y-%m-%d-%H-%M-%S-%f}".format(datetime.datetime.now())
    args.work_dir = os.path.join(args.work_dir, timestamp)
    args.ckpt_dir = os.path.join(args.work_dir, "checkpoints")
    args.pred_dir = os.path.join(args.work_dir, "predictions")

    if args.local_rank ==0:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(args.pred_dir, exist_ok=True)

        setup_logger(filename=os.path.join(args.work_dir, 'train.log'))
        logging.info('Pytorch version: %s' % torch.__version__)
        logging.info("GPU type: %s"%(torch.cuda.get_device_name(0)))
        logging.info('\nargs: %s' % args)

    ## fix random seed
    setup_seed(args.seed)
    train(args=args)

