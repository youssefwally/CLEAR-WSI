#imports
import argparse
import ast

############################################################################################
'''
********************************************************
                   CTransPath
********************************************************
img_size (int | tuple(int)): Input image size. Default (160, 192, 224)
patch_size (int | tuple(int)): Patch size. Default: 4
embed_dims (tuple(int)): Embedding dimensions in each layer.
num_heads (int): Number of attention heads in different layers.
mlp_ratio (tuple(float)): Ratio of mlp hidden dim to embedding dim.
qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
depths (tuple(int)): Depth of each PVT Transformer layer.
sr_ratios (tuple(float)): Spatial-reduction ratio that reduces the size of image via Conv.
drop_rate (float): Dropout rate. Default: 0
drop_path_rate (float): Stochastic depth rate. Default: 0.1
if_transskip (bool): Enable skip connections from Transformer Blocks
if_convskip (bool): Enable skip connections from Convolutional Blocks
reg_head_chan (int): Number of channels in the registration head (i.e., the final convolutional layer) 
embed_dim (int): Patch embedding dimension. Default: 96
depths (tuple(int)): Depth of each Swin Transformer layer.
num_heads (tuple(int)): Number of attention heads in different layers.
window_size (tuple): Window size. Default: 7
'''
# config.norm_layer = partial(nn.LayerNorm, eps=1e-6)
# ssh login.ai.lrz.de -l di35zuz
#SBATCH -p lrz-dgx-a100-80x8
#SBATCH -p mcml-dgx-a100-40x8
#SBATCH -q mcml
# srun --pty --container-mounts=./dss/dssmcmlfs01/pr62la/pr62la-dss-0002/MSc/Yussef/data:/mnt/data,./dss/dsshome1/05/di35zuz/project:/mnt/code  --container-image=./dss/dsshome1/05/di35zuz/ai-med+petmrilrz+latest.sqsh --container-workdir=/mnt/code/  bash /opt/conda/envs/ai-med+petmrilrz+latest/bin/python/scripts/3d_pvt/train.py 
# scp environments/Miniconda3-latest-Linux-x86_64.sh  di35zuz@login.ai.lrz.de:.

def arg_parse():
        parser = argparse.ArgumentParser(description='Arguments.')

#local
        parser.add_argument("--data_path", type=str, default="../../mnt/data/")
#lrz
#     parser.add_argument("--data_path", type=str, default="/mnt/data/")   


        parser.add_argument("--lrz", type=ast.literal_eval, default=True) 
        parser.add_argument("--model", type=str, default="") 
        parser.add_argument("--type", type=str, default="") 
        parser.add_argument("--wandb_mode", type=str, default="disabled") 
        parser.add_argument("--loader", type=str, default="h5") 
        parser.add_argument('--split', dest='split', type=int, default=3)
        parser.add_argument("--binary", type=ast.literal_eval, default=True) 
        parser.add_argument("--test", type=ast.literal_eval, default=False) 
        parser.add_argument("--save", type=ast.literal_eval, default=False) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
        parser.add_argument("--label", type=str, default="")
        parser.add_argument('--epochs', dest='epochs', type=int, help='Epochs.', default=10)
        parser.add_argument('--batch_size', dest='batch_size', type=int, help='batch_size.', default=32)
        parser.add_argument('--seed', dest='seed', type=int, help='seed.', default=42)
    
        parser.add_argument('--img_size', dest='img_size', type=int, help='img_size.', default=128)
        parser.add_argument('--patch_size', dest='patch_size', type=int, help='patch_size.', default=8)
        parser.add_argument('--in_chans', dest='in_chans', type=int, help='in_chans.', default=1)
        parser.add_argument('--num_classes', dest='num_classes', type=int, help='Batches.', default=2)

        #ViT
        parser.add_argument('--dim', dest='dim', type=int, help='dim.', default=768)
        parser.add_argument('--depth', dest='depth', type=int, help='depth.', default=12)
        parser.add_argument('--heads', dest='heads', type=int, help='heads.', default=12)
        parser.add_argument('--dropout', dest='dropout', type=float, help='dropout.', default=0.01)
        parser.add_argument('--emb_dropout', dest='emb_dropout', type=float, help='emb_dropout.', default=0.01)
    
        parser.add_argument("--optimizer", type=str, default="adamW")
        parser.add_argument("--scheduler", type=str, default="CosineAnnealingLR")
        parser.add_argument('--lr', dest='lr', type=float, help='lr.', default=0.001)
        parser.add_argument('--step_size', dest='step_size', type=float, help='step_size.', default=1.00)
        parser.add_argument('--gamma', dest='gamma', type=float, help='gamma.', default=0.0)
        parser.add_argument('--weight_decay', dest='weight_decay', type=float, help='weight_decay.', default=0.0001)
        parser.add_argument('--momentum', dest='momentum', type=float, help='momentum.', default=0.9)

        #Fine Tunning
        parser.add_argument("--pretrained", type=ast.literal_eval, default=False)
        parser.add_argument("--freeze", type=ast.literal_eval, default=True)

        #MoCoV3
        parser.add_argument("--arch", type=str, default="small")
        parser.add_argument("--stop_grad_conv1", type=ast.literal_eval, default=True)
        parser.add_argument("--input_dim", type=int, default=256)
        parser.add_argument("--moco_mlp_dim", type=int, default=4096)
        parser.add_argument("--moco_t", type=float, default=1.0)

        return parser.parse_args()

############################################################################################
