import subprocess
import sys
import argparse

def run_models_train_with_data(data, device, output_file):
    # Define the commands with the [data] placeholder
    wss_commands = [
        f'python ./projects/fm_histopathology/scripts/models/deit/main.py --output_dir ./projects/fm_histopathology/models/wss_deit_tiny_patch16_224 --model deit_tiny_patch16_224 --data-path ../../../../../../../mnt/data/WSSS4LUAD --batch-size 128 --data-set WSS --label patch_level_multi_class_labels --nb_classes 3 --device {device}',
        f'python ./projects/fm_histopathology/scripts/models/deit/main.py --output_dir ./projects/fm_histopathology/models/wss_general_pretrained_deit_tiny_patch16_224 --resume https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth --model deit_tiny_patch16_224 --data-path ../../../../../../../mnt/data/WSSS4LUAD --batch-size 128 --data-set WSS --label patch_level_multi_class_labels --nb_classes 3 --device {device}',
        f'python ./projects/fm_histopathology/scripts/models/deit/main.py --output_dir ./projects/fm_histopathology/models/wss_pretrained_mocov3_tiny_checkpoint --resume ./projects/fm_histopathology/models/mocov3_tiny_checkpoint/mocov3_deit_vit_small.pth --model deit_small_patch16_224 --data-path ../../../../../../../mnt/data/WSSS4LUAD --batch-size 128 --data-set WSS --label patch_level_multi_class_labels --nb_classes 3 --device {device}',
    ]
    patch_cam_commands = [
        f'python ./projects/fm_histopathology/scripts/models/deit/main.py --output_dir ./projects/fm_histopathology/models/patch_cam_deit_tiny_patch16_224 --model deit_tiny_patch16_224 --data-path ../../../../../../../mnt/data/patch_cam/pcamv1/camelyonpatch_level_2_split --batch-size 128 --data-set patch_cam --bce-loss --device {device}',
        f'python ./projects/fm_histopathology/scripts/models/deit/main.py --output_dir ./projects/fm_histopathology/models/patch_cam_general_pretrained_deit_tiny_patch16_224 --resume https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth --model deit_tiny_patch16_224 --data-path ../../../../../../../mnt/data/patch_cam/pcamv1/camelyonpatch_level_2_split --batch-size 128 --data-set patch_cam --bce-loss --device {device}',
        f'python ./projects/fm_histopathology/scripts/models/deit/main.py --output_dir ./projects/fm_histopathology/models/patch_cam_pretrained_mocov3_tiny_checkpoint --resume ./projects/fm_histopathology/models/mocov3_tiny_checkpoint/mocov3_deit_vit_small.pth --model deit_small_patch16_224 --data-path ../../../../../../../mnt/data/patch_cam/pcamv1/camelyonpatch_level_2_split --batch-size 128 --data-set patch_cam --bce-loss --device {device}',
    ]
    mhist_commands = [
        f'python ./projects/fm_histopathology/scripts/models/deit/main.py --output_dir ./projects/fm_histopathology/models/mhist_deit_tiny_patch16_224 --model deit_tiny_patch16_224 --data-path ../../../../../media/research/data_slow/mhist --batch-size 128 --data-set mhist --bce-loss --device {device}',
        f'python ./projects/fm_histopathology/scripts/models/deit/main.py --output_dir ./projects/fm_histopathology/models/mhist_general_pretrained_deit_tiny_patch16_224 --resume https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth --model deit_tiny_patch16_224 --data-path ../../../../../media/research/data_slow/mhist --batch-size 128 --data-set mhist --bce-loss --device {device}',
        f'python ./projects/fm_histopathology/scripts/models/deit/main.py --output_dir ./projects/fm_histopathology/models/mhist_pretrained_mocov3_tiny_checkpoint --resume ./projects/fm_histopathology/models/mocov3_tiny_checkpoint/mocov3_deit_vit_small.pth --model deit_small_patch16_224 --data-path ../../../../../media/research/data_slow/mhist --batch-size 128 --data-set mhist --bce-loss --device {device}',
    ]
    crc_commands = [
        f'python ./projects/fm_histopathology/scripts/models/deit/main.py --output_dir ./projects/fm_histopathology/models/crc_deit_tiny_patch16_224 --model deit_tiny_patch16_224 --data-path ../../../../../..//mnt/data/CRC/ --batch-size 128 --data-set crc --nb_classes 9 --device {device}',
        f'python ./projects/fm_histopathology/scripts/models/deit/main.py --output_dir ./projects/fm_histopathology/models/crc_general_pretrained_deit_tiny_patch16_224 --resume https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth --model deit_tiny_patch16_224 --data-path ../../../../../..//mnt/data/CRC/ --batch-size 128 --data-set crc --nb_classes 9 --device {device}',
        f'python ./projects/fm_histopathology/scripts/models/deit/main.py --output_dir ./projects/fm_histopathology/models/crc_pretrained_mocov3_tiny_checkpoint --resume ./projects/fm_histopathology/models/mocov3_tiny_checkpoint/mocov3_deit_vit_small.pth --model deit_small_patch16_224 --data-path ../../../../../..//mnt/data/CRC/ --batch-size 128 --data-set crc --nb_classes 9 --device {device}',
    ]
    camelyon16_pca_commands = [
        f'python ./projects/fm_histopathology/scripts/models/deit/main.py --output_dir ./projects/fm_histopathology/models/camelyon_16_pca_deit_tiny_patch16_224 --model deit_tiny_patch16_224 --data-path ../../../../../../../../mnt/data/nfs03-R6/CAMELYON16/ --batch-size 128 --data-set camelyon16 --bce-loss --slides --wsi_method pca --lr 0.001 --unscale-lr --device {device}',
        f'python ./projects/fm_histopathology/scripts/models/deit/main.py --output_dir ./projects/fm_histopathology/models/camelyon_16_pca_general_pretrained_deit_tiny_patch16_224 --resume https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth --model deit_tiny_patch16_224 --data-path ../../../../../../../../mnt/data/nfs03-R6/CAMELYON16/ --batch-size 128 --data-set camelyon16 --bce-loss --slides --wsi_method pca --lr 0.001 --unscale-lr --device {device}',
        f'python ./projects/fm_histopathology/scripts/models/deit/main.py --output_dir ./projects/fm_histopathology/models/camelyon_16_pca_pretrained_mocov3_tiny_checkpoint --resume ./projects/fm_histopathology/models/mocov3_tiny_checkpoint/mocov3_deit_vit_small.pth --model deit_small_patch16_224 --data-path ../../../../../../../../mnt/data/nfs03-R6/CAMELYON16/ --batch-size 128 --data-set camelyon16 --bce-loss --slides --wsi_method pca --lr 0.001 --unscale-lr --device {device}',
    ]
    camelyon16_attn_commands = [
        f'python ./projects/fm_histopathology/scripts/models/deit/main.py --output_dir ./projects/fm_histopathology/models/camelyon_16_attn_deit_tiny_patch16_224 --model deit_tiny_patch16_224 --data-path ../../../../../../../../mnt/data/nfs03-R6/CAMELYON16/ --batch-size 1 --data-set camelyon16  --bce-loss --slides --wsi_method attn --lr 0.001 --unscale-lr --device {device}',
        f'python ./projects/fm_histopathology/scripts/models/deit/main.py --output_dir ./projects/fm_histopathology/models/camelyon_16_attn_general_pretrained_deit_tiny_patch16_224 --resume https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth --model deit_tiny_patch16_224 --data-path ../../../../../../../../mnt/data/nfs03-R6/CAMELYON16/ --batch-size 1 --data-set camelyon16 --bce-loss --slides --wsi_method attn --lr 0.001 --unscale-lr --device {device}',
        f'python ./projects/fm_histopathology/scripts/models/deit/main.py --output_dir ./projects/fm_histopathology/models/camelyon_16_attn_pretrained_mocov3_tiny_checkpoint --resume ./projects/fm_histopathology/models/mocov3_tiny_checkpoint/mocov3_deit_vit_small.pth --model deit_small_patch16_224 --data-path ../../../../../../../../mnt/data/nfs03-R6/CAMELYON16/ --batch-size 1 --data-set camelyon16 --bce-loss --slides --wsi_method attn --lr 0.001 --unscale-lr --device {device}',
    ]

    all_commands = {"wss": wss_commands, "patch_cam": patch_cam_commands, "mhist": mhist_commands, "crc": crc_commands, "camelyon_16_pca": camelyon16_pca_commands, "camelyon_16_attn": camelyon16_attn_commands}
    commands = all_commands[data]

    # Open the output file and run each command
    with open(output_file, 'w') as f:
        for cmd in commands:
            try:
                # Execute each command
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                # Write the output and errors to the file
                f.write(f"Running command: {cmd}\n")
                f.write(f"Output:\n{result.stdout}\n")
                if result.stderr:
                    f.write(f"Errors:\n{result.stderr}\n")
                f.write("-" * 40 + "\n")
            except Exception as e:
                f.write(f"Error running command {cmd}: {e}\n")
                f.write("-" * 40 + "\n")

def get_args_parser():
    parser = argparse.ArgumentParser('Models train script', add_help=False)

    parser.add_argument('--data', default="", type=str)
    parser.add_argument('--device', default="", type=str)
    
    return parser

def main(args):
    print(args)

    sys.path[0] = "/home/ge26xaj/./projects/fm_histopathology/scripts/"
    print("In module products sys.path[0], __package__ ==", sys.path[0], __package__)

    # The output file where the prints and errors will be written
    output_file = f'./projects/fm_histopathology/scripts/outputs/{args.data}_models_train_output.txt'

    # Call the function to run slide_matching.py multiple times
    run_models_train_with_data(args.data, args.device, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Models script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
