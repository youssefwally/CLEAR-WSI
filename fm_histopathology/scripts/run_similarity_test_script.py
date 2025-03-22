import subprocess
import sys
import argparse

# python run_similarity_test_script.py --data camelyon_16 --wsi
# python run_similarity_test_script.py --data camelyon_16 --wsi --label_test

def run_slide_matching_with_data(data, wsi, label_test, output_file):
    # Define the commands with the [data] placeholder
    if wsi:
        if label_test:
            commands = [
                # f'python ./slide_matching.py --model-name {data}_attn_deit_tiny_patch16_224 --data {data} --wsi --agg attn --label_filter --filter-type pred',
                # f'python ./slide_matching.py --model-name {data}_attn_deit_tiny_patch16_224 --data {data} --wsi --agg attn --label_filter --filter-type mv',
                # f'python ./slide_matching.py --model-name {data}_attn_deit_tiny_patch16_224 --data {data} --wsi --agg attn --label_filter --filter-type gt',

                f'python ./slide_matching.py --model-name {data}_lvl_2_uni_attn --data {data} --agg attn --wsi --label_filter --filter-type pred --uni',
                f'python ./slide_matching.py --model-name {data}_lvl_2_uni_attn --data {data} --agg attn --wsi --label_filter --filter-type mv --uni',
                f'python ./slide_matching.py --model-name {data}_lvl_2_uni_attn --data {data} --agg attn --wsi --label_filter  --filter-type gt --uni',

                # f'python ./slide_matching.py --model-name {data}_full_wattn_uni --data {data} --wsi --agg attn --label_filter --filter-type pred',
                # f'python ./slide_matching.py --model-name {data}_full_wattn_uni --data {data} --wsi --agg attn --label_filter --filter-type mv',
                # f'python ./slide_matching.py --model-name {data}_full_wattn_uni --data {data} --wsi --agg attn --label_filter --filter-type gt',
            ]
        else:
            commands = [
                # f'python ./slide_matching.py --model-name {data}_pca_deit_tiny_patch16_224 --data {data} --wsi --agg pca',
                # f'python ./slide_matching.py --model-name {data}_attn_deit_tiny_patch16_224 --data {data} --wsi --agg attn',
                # f'python ./slide_matching.py --model-name {data}_pca_deit_tiny_patch16_224 --data {data} --wsi --agg pca --label_filter',
                # f'python ./slide_matching.py --model-name {data}_attn_deit_tiny_patch16_224 --data {data} --wsi --agg attn --label_filter',

                # f'python ./slide_matching.py --model-name {data}_pca_general_pretrained_deit_tiny_patch16_224 --data {data} --wsi --agg pca',
                f'python ./slide_matching.py --model-name {data}_lvl_2_uni_attn --data {data} --agg attn --wsi --uni',
                # f'python ./slide_matching.py --model-name {data}_pca_general_pretrained_deit_tiny_patch16_224 --data {data} --wsi --agg pca --label_filter',
                # f'python ./slide_matching.py --model-name {data}_full_wattn_general_pretrained_deit_tiny_patch16_224 --data {data} --wsi --agg attn --label_filter',

                # f'python ./slide_matching.py --model-name {data}_pca_pretrained_mocov3_tiny_checkpoint --data {data} --wsi --agg pca',
                # f'python ./slide_matching.py --model-name {data}_full_wattn_pretrained_mocov3_tiny_checkpoint --data {data} --wsi --agg attn',
                # f'python ./slide_matching.py --model-name {data}_pca_pretrained_mocov3_tiny_checkpoint --data {data} --wsi --agg pca --label_filter',
                # f'python ./slide_matching.py --model-name {data}_full_wattn_pretrained_mocov3_tiny_checkpoint --data {data} --wsi --agg attn --label_filter',
            ]
    elif label_test:
        commands = [
        #     f'python ./slide_matching.py --model-name {data}_deit_tiny_patch16_224 --data {data} --label_filter --filter-type pred',
            # f'python ./slide_matching.py --model-name {data}_deit_tiny_patch16_224 --data {data} --label_filter --filter-type mv',
            # f'python ./slide_matching.py --model-name {data}_deit_tiny_patch16_224 --data {data} --label_filter --filter-type gt',

            f'python ./slide_matching.py --model-name {data}_general_pretrained_deit_tiny_patch16_224 --data {data} --label_filter --filter-type pred',
            f'python ./slide_matching.py --model-name {data}_general_pretrained_deit_tiny_patch16_224 --data {data} --label_filter --filter-type mv',
            f'python ./slide_matching.py --model-name {data}_general_pretrained_deit_tiny_patch16_224 --data {data} --label_filter  --filter-type gt',

            f'python ./slide_matching.py --model-name {data}_pretrained_mocov3_tiny_checkpoint --data {data} --label_filter --filter-type pred,'
            f'python ./slide_matching.py --model-name {data}_pretrained_mocov3_tiny_checkpoint --data {data} --label_filter --filter-type mv',
            f'python ./slide_matching.py --model-name {data}_pretrained_mocov3_tiny_checkpoint --data {data} --label_filter --filter-type gt',
        ]
    else:
        commands = [
            # f'python ./slide_matching.py --model-name deit_tiny_patch16_224 --data {data} --scratch True',
            f'python ./slide_matching.py --model-name {data}_deit_tiny_patch16_224 --data {data}',
            f'python ./slide_matching.py --model-name {data}_deit_tiny_patch16_224 --data {data} --label_filter',

            f'python ./slide_matching.py --model-name deit_tiny_patch16_224 --data {data} --scratch https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth',
            f'python ./slide_matching.py --model-name {data}_general_pretrained_deit_tiny_patch16_224 --data {data}',
            f'python ./slide_matching.py --model-name {data}_general_pretrained_deit_tiny_patch16_224 --data {data} --label_filter',

            f'python ./slide_matching.py --model-name deit_small_patch16_224 --data {data} --scratch ../models/mocov3_tiny_checkpoint/mocov3_deit_vit_small.pth',
            f'python ./slide_matching.py --model-name {data}_pretrained_mocov3_tiny_checkpoint --data {data}',
            f'python ./slide_matching.py --model-name {data}_pretrained_mocov3_tiny_checkpoint --data {data} --label_filter'
        ]

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
    parser = argparse.ArgumentParser('Similarity test script', add_help=False)

    parser.add_argument('--data', default="", type=str)
    parser.add_argument('--wsi', action='store_true')
    parser.add_argument('--label-test', action='store_true')
    
    return parser

def main(args):
    print(args)

    sys.path[0] = "/home/ge26xaj/./projects/fm_histopathology/scripts/"
    print("In module products sys.path[0], __package__ ==", sys.path[0], __package__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Similarity test script', parents=[get_args_parser()])
    args = parser.parse_args()

    # The output file where the prints and errors will be written
    if args.label_test:
        output_file = f'./outputs/{args.data}_lvl_2_uni_label_test_slide_matching_output.txt'
    else:
        output_file = f'./outputs/{args.data}_lvl_2_uni_slide_matching_output.txt'

    # Call the function to run slide_matching.py multiple times
    run_slide_matching_with_data(args.data, args.wsi, args.label_test, output_file)