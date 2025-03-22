import subprocess
import sys
import argparse

def run_models_test_with_data(data, output_file):
    # Define the commands with the [data] placeholder
    commands = [
        f'python ./models_test.py --model_name {data}_deit_tiny_patch16_224',
        f'python ./models_test.py --model_name {data}_general_pretrained_deit_tiny_patch16_224',
        f'python ./models_test.py --model_name {data}_pretrained_mocov3_tiny_checkpoint',
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
    parser = argparse.ArgumentParser('Models train script', add_help=False)

    parser.add_argument('--data', default="", type=str)
    
    return parser

def main(args):
    print(args)

    sys.path[0] = "/home/ge26xaj/./projects/fm_histopathology/scripts/"
    print("In module products sys.path[0], __package__ ==", sys.path[0], __package__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Models test script', parents=[get_args_parser()])
    args = parser.parse_args()

    # The output file where the prints and errors will be written
    output_file = f'./outputs/{args.data}_models_test_output.txt'

    # Call the function to run slide_matching.py multiple times
    run_models_test_with_data(args.data, output_file)
