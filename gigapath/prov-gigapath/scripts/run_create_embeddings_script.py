import subprocess
import sys

def run_create_gigapath_embeddings():
    commands = [
        "python create_embeddings.py --tiles-dir ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_tiles/train/Group_MT/Type_DCIS/output/ --save-dir ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_embeddings/train/Group_MT/Type_DCIS/",
        "python create_embeddings.py --tiles-dir ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_tiles/train/Group_MT/Type_IC/output/ --save-dir   ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_embeddings/train/Group_MT/Type_IC/",
        "python create_embeddings.py --tiles-dir ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_tiles/train/Group_AT/Type_FEA/output/ --save-dir  ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_embeddings/train/Group_AT/Type_FEA/",
        "python create_embeddings.py --tiles-dir ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_tiles/train/Group_AT/Type_ADH/output/ --save-dir  ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_embeddings/train/Group_AT/Type_ADH/",
        "python create_embeddings.py --tiles-dir ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_tiles/train/Group_BT/Type_N/output/ --save-dir    ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_embeddings/train/Group_BT/Type_N/",
        "python create_embeddings.py --tiles-dir ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_tiles/train/Group_BT/Type_PB/output/ --save-dir   ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_embeddings/train/Group_BT/Type_PB/",
        "python create_embeddings.py --tiles-dir ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_tiles/train/Group_BT/Type_UDH/output/ --save-dir  ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_embeddings/train/Group_BT/Type_UDH/",
        "python create_embeddings.py --tiles-dir ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_tiles/val/Group_MT/Type_DCIS/output/ --save-dir   ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_embeddings/val/Group_MT/Type_DCIS/",
        "python create_embeddings.py --tiles-dir ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_tiles/val/Group_MT/Type_IC/output/ --save-dir     ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_embeddings/val/Group_MT/Type_IC/",
        "python create_embeddings.py --tiles-dir ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_tiles/val/Group_AT/Type_FEA/output/ --save-dir    ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_embeddings/val/Group_AT/Type_FEA/",
        "python create_embeddings.py --tiles-dir ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_tiles/val/Group_AT/Type_ADH/output/ --save-dir    ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_embeddings/val/Group_AT/Type_ADH/",
        "python create_embeddings.py --tiles-dir ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_tiles/val/Group_BT/Type_N/output/ --save-dir      ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_embeddings/val/Group_BT/Type_N/",
        "python create_embeddings.py --tiles-dir ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_tiles/val/Group_BT/Type_PB/output/ --save-dir     ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_embeddings/val/Group_BT/Type_PB/",
        "python create_embeddings.py --tiles-dir ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_tiles/val/Group_BT/Type_UDH/output/ --save-dir    ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_embeddings/val/Group_BT/Type_UDH/",
        "python create_embeddings.py --tiles-dir ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_tiles/test/Group_MT/Type_DCIS/output/ --save-dir  ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_embeddings/test/Group_MT/Type_DCIS/",
        "python create_embeddings.py --tiles-dir ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_tiles/test/Group_MT/Type_IC/output/ --save-dir    ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_embeddings/test/Group_MT/Type_IC/",
        "python create_embeddings.py --tiles-dir ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_tiles/test/Group_AT/Type_FEA/output/ --save-dir   ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_embeddings/test/Group_AT/Type_FEA/",
        "python create_embeddings.py --tiles-dir ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_tiles/test/Group_AT/Type_ADH/output/ --save-dir   ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_embeddings/test/Group_AT/Type_ADH/",
        "python create_embeddings.py --tiles-dir ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_tiles/test/Group_BT/Type_N/output/ --save-dir     ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_embeddings/test/Group_BT/Type_N/",
        "python create_embeddings.py --tiles-dir ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_tiles/test/Group_BT/Type_PB/output/ --save-dir    ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_embeddings/test/Group_BT/Type_PB/",
        "python create_embeddings.py --tiles-dir ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_tiles/test/Group_BT/Type_UDH/output/ --save-dir   ../../../../../../../../../mnt/nfs03-R6/BRACS/gigapath_embeddings/test/Group_BT/Type_UDH/"
        ]
    
    for cmd in commands:
        try:
            # Execute each command
            _ = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        except Exception as e:
            print(f"Error running command {cmd}: {e}\n")
            print("-" * 40 + "\n")

    return True

# def get_args_parser():
    # parser = argparse.ArgumentParser('Similarity test script', add_help=False)

    # parser.add_argument('--tile', default="", type=str)
    # parser.add_argument('--wsi', action='store_true')
    # parser.add_argument('--label-test', action='store_true')
    
    # return parser

def main(args):
    print(args)

    sys.path[0] = "/home/ge26xaj/./projects/fm_histopathology/scripts/"
    print("In module products sys.path[0], __package__ ==", sys.path[0], __package__)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser('Similarity test script', parents=[get_args_parser()])
    # args = parser.parse_args()
    
    run_create_gigapath_embeddings()