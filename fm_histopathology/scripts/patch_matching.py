from utils import cos_mul,get_topk_stats
import torch
import torch.nn.functional as F
import argparse
from tqdm import tqdm




@torch.no_grad()
def similarity_single_patch(k,search_index, sim_matrix,label_vector):
    sim_vector = sim_matrix[search_index][:]
    all_targets = label_vector
    all_ids = torch.tensor(list(range(0, len(all_targets))),device=sim_matrix.device,dtype=torch.int64)


    # remebmer the search_images attibutes
    search_image = [all_ids[search_index], all_targets[search_index], sim_vector[search_index]]

    #exclude the idx with value search_index from all the all_ids, all_targets, sim_vector
    sim_vector = torch.cat([sim_vector[0:search_index], sim_vector[search_index + 1:]])
    all_targets = torch.cat([all_targets[0:search_index], all_targets[search_index + 1:]])
    all_ids = torch.cat([all_ids[0:search_index], all_ids[search_index + 1:]])

    #order the all_targets, all_ids according to sim_vector from highest to lowest
    sorted_indices = torch.argsort(sim_vector, descending=True)
    sorted_sim_vector = sim_vector[sorted_indices]
    sorted_all_targets = all_targets[sorted_indices]
    sorted_all_ids = all_ids[sorted_indices]




    #
    #similarities.append([all_ids[i], all_targets[i], sim_vector[i]] for i in range(0, len(all_targets)))
    #similarities.sort(key=sim_vector, reverse=True)

    #similarities.sort(key=lambda sim_vector: sim_vector, reverse=True)

    metrics = get_topk_stats(search_image,sorted_sim_vector,sorted_all_targets,sorted_all_ids, k)
    return metrics

def similarity_dataset(k, sim_matrix,labels):
    total_metrics = {"percision": 0.0, "recall": 0.0, "NDCG": 0.0, "f1": 0.0}
    count = sim_matrix.shape[0]
    #count = 1
    # convert both sim_matrix and labels to cpu
    sim_matrix = sim_matrix.cpu()
    labels = labels.cpu()
    for i in tqdm(range(0, count)):
        # if i % 100 == 0:
        #     print(i)
        metrics = similarity_single_patch(k, i,sim_matrix,labels)

        total_metrics["percision"] = total_metrics["percision"] + metrics["percision"]
        total_metrics["recall"] = total_metrics["recall"] + metrics["recall"]
        total_metrics["NDCG"] = total_metrics["NDCG"] + metrics["NDCG"]
        total_metrics["f1"] = total_metrics["f1"] + metrics["f1"]

    print({total_metrics['f1'] / count}, {total_metrics['percision'] / count},
          {total_metrics['recall'] / count}, {total_metrics['NDCG'] / count})

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference script to obtain matching metrics')
    parser.add_argument('--model', default='Dino_tum', type=str, help='Path to the model file')
    parser.add_argument('--data', default='Patch_cam', type=str, choices=['MHIST', 'Patch_cam','CRC_norm','CCRCC'],help='Path to the test data file')
    args = parser.parse_args()

    #first step: load the embeddings, labels and ids
    feats_dir = '../feats/'
    labels_dir = '../labels/'
    if args.model == "Dino_tum":
        model_suffix = "Dino_manual_76400"
    elif args.model == "UNI":
        model_suffix = "UNI"
    else:
        #raise error
        raise ValueError(
            f"Model {args.model} not implemented. Please choose from [Dino_tum, UNI]")
    if args.data == "MHIST":
        data_suffix = "MHIST"
    elif args.data == "Patch_cam":
        data_suffix = "Patch_cam"
    else:
        data_suffix = args.data

    emds = torch.load(f"{feats_dir}{data_suffix}_{model_suffix}_test.pth")
    labels = torch.load(f"{labels_dir}{data_suffix}_{model_suffix}_test.pth")
    #print(emds.shape)

    #second step: calculate the similarity matrix
    sim_matrix = cos_mul(emds,emds) #N,D
    #print(sim_matrix.shape) #N,N
    #third step: calculate the similarities
    similarity_dataset(k=5,sim_matrix=sim_matrix,labels=labels)
    #fourth step: collect the metrics for all patches
    # {0.8896770019533103} {0.8896770019533103} {0.8896770019533103} {0.8941820305180002} with default Dino_tum and Patch_cam
