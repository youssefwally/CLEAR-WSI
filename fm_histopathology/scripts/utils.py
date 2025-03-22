from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn.functional as F
import math
import numpy as np


def cos_sklearn(x, y):
    return cosine_similarity(x, y)


def cos_torch(x, y):
    return F.cosine_similarity(x, y)


def cos_mul(x, y):
    x_norm = F.normalize(x, p=2, dim=1)
    y_norm = F.normalize(y, p=2, dim=1)

    return torch.matmul(x_norm, y_norm.T)

def distance(bob, binary_bob_embeddings):
    '''
    Function to compute the distance between two BoBs
    '''
    # Initialize the total distance
    total_dist = []
    for feat in binary_bob_embeddings:
        # Compute the distance to all barcodes in the other BoB
        distance = [butil.count_xor(bob[2], feat[2])]
        # Append the minimum distance
        total_dist.append([feat[0], feat[1], np.min(distance)])
        # total_dist.append(np.min(distances))
    # Return the median distance
    for item in total_dist:
        item[2] = np.median(item[2])

    retval = total_dist
    # Return the median distance
    return retval

def get_search_image_index(data, search_id, nested=True):
    search_id = str(search_id)
    for i, item in enumerate(data):
        if nested:
            if str(item[0]) == search_id:
                return i
        else:
            if str(item) == search_id:
                return i

def yottixel_search(embeddings, k, image_id):
    bob_embeddings = []
    binary_bob_embeddings = []
    for embedding in embeddings:
        bob_embedding = (np.diff(np.array(embedding[2]), axis=1) < 0) * 1
        bob_embeddings.append([embedding[0], embedding[1], bob_embedding])

    for item in bob_embeddings:
        binary_bob_embeddings.append([item[0], item[1], bitarray.bitarray(item[2][0].tolist())])

    # Compute the distances
    # search_index = get_search_image_index(all_ids, image_id, False) #not needed in patchcam
    search_index = 0
    distances = [distance(binary_bob_embeddings[search_index], binary_bob_embeddings)]
    distances[0].sort(key=lambda row: row[2], reverse=False)
    get_topk_stats(distances[0], k)
    return distances[0]


def yottixel_slide_search(embeddings, k, image_id):
    bob_embeddings = []
    binary_bob_embeddings = []
    for embedding in embeddings:
        bob_embedding = (np.diff(np.array(embedding[2]), axis=1) < 0) * 1
        bob_embeddings.append([embedding[0], embedding[1], bob_embedding])

    for item in bob_embeddings:
        binary_bob_embeddings.append([item[0], item[1], bitarray.bitarray(item[2][0].tolist())])

    # Compute the distances
    # search_index = get_search_image_index(all_ids, image_id, False) #not needed in patchcam
    search_index = 0
    distances = [distance(binary_bob_embeddings[search_index], binary_bob_embeddings)]
    distances_df = pd.DataFrame(distances[0], columns=("id", "target", "distance"))
    distances_df = distances_df.groupby("id").sum()
    distances_df["patient_id"] = distances_df.index
    distances_df = distances_df[["patient_id", "target", "distance"]]

    distances_df["target"] = [target > torch.tensor([0, 0, 0]) for target in distances_df["target"]]
    distances_df["target"] = [[1 if label == torch.tensor(True) else 0 for label in target] for target in
                              distances_df["target"]]

    distances_np = distances_df.to_numpy()
    distances_list = distances_np.tolist()

    # search_index = get_search_image_index(all_ids, image_id, False) #not needed in patchcam
    search_index = 0
    search_image = distances_list[search_index]
    distances_list.sort(key=lambda row: row[2], reverse=False)
    distances_list.remove(search_image)
    distances_list.insert(0, search_image)

    get_topk_stats(distances_list, k)
    return distances_list


def get_topk_stats(search_image,sorted_sim_vector,sorted_all_targets,sorted_all_ids, k):

    relavent = 0.01
    dcg = 0.01
    idcg = 0.01

    for i in range(k):
        idcg = idcg + 1 / math.log2((i + 1) + 1)
    for i in range(k):
        if sorted_all_targets[i] == search_image[1]:
            relavent += 1
            dcg = dcg + 1 / (math.log2((i + 1) + 1))
        else:
            dcg = dcg + 0 / (math.log2((i + 1) + 1))

    percision = relavent / (k)
    recall = relavent / (k)
    ndcg = dcg / idcg
    similarity_f1_score = 2 * ((percision * recall) / (percision + recall))

    metrics = {"percision": percision, "recall": recall, "NDCG": ndcg, "f1": similarity_f1_score}

    return metrics
