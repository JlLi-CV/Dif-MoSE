import numpy as np
from scipy.io import loadmat
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity



def compute_cosine_similarity(bands):
    min_vals = np.min(bands, axis=(0, 1), keepdims=True)
    max_vals = np.max(bands, axis=(0, 1), keepdims=True)

    ranges = max_vals - min_vals
    ranges[ranges < 1e-10] = 1.0

    bands_normalized = (bands - min_vals) / ranges
    bands_T = bands_normalized.T

    cos_sim_matrix = cosine_similarity(bands_T)

    cos_sim_matrix = np.clip(cos_sim_matrix, -1.0, 1.0)
    return cos_sim_matrix


def calculate_metrics(S, split_points):
    K = len(split_points) - 1
    intra_similarity = 0.0
    inter_similarity = 0.0

    for g in range(K):
        start, end = split_points[g], split_points[g + 1]
        if end - start <= 1:
            continue

        group_size = end - start
        sub_matrix = S[start:end, start:end].copy()
        np.fill_diagonal(sub_matrix, 0)  # Exclude self-similarity

        # Total valid elements (excluding diagonal)
        valid_elements = group_size * (group_size - 1)
        if valid_elements > 0:
            intra_similarity += np.sum(sub_matrix) / valid_elements

    # Calculate between-group similarity
    group_pairs = 0
    for g in range(K):
        for h in range(g + 1, K):
            g_start, g_end = split_points[g], split_points[g + 1]
            h_start, h_end = split_points[h], split_points[h + 1]

            sub_matrix = S[g_start:g_end, h_start:h_end]
            inter_similarity += np.mean(sub_matrix)
            group_pairs += 1

    # Average over all groups/pairs
    intra_similarity /= K if K > 0 else 1
    inter_similarity /= group_pairs if group_pairs > 0 else 1

    return intra_similarity, inter_similarity


def optimize_partition_with_cosine(bands, K, lambda_=0.5, max_iter=10, tolerance=1e-6):

    L = bands.shape[1]

    if K <= 0 or K > L:
        raise ValueError(f"The number of groups K must be between 1 and {L}")


    cos_sim_matrix = compute_cosine_similarity(bands)

    initial_split = np.linspace(0, L, K + 1, dtype=int)


    kmeans = KMeans(n_clusters=K, random_state=42).fit(cos_sim_matrix)
    cluster_centers = kmeans.cluster_centers_

    band_distances = np.zeros((L, K))
    for k in range(K):
        band_distances[:, k] = np.linalg.norm(cos_sim_matrix - cluster_centers[k], axis=1)

    representative_bands = np.argmin(band_distances, axis=0)
    print('Representative band indices:', representative_bands)
    representative_bands = np.sort(representative_bands)
    print('Initial equally spaced split points:', initial_split)


    split_points = np.unique(np.concatenate([initial_split, representative_bands, [0, L]]))
    split_points = np.sort(split_points)


    while len(split_points) < K + 1:

        intervals = np.diff(split_points)
        max_interval_idx = np.argmax(intervals)
        new_point = (split_points[max_interval_idx] + split_points[max_interval_idx + 1]) // 2
        split_points = np.append(split_points, new_point)
        split_points = np.sort(split_points)

    split_points = np.clip(split_points, 0, L)
    print('Merged split points:', split_points)


    split_points = np.unique(split_points)

    while len(split_points) > K + 1:
        intervals = np.diff(split_points)
        min_interval_idx = np.argmin(intervals)
        split_points = np.delete(split_points, min_interval_idx + 1)


    best_split = split_points.copy()
    print('Final initial split points:', best_split)
    best_intra, best_inter = calculate_metrics(cos_sim_matrix, split_points)
    best_score = best_intra - lambda_ * best_inter

    history = []
    converged = False

    for iteration in range(max_iter):
        improved = False

        for k in range(1, K):
            current_split = split_points.copy()
            original_score = best_score

            search_radius = min(10, L // 10)
            search_start = max(split_points[k - 1] + 1, split_points[k] - search_radius)
            search_end = min(split_points[k + 1] - 1, split_points[k] + search_radius)

            for s in range(search_start, search_end + 1):
                if s == split_points[k]:
                    continue

                current_split[k] = s


                intra, inter = calculate_metrics(cos_sim_matrix, current_split)
                score = intra - lambda_ * inter


                history.append({
                    'iteration': iteration,
                    'split_point_idx': k,
                    'new_position': s,
                    'intra': intra,
                    'inter': inter,
                    'score': score
                })


                if score > best_score:
                    best_score = score
                    best_intra = intra
                    best_inter = inter
                    best_split = current_split.copy()
                    improved = True

            if improved:
                split_points = best_split.copy()


        if not improved:
            converged = True
            break


        if iteration % 10 == 0:
            print(f"Iteration {iteration}: Best Score = {best_score:.4f}, "
                  f"Intra = {best_intra:.4f}, Inter = {best_inter:.4f}")

    print(f"Optimization completed, total iterations: {iteration + 1}")
    print(f"Final score: {best_score:.4f}, Within-group similarity: {best_intra:.4f}, "
          f"Between-group similarity: {best_inter:.4f}")
    print(f"Final split points: {best_split}")

    return best_split, history, cos_sim_matrix


if __name__ == "__main__":

    mat_data = loadmat('')

    bands = mat_data['paviaU']
    print(bands.shape)

    data = bands.reshape(-1, bands.shape[2])

    K = 3


    split_points, history, cos_sim_matrix = optimize_partition_with_cosine(
        data, K
    )

    for i in range(K):
        start, end = split_points[i], split_points[i + 1]
        print(f"Group {i + 1}: Bands {start} to {end - 1} (total {end - start} bands)")