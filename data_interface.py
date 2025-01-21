# -*- coding: utf8 -*
import sys

# system lib
import time
# import random

# third part lib
import pandas as pd
import numpy as np
import torch
from numba import njit

SEP_SIGN = "*" * 100
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


def generate_simulation_data(args):
    # args.llr1RowLen = int(args.rowLen * (args.llr1RowLen / 1000))
    # args.llr1ColLen = args.llr1RowLen
    # args.colLen = args.rowLen
    # seed = 16
    add_mean_time = args.add_mean_time / 10
    add_error_time = args.add_error_time / 10

    col_vec = None
    row_vec = None
    if args.seed_col != -1 and args.seed_row != -1:
        np.random.seed(args.seed_col)
        col_vec = np.random.rand(args.n_row_pattern, args.lk).reshape(
            (args.n_row_pattern, args.lk)
        )
        np.random.seed(args.seed_row)
        row_vec = np.random.rand(args.lk, args.n_row_pattern).reshape(
            (args.lk, args.n_row_pattern)
        )
    else:
        col_vec = np.random.rand(args.n_row_pattern, args.lk).reshape(
            (args.n_row_pattern, args.lk)
        )
        row_vec = np.random.rand(args.lk, args.n_row_pattern).reshape(
            (args.lk, args.n_row_pattern)
        )
    lk_matrix = np.matmul(col_vec, row_vec)
    lk_matrix_std = np.std(lk_matrix)
    lk_matrix_std = lk_matrix_std if 0 < lk_matrix_std < 1 else 0.22

    # normalize the data by "- row mean / row std"
    lk_matrix = normalize_matrix(lk_matrix)

    # generate the error
    # N(0,a*std)
    background_noise = np.random.normal(
        loc=0, scale=(lk_matrix_std), size=(args.n_row_data, args.n_col_data)
    )
    # background_noise=normalize_matrix(background_noise)

    # N(0,std)
    error_4_pattern = np.random.normal(
        loc=0,
        scale=(add_error_time * lk_matrix_std),
        size=(args.n_row_pattern, args.n_col_pattern),
    )

    mean_4_pattern = add_mean_time * lk_matrix_std

    lk_matrix += error_4_pattern + mean_4_pattern

    background_noise[: args.n_row_pattern, : args.n_col_pattern] = lk_matrix

    data = pd.DataFrame(background_noise)

    data.index = data.index.map(lambda x: "rowId" + str(x))
    data.columns = data.columns.map(lambda x: "colId" + str(x))

    return data


def load_data(args):
    print(args.SEP_SIGN)
    print("Loading data...")

    file_name = [
        "mtrxSize",
        str(args.n_row_data),
        "_",
        str(args.n_col_data),
        "_llr1Size",
        str(args.n_row_pattern),
        "_",
        str(args.n_col_pattern),
        "_mean",
        str(args.add_mean_time),
        "sd_error",
        str(args.add_error_time),
        "sd_noShffule.csv.gz",
    ]
    file_name = "".join(file_name)
    read_path = args.project_dir + args.data_dir + args.data_source + file_name

    data = pd.read_csv(read_path, index_col=0, compression="gzip")

    print(args.SEP_SIGN)
    print("\nData Sample:\n{0}\n".format(data.iloc[:5, :5]))
    print(args.SEP_SIGN)
    print("Load Data Done!")
    print(args.SEP_SIGN)
    return data


def shuffle_data(data):
    # shuffle the data by whole row and then whole col
    from sklearn.utils import shuffle

    print("Shuffling data...")
    data = shuffle(data)
    data = shuffle(data.T)
    data = data.T

    SEP_SIGN = "*" * 100
    print(SEP_SIGN)
    print("\nShuffled Data Sample:\n{0}\n".format(data.iloc[:5, :5]))
    print(SEP_SIGN)
    print("\nshuffle Data Done!\n")
    print(SEP_SIGN)

    return data


def normalize_matrix(data, by="row"):
    row_mean = np.mean(data, axis=1).reshape(data.shape[0], 1)
    row_std = np.std(data, axis=1).reshape(data.shape[0], 1)
    row_std[row_std == 0] = 1
    row_std[row_std == None] = 1
    row_std[row_std > 10000] = 1
    data = (data - row_mean) / row_std
    return data


def normalize_dataframe(data, by="row"):
    row_mean = np.mean(data.values, axis=1).reshape(data.shape[0], 1)
    row_std = np.std(data.values, axis=1).reshape(data.shape[0], 1)
    row_std[row_std == 0] = 1
    row_std[row_std == None] = 1
    row_std[row_std > 10000] = 1
    data = (data - row_mean) / row_std
    return data


def normalize_data(samples_3d, by="row"):
    row_mean = torch.mean(samples_3d, dim=(2), keepdim=True)
    row_std = torch.std(samples_3d, dim=(2), keepdim=True)
    row_std[row_std == 0] = 1
    samples_3d = (samples_3d - row_mean) / row_std
    return samples_3d


@njit
def get_samples_jit(data, rowIdx_colIdx):
    samples_3d = []
    for rowIdx, colIdx in rowIdx_colIdx:
        sample_i = data[rowIdx, :]
        sample_i = sample_i[:, colIdx]
        samples_3d.append(sample_i)
    return samples_3d


def random_sampling_slow(q_dim, n_sample, data):
    print("Random Sampling...")
    import random

    time_s1 = time.time()

    n_row_data, n_col_data = data.shape
    all_row_idx = list(range(n_row_data))
    # all_col_idx = list(range(n_col_data))
    rowIdx_colIdx = np.array(
        [
            [random.sample(all_row_idx, q_dim), random.sample(all_row_idx, q_dim)]
            for _ in range(n_sample)
        ]
    )

    time_s2 = time.time()

    print("\nSampling row col idx time:{0}".format(time_s2 - time_s1))

    samples_3d = get_samples_jit(data, np.array(rowIdx_colIdx))
    samples_3d = np.stack(samples_3d)
    samples_3d = torch.from_numpy(samples_3d).to(torch.float32)

    time_e = time.time()

    print("\n The extract sample time is:{0}\n".format(time_e - time_s2))

    print(SEP_SIGN)
    print("\nRandom Row Idx & Col Idx:\n{0}\n".format(rowIdx_colIdx[:2]))
    print("\nRandom Row Idx & Col Idx shape:\n{0}\n".format(rowIdx_colIdx.shape))
    print(SEP_SIGN)
    sample_i = np.random.randint(0, len(samples_3d), 1)
    print("Sample id: {0}:\n{1}\n".format(sample_i, samples_3d[sample_i]))
    print(SEP_SIGN)
    print("\nSample Shape:{0}\n".format(samples_3d.shape))
    print("Random Sampling Done!")
    print(SEP_SIGN)

    """
    samples_3d = normalize_data(samples_3d)
    print("Normalized Sample id: {0}:\n{1}\n".format(sample_i, samples_3d[sample_i]))
    print(SEP_SIGN)
    print("\nNormalized Sample Shape:{0}\n".format(samples_3d.shape))
    print(SEP_SIGN)
    print("\n Normalized Sample id: {0}: row mean:\n{1}\n".format(sample_i, torch.mean(samples_3d[sample_i], dim=1)))
    print(SEP_SIGN)
    """

    return rowIdx_colIdx, samples_3d


def random_sampling(q_dim, qDim_nSamples, data_flatten, args):
    print("Random Sampling...")
    time_s = time.time()

    n_sample = qDim_nSamples[q_dim]
    # n_row_data, n_col_data = data.shape
    vec_ids = np.random.choice(
        len(data_flatten), (n_sample * q_dim * q_dim), replace=True
    )
    samples_flatten = np.take(data_flatten, vec_ids)

    time_s2 = time.time()

    print("\nSampling row col idx time:{0}".format(time_s2 - time_s))

    # samples_3d=torch.round(torch.from_numpy(samples_flatten).to(torch.float32),decimals=4)
    samples_3d = torch.from_numpy(samples_flatten).to(torch.float32)

    samples_3d = samples_3d.view(n_sample, q_dim, q_dim)
    vec_ids = torch.from_numpy(vec_ids)
    vec_ids = vec_ids.view(n_sample, -1)

    time_e = time.time()
    print("\n The extract sample time is:{0}\n".format(time_e - time_s2))

    print(args.SEP_SIGN)
    print("\nRandom Row Idx & Col Idx:\n{0}\n".format(vec_ids[:3]))
    print("\nRandom Row Idx & Col Idx shape:\n{0}\n".format(vec_ids.shape))
    print(args.SEP_SIGN)
    sample_i = np.random.randint(0, len(samples_3d), 1)
    print("Sample id: {0}:\n{1}\n".format(sample_i, samples_3d[sample_i]))
    print(args.SEP_SIGN)
    print("\nSample Shape:{0}\n".format(samples_3d.shape))
    print("Random Sampling Done!")
    print(args.SEP_SIGN)

    """
    samples_3d = normalize_data(samples_3d)
    print("Normalized Sample id: {0}:\n{1}\n".format(sample_i, samples_3d[sample_i]))
    print(args.SEP_SIGN)
    print("\nNormalized Sample Shape:{0}\n".format(samples_3d.shape))
    print(args.SEP_SIGN)
    print("\n Normalized Sample id: {0}: row mean:\n{1}\n".format(sample_i,torch.mean(samples_3d[sample_i],dim=1)))
    print(args.SEP_SIGN)
    """

    return vec_ids, samples_3d


def get_number_probes(cos_val=0.9):
    qDim_nProbe = {}
    theta = np.arccos(cos_val)
    qDim_nProbe[2] = 40
    qDim_nProbe[4] = 200
    qDim_nProbe[8] = 2000
    qDim_nProbe[16] = 6000

    """
    for qi in [8,16]:
        if qi==16:
            theta=np.arccos(0.8)
        n_probe = np.exp(-qi * np.log(np.sin(theta))*(1+1/qi))
        qDim_nProbe[qi] = int(n_probe)
    """

    SEP_SIGN = "*" * 100
    print(SEP_SIGN)
    print("\n The q_dim and its number of probes:\n{0}\n".format(qDim_nProbe))
    print(SEP_SIGN)

    return qDim_nProbe


def generate_probes(q_dim, n_probe):
    from sklearn import preprocessing
    from torch import linalg as LA

    SEP_SIGN = "*" * 100
    print(SEP_SIGN)
    print("Generating probes...\n")
    probes = np.random.randn(q_dim, n_probe)
    probes = preprocessing.normalize(probes.T).T
    probes = torch.from_numpy(probes).to(torch.float32)

    probes_col_norm = LA.norm(probes, dim=0)
    if (len(probes_col_norm) != n_probe) and (not torch.all(probes_col_norm == 1)):
        print("Error!!")
        return False

    print(SEP_SIGN)
    print("\nProbes Samples:\n{0}".format(probes[:, :3]))
    print("\nProbes Number:{0}\n".format(probes.shape))
    print("Generation of probes Done!")
    print(SEP_SIGN)

    return probes


def inner_square_sum(samples_3d, probes):
    print("Inner square sum...........")

    singular_vals_3d = torch.matmul(samples_3d.to(device), probes.to(device)) ** 2
    singular_vals_3d = torch.sqrt(torch.sum(singular_vals_3d, dim=1))

    print(SEP_SIGN)
    print("\n singular_vals res:\n{0}".format(singular_vals_3d))
    print("\n singular_vals res:{0}\n".format(singular_vals_3d.shape))
    print("singular_vals est Done!")
    print(SEP_SIGN)

    return singular_vals_3d


def orthogonal_max_plooing(q_dim, singular_vals_3d):
    print("Orthogonal Max Plooing..............!")

    need_pos = {
        2: [0, -1],
        4: [0, 50, 160, -1],
        8: [0, 780, 830, 880],
        16: [0, 500, 1000, 1500],
    }

    singular_vals_3d_max = torch.sort(singular_vals_3d, dim=1, descending=True)[0][
        :, need_pos[q_dim]
    ]

    SEP_SIGN = "*" * 100
    print(SEP_SIGN)
    print("\nsingular_vals_3d max Samples:\n{0}".format(singular_vals_3d_max[:10, :]))
    print("\n The max is: {0}\n".format(singular_vals_3d_max.max()))
    print("\n The min is: {0}\n".format(singular_vals_3d_max.min()))
    print("\nsingular_vals_3d max shape:{0}\n".format(singular_vals_3d_max.shape))

    print("Orthogonal Max Plooing Done!")
    print(SEP_SIGN)

    return singular_vals_3d_max


def orthogonal_max_plooing_halfAngleThetaTest(q_dim, singular_vals_3d, qDim_nProbe):
    print("Orthogonal Max Plooing..............!")
    """
    need_pos = {2: [0, -1], 4: [0, 50, 160, -1],
                8: [0, 780, 830, 880],
                16: [0, 500, 1000, 1500]}
    """

    pos_4_2 = int(qDim_nProbe[q_dim] * (0.25))
    pos_4_3 = int(qDim_nProbe[q_dim] * (0.50))

    pos_8_2 = int(qDim_nProbe[q_dim] * (0.25))
    pos_8_3 = int(qDim_nProbe[q_dim] * (0.50))
    pos_8_4 = int(qDim_nProbe[q_dim] * (0.75))

    pos_16_2 = int(qDim_nProbe[q_dim] * (0.25))
    pos_16_3 = int(qDim_nProbe[q_dim] * (0.50))
    pos_16_4 = int(qDim_nProbe[q_dim] * (0.75))

    need_pos = {
        2: [0, -1],
        4: [0, pos_4_2, pos_4_3, -1],
        8: [0, pos_8_2, pos_8_3, pos_8_4],
        16: [0, pos_16_2, pos_16_3, pos_16_4],
    }

    singular_vals_3d_max = torch.sort(singular_vals_3d, dim=1, descending=True)[0][
        :, need_pos[q_dim]
    ]

    SEP_SIGN = "*" * 100
    print(SEP_SIGN)
    print("\nsingular_vals_3d max Samples:\n{0}".format(singular_vals_3d_max[:10, :]))
    print("\n The max is: {0}\n".format(singular_vals_3d_max.max()))
    print("\n The min is: {0}\n".format(singular_vals_3d_max.min()))
    print("\nsingular_vals_3d max shape:{0}\n".format(singular_vals_3d_max.shape))

    print("Orthogonal Max Plooing Done!")
    print(SEP_SIGN)

    return singular_vals_3d_max


@njit
def get_score_matrix_jit(q_dim, score_matrix, times_matrix, weights, rdm_rowIdx_colIdx):
    for i in range(len(rdm_rowIdx_colIdx)):
        rowIdxs, colIdxs = rdm_rowIdx_colIdx[i]
        wi = weights[i]

        """
        for rowi in rowIdxs:
            for coli in colIdxs:
                score_matrix[rowi, coli] += wi**4
                times_matrix[rowi, coli] += 1
        """

        # '''
        if (
            (q_dim == 2 and wi < 0.95)
            or (q_dim == 4 and wi < 0.8)
            or (q_dim == 8 and wi < 0.8)
            or (q_dim == 16 and wi < 0.8)
        ):
            for rowi in rowIdxs:
                for coli in colIdxs:
                    score_matrix[rowi, coli] += 0.01
                    times_matrix[rowi, coli] += 1
        else:
            for rowi in rowIdxs:
                for coli in colIdxs:
                    score_matrix[rowi, coli] += 10
                    times_matrix[rowi, coli] += 1
        # '''

    score_matrix /= times_matrix
    return score_matrix


def get_score_matrix(args, q_dim, score_matrix_old, weights, rdm_rowIdx_colIdx):
    print("Calculating score matrix..................\n")

    time_s = time.time()

    score_matrix = np.zeros((args.n_row_data, args.n_col_data))
    times_matrix = np.ones((args.n_row_data, args.n_col_data))

    if weights.is_cuda:
        weights = weights.cpu()

    score_matrix = get_score_matrix_jit(
        q_dim, score_matrix, times_matrix, weights.numpy(), np.array(rdm_rowIdx_colIdx)
    )

    if q_dim >= 4 and len(score_matrix_old) > 0:
        score_matrix = score_matrix + score_matrix_old

    time_e = time.time()

    print("The time of generation of scorematrix is : {0}".format(time_e - time_s))
    # score_matrix=pd.DataFrame(score_matrix, index=row_names,columns=col_names)
    SEP_SIGN = "*" * 100
    print(SEP_SIGN)
    print("\nScore matrix:\n{0}\n".format(np.round(score_matrix, decimals=4)))
    print("\nThe max in Score matrix:\n{0}\n".format(score_matrix.max()))
    print("\nThe min in Score matrix:\n{0}\n".format(score_matrix.min()))
    print(SEP_SIGN)

    return score_matrix


@njit
def get_weighted_samples_jit(q_dim, data, weights, sample_ids, rdm_rowIdx_colIdx):
    samples_3d = []
    rowIdx_colIdx = []
    mean_weights = weights.mean()
    for i in range(len(sample_ids)):
        id1, id2 = sample_ids[i]
        if (weights[id1] + weights[id2]) / 2 < (0.1 * mean_weights):
            continue

        row_idx = np.append(rdm_rowIdx_colIdx[id1][0], rdm_rowIdx_colIdx[id2][0])
        col_idx = np.append(rdm_rowIdx_colIdx[id1][1], rdm_rowIdx_colIdx[id2][1])
        row_idx = np.unique(row_idx)
        col_idx = np.unique(col_idx)
        if len(row_idx) != q_dim or len(col_idx) != q_dim:
            continue

        sample_i = data[row_idx, :]
        sample_i = sample_i[:, col_idx]

        samples_3d.append(sample_i)
        rowIdx_colIdx.append([row_idx, col_idx])
    return samples_3d, rowIdx_colIdx


def weighted_sampling_on_samples(q_dim, n_sample, weights, rdm_rowIdx_colIdx, data_np):
    import random

    print("Weighted sampling on samples...............")

    time_s1 = time.time()

    if weights.is_cuda:
        weights = weights.cpu()
    weights = weights.numpy()
    # print(weights)
    n_cur_samples = list(range(len(weights)))
    # sample_ids = [np.random.choice(n_cur_samples, 2, p=weights/weights.sum()) for _ in range(n_sample)]
    sample_ids = [random.sample(n_cur_samples, 2) for _ in range(int(1 * n_sample))]

    time_s2 = time.time()

    print("\nGet random sample ids time is {0}\n".format(time_s2 - time_s1))

    samples_3d, rowIdx_colIdx = get_weighted_samples_jit(
        q_dim,
        data_np,
        np.array(weights),
        np.array(sample_ids),
        np.array(rdm_rowIdx_colIdx),
    )

    samples_3d = np.stack(samples_3d)
    samples_3d = torch.from_numpy(samples_3d).to(torch.float32)

    time_s3 = time.time()

    print("\nGet random samples time is {0}\n".format(time_s3 - time_s2))

    print(SEP_SIGN)
    # print("\nRandom Row Idx & Col Idx:\n{0}\n".format(rowIdx_colIdx))
    print(SEP_SIGN)
    sample_i = np.random.randint(0, len(samples_3d), 1)
    print("Sample id: {0}:\n{1}\n".format(sample_i, samples_3d[sample_i]))
    print(SEP_SIGN)
    print("\nSample Shape:{0}\n".format(samples_3d.shape))
    print("Weighted Sampling Done!")
    print(SEP_SIGN)

    """
    samples_3d = normalize_data(samples_3d)
    print("Normalized Sample id: {0}:\n{1}\n".format(sample_i, samples_3d[sample_i]))
    print(SEP_SIGN)
    print("\nNormalized Sample Shape:{0}\n".format(samples_3d.shape))
    print(SEP_SIGN)
    print("\n Normalized Sample id: {0}: row mean:\n{1}\n".format(sample_i, torch.mean(samples_3d[sample_i], dim=1)))
    print(SEP_SIGN)
    """

    return rowIdx_colIdx, samples_3d


def weighted_sampling_on_samples_slow(
    q_dim, qDim_nSamples, weights, rdm_rowIdx_colIdx, data_np
):
    # import random

    print("Weighted sampling on samples...............")

    time_s1 = time.time()
    n_sample = qDim_nSamples[q_dim]

    weights = weights.numpy()
    # print(weights)
    n_cur_samples = list(range(len(weights)))
    sample_ids = [
        np.random.choice(n_cur_samples, 2, p=weights / weights.sum())
        for _ in range(n_sample)
    ]
    # sample_ids = [random.sample(n_cur_samples, 2) for _ in range(int(1 * n_sample))]

    time_s2 = time.time()

    print("\nGet random sample ids time is {0}\n".format(time_s2 - time_s1))

    samples_3d = []
    rowIdx_colIdx = []
    for i in range(len(sample_ids)):
        id1, id2 = sample_ids[i]
        row_idx = np.append(rdm_rowIdx_colIdx[id1][0], rdm_rowIdx_colIdx[id2][0])
        col_idx = np.append(rdm_rowIdx_colIdx[id1][1], rdm_rowIdx_colIdx[id2][1])
        row_idx = np.unique(row_idx)
        col_idx = np.unique(col_idx)
        if len(row_idx) != q_dim or len(col_idx) != q_dim:
            continue

        sample_i = data_np[row_idx, :]
        sample_i = sample_i[:, col_idx]

        samples_3d.append(sample_i)
        rowIdx_colIdx.append([row_idx, col_idx])

    # samples_3d, rowIdx_colIdx = get_weighted_samples_jit(q_dim, data_np, np.array(weights), np.array(sample_ids),
    #                                                     np.array(rdm_rowIdx_colIdx))

    samples_3d = np.stack(samples_3d)
    samples_3d = torch.from_numpy(samples_3d).to(torch.float32)

    time_s3 = time.time()

    print("\nGet random samples time is {0}\n".format(time_s3 - time_s2))

    print(SEP_SIGN)
    # print("\nRandom Row Idx & Col Idx:\n{0}\n".format(rowIdx_colIdx))
    print(SEP_SIGN)
    sample_i = np.random.randint(0, len(samples_3d), 1)
    print("Sample id: {0}:\n{1}\n".format(sample_i, samples_3d[sample_i]))
    print(SEP_SIGN)
    print("\nSample Shape:{0}\n".format(samples_3d.shape))
    print("Weighted Sampling Done!")
    print(SEP_SIGN)

    """
    samples_3d = normalize_data(samples_3d)
    print("Normalized Sample id: {0}:\n{1}\n".format(sample_i, samples_3d[sample_i]))
    print(SEP_SIGN)
    print("\nNormalized Sample Shape:{0}\n".format(samples_3d.shape))
    print(SEP_SIGN)
    print("\n Normalized Sample id: {0}: row mean:\n{1}\n".format(sample_i, torch.mean(samples_3d[sample_i], dim=1)))
    print(SEP_SIGN)
    """

    return rowIdx_colIdx, samples_3d


def perturb_weights(weights):
    mean_weights = weights.mean()
    n_weights = len(weights)
    sort_weights_idx = torch.sort(weights)[1]
    max_idxs = sort_weights_idx[: int(0.3 * n_weights)]
    min_idxs = sort_weights_idx[int(0.3 * n_weights) :]
    max_idxs = np.random.choice(max_idxs, int(0.1 * n_weights), replace=False)
    min_idxs = np.random.choice(min_idxs, int(0.1 * n_weights), replace=False)
    weights[max_idxs] = mean_weights
    weights[min_idxs] = mean_weights

    return weights


# @pysnooper.snoop()
def get_samples_weights_by_inner_sigmaRate(singular_vals_3d_max):
    print("\n Get samples' weights by sigmaRate..............")

    weights = None
    max_vals = torch.max(singular_vals_3d_max, dim=1)[0]
    sum_vals = torch.sum(singular_vals_3d_max, dim=1)
    sum_vals[sum_vals == 0] = 1
    weights = max_vals / sum_vals
    weights = weights.view(1, len(singular_vals_3d_max))[0]
    weights = weights.cpu()
    # weights=perturb_weights(weights)

    print(SEP_SIGN)
    print("\n The weights are:\n{0}\n".format(weights[:20]))
    print("\n The max weight is:{0}\n".format(weights.max()))
    print("\n The min weight is:{0}\n".format(weights.min()))
    print("\n The mean weight is:{0}\n".format(weights.mean()))
    print("\n The sum weight is:{0}\n".format(weights.sum()))

    print(SEP_SIGN)

    return weights


def get_samples_weights_by_svd_sigmaRate(samples_3d):
    print("\n Get samples' weights by svd sigmaRate..............")

    if samples_3d.is_cuda:
        samples_3d = samples_3d.cpu().copy()

    _, singular_vals_3d, _ = torch.linalg.svd(samples_3d)
    max_vals = torch.max(singular_vals_3d, dim=1)[0]

    sum_vals = None
    if singular_vals_3d.shape[1] > 2:
        sum_vals = torch.sum(singular_vals_3d[:, :4], dim=1)
    else:
        sum_vals = torch.sum(singular_vals_3d, dim=1)
    sum_vals[sum_vals == 0] = 1
    weights = max_vals / sum_vals
    weights = weights.view(1, len(singular_vals_3d))[0]
    # weights = perturb_weights(weights)

    # weights = weights.to(device)

    print(SEP_SIGN)
    print("\n The weights are:\n{0}\n".format(weights[:20]))
    print("\n The max weight is:{0}\n".format(weights.max()))
    print("\n The min weight is:{0}\n".format(weights.min()))
    print("\n The mean weight is:{0}\n".format(weights.mean()))
    print("\n The sum weight is:{0}\n".format(weights.sum()))
    print(SEP_SIGN)

    return weights


# @pysnooper.snoop()
def normalize_score_matrix(score_matrix):
    max_val = score_matrix.max()
    min_val = score_matrix.min()
    dif = max_val - min_val
    if dif == 0:
        dif = 1
    score_matrix = (score_matrix - min_val) / (dif)
    # score_matrix/=max_val
    SEP_SIGN = "*" * 100
    print(SEP_SIGN)
    print(
        "\nNormalized Score Matrix:\n{0}\n".format(np.round(score_matrix, decimals=4))
    )
    print("\nThe max in Score Matrix:\n{0}\n".format(max_val))
    print("\nThe min in Score Matrix:\n{0}\n".format(min_val))
    print(SEP_SIGN)

    return score_matrix


@njit
def get_n_point_lr_jit(rowColId, n_row_pattern, n_col_pattern):
    n_point_lr = 0
    for i in range(len(rowColId)):
        if rowColId[i][0] < n_row_pattern and rowColId[i][1] < n_col_pattern:
            n_point_lr += 1
    return n_point_lr


# @pysnooper.snoop()
def calculate_accuracy(score_matrix, row_ids, col_ids, args):
    # row_names_predicted=[]
    # col_names_predicted=[]
    # max_recall=-1
    # max_tp=-1
    n_points_all = args.n_row_data * args.n_col_data
    n_points_pattern_all = args.n_row_pattern * args.n_col_pattern
    f1s = []

    accs = []
    acc_ts = []
    for threshold_i in range(0, 101):
        cur_threshold_i = threshold_i / 100
        idx_lr = np.where(score_matrix >= cur_threshold_i)
        # idx_lr = np.where((score_matrix >= (cur_threshold_i - 50)) & (score_matrix <= cur_threshold_i))
        row_ids_lr = row_ids[idx_lr[0]]
        col_ids_lr = col_ids[idx_lr[1]]

        if len(row_ids_lr) == 0 or len(col_ids_lr) == 0:
            print("Empty Results!!")
            break

        rowColId = set(zip(row_ids_lr, col_ids_lr))
        rowColId = np.array(list(map(list, rowColId)))

        n_points_pattern_cur = get_n_point_lr_jit(
            rowColId, args.n_row_pattern, args.n_col_pattern
        )

        # recall = np.round(n_point_lr / (args.n_row_pattern * args.n_col_pattern), decimals=4)
        # tp = np.round(n_point_lr / len(row_ids_lr), decimals=4)
        """
        if recall+tp>max_recall+max_tp:
            max_recall=recall
            max_tp=tp
            row_names_predicted=row_ids_lr
            col_names_predicted=col_ids_lr
        """
        # print("\n The # of lr points:{0}".format(n_point_lr))
        # print("The # of total points:{0}\n".format(len(rowColId)))

        n_points_cur_all = len(rowColId)
        tp = n_points_pattern_cur
        tn = (
            n_points_all
            - n_points_cur_all
            - n_points_pattern_all
            + n_points_pattern_cur
        )
        fp = n_points_cur_all - n_points_pattern_cur
        fn = n_points_pattern_all - n_points_pattern_cur
        acc = 0
        if (tp + tn + fp + fn) == 0:
            acc = 0
        else:
            acc = (tp + tn) / (tp + tn + fp + fn)

        recall = n_points_pattern_cur / n_points_pattern_all
        precision = n_points_pattern_cur / n_points_cur_all
        acc_t = (recall + precision) / 2
        f1 = 0
        if precision == 0 and recall == 0:
            f1 = 0
        else:
            f1 = 2 * (recall * precision) / (recall + precision)
            f1 = np.round(f1, decimals=3)
            f1s.append(f1)
        acc_ts.append(acc_t)

        accs.append(acc)

        """
        if (tp + tn + fp + fn) == 0:
            acc = 0
        else:
            acc = (tp + tn) / (tp + tn + fp + fn)

        print("\ntp={0},tn={1},fp={2},fn={3},acc={4}\n".format(tp,tn,fp,fn,acc))
        accs.append(acc)
        """
        """
        recall = n_points_pattern_cur / n_points_pattern_all
        precision = n_points_pattern_cur / n_points_cur_all
        f1 = 0
        if precision == 0 and recall == 0:
            f1 = 0
        else:
            f1 = 2 * (recall * precision) / (recall + precision)
        f1s.append(f1)
        """
    # recall_tp = pd.DataFrame(recall_tp).T
    # recall_tp.to_csv("./recall_tp.csv")
    # print("\nThe max recall & tp is: ({0},{1})\n".format(max_recall,max_tp))

    # row_names_predicted=list(map(lambda x:"rowId"+str(x), row_names_predicted))
    # col_names_predicted=list(map(lambda x:"colId"+str(x), col_names_predicted))
    print(acc_ts)
    print(accs)
    max_acc_t_idx = np.argmax(acc_ts)
    return accs[max_acc_t_idx], max(f1s)
    # return max(f1s)


# @pysnooper.snoop()
def calculate_accuracy_f1(score_matrix, row_ids, col_ids, args):
    print("\n Calculating Accuracy and F1...........\n")

    score_matrix = score_matrix - np.mean(score_matrix)
    score_matrix = 1 / (1 + np.exp(-score_matrix))

    n_points_all = args.n_row_data * args.n_col_data
    n_points_pattern_all = args.n_row_pattern * args.n_col_pattern

    accs = []
    acc_ts = []
    f1s = []
    for _ in range(1):
        cur_threshold_i = 0.5
        idx_lr = np.where(score_matrix > cur_threshold_i)
        # idx_lr = np.where((score_matrix >= (cur_threshold_i - 50)) & (score_matrix <= cur_threshold_i))
        row_ids_lr = row_ids[idx_lr[0]]
        col_ids_lr = col_ids[idx_lr[1]]

        if len(row_ids_lr) == 0 or len(col_ids_lr) == 0:
            print("Empty Results!!")
            break

        rowColId = set(zip(row_ids_lr, col_ids_lr))
        rowColId = np.array(list(map(list, rowColId)))

        n_points_pattern_cur = get_n_point_lr_jit(
            rowColId, args.n_row_pattern, args.n_col_pattern
        )

        # recall = np.round(n_point_lr / (args.n_row_pattern * args.n_col_pattern), decimals=4)
        # tp = np.round(n_point_lr / len(row_ids_lr), decimals=4)
        """
        if recall+tp>max_recall+max_tp:
            max_recall=recall
            max_tp=tp
            row_names_predicted=row_ids_lr
            col_names_predicted=col_ids_lr
        """
        # print("\n The # of lr points:{0}".format(n_point_lr))
        # print("The # of total points:{0}\n".format(len(rowColId)))

        n_points_cur_all = len(rowColId)
        tp = n_points_pattern_cur
        tn = (
            n_points_all
            - n_points_cur_all
            - n_points_pattern_all
            + n_points_pattern_cur
        )
        fp = n_points_cur_all - n_points_pattern_cur
        fn = n_points_pattern_all - n_points_pattern_cur
        acc = 0
        if (tp + tn + fp + fn) == 0:
            acc = 0
        else:
            acc = (tp + tn) / (tp + tn + fp + fn)

        recall = n_points_pattern_cur / n_points_pattern_all
        precision = n_points_pattern_cur / n_points_cur_all
        acc_t = (recall + precision) / 2

        acc_ts.append(acc_t)

        accs.append(acc)

        if (tp + tn + fp + fn) == 0:
            acc = 0
        else:
            acc = (tp + tn) / (tp + tn + fp + fn)

        accs.append(acc)

        recall = n_points_pattern_cur / n_points_pattern_all
        precision = n_points_pattern_cur / n_points_cur_all
        f1 = 0
        if precision == 0 and recall == 0:
            f1 = 0
        else:
            f1 = 2 * (recall * precision) / (recall + precision)
            f1 = np.round(f1, decimals=3)
        f1s.append(f1)

    # recall_tp = pd.DataFrame(recall_tp).T
    # recall_tp.to_csv("./recall_tp.csv")
    # print("\nThe max recall & tp is: ({0},{1})\n".format(max_recall,max_tp))

    # row_names_predicted=list(map(lambda x:"rowId"+str(x), row_names_predicted))
    # col_names_predicted=list(map(lambda x:"colId"+str(x), col_names_predicted))
    print(acc_ts)
    print(accs)
    print(
        "\ntp={0},tn={1},fp={2},fn={3},f1={4},acc={5}\n".format(tp, tn, fp, fn, f1, acc)
    )
    max_acc_t_idx = np.argmax(acc_ts)
    return accs[max_acc_t_idx]
    # return max(f1s)


# @pysnooper.snoop()
def calculate_accuracy_on_coclustering_label(label_rowColId, args):
    # tps = []
    # recalls = []

    accs = []  # (tp+tn)/(all)
    f1s = []
    n_points_pattern_all = args.n_row_pattern * args.n_col_pattern
    n_points_all = args.n_row_data * args.n_col_data

    for i in label_rowColId.keys():
        print("\nLabel {0}:\n".format(i))

        row_ids = label_rowColId[i]["row_ids"]
        col_ids = label_rowColId[i]["col_ids"]

        n_cur_row_ids = len(row_ids)
        n_cur_col_ids = len(col_ids)

        if n_cur_col_ids == 0 or n_cur_row_ids == 0:
            break

        n_cur_row_ids_lr = len(np.where(row_ids < args.n_row_pattern)[0])
        n_cur_col_ids_lr = len(np.where(col_ids < args.n_col_pattern)[0])

        n_points_pattern_cur = n_cur_col_ids_lr * n_cur_row_ids_lr
        n_points_cur_all = n_cur_row_ids * n_cur_col_ids
        tp = n_points_pattern_cur
        tn = (
            n_points_all
            - n_points_cur_all
            - n_points_pattern_all
            + n_points_pattern_cur
        )
        fp = n_points_cur_all - n_points_pattern_cur
        fn = n_points_pattern_all - n_points_pattern_cur

        acc = 0
        if (tp + tn + fp + fn) == 0:
            acc = 0
        else:
            acc = (tp + tn) / (tp + tn + fp + fn)

        accs.append(acc)

        recall = n_points_pattern_cur / n_points_pattern_all
        precision = n_points_pattern_cur / n_points_cur_all
        f1 = 0
        if precision == 0 and recall == 0:
            f1 = 0
        else:
            f1 = 2 * (recall * precision) / (recall + precision)
            f1 = np.round(f1, decimals=3)
        f1s.append(f1)
        print(
            "\ntp={0},tn={1},fp={2},fn={3},f1={4},acc={5}\n".format(
                tp, tn, fp, fn, f1, acc
            )
        )
    return max(accs), max(f1s)
    # return max(f1s)


def save_res(args):
    res = []
    # col_names=[]
    if args.test_type == "error_test":
        res = [
            args.test_type,
            args.n_row_data,
            args.n_col_data,
            args.n_row_pattern,
            args.n_col_pattern,
            args.add_mean_time,
            args.add_error_time,
            args.n_samples_2_4,
            args.n_samples_8_16,
            args.ith_test,
            args.time_cost,
            args.accuracy_threshold,
            args.accuracy_coclustering,
        ]
        # col_names=['test_type','n_row_data','n_col_data','n_row_pattern','n_col_pattern',
        #           'mean','error','n_samples_2_4','n_samples_8_16','ith_test','time_cost',
        #           'f1']

    elif args.test_type == "mean_test":
        res = [
            args.test_type,
            args.n_row_data,
            args.n_col_data,
            args.n_row_pattern,
            args.n_col_pattern,
            args.add_error_time,
            args.add_mean_time,
            args.n_samples_2_4,
            args.n_samples_8_16,
            args.ith_test,
            args.time_cost,
            args.accuracy_threshold,
            args.accuracy_coclustering,
        ]
        # col_names = []
    elif args.test_type == "size_test" or args.test_type == "small_size_test":
        res = [
            args.test_type,
            args.n_row_data,
            args.n_col_data,
            args.add_error_time,
            args.add_mean_time,
            args.n_row_pattern,
            args.n_col_pattern,
            args.n_samples_2_4,
            args.n_samples_8_16,
            args.ith_test,
            args.time_cost,
            args.accuracy_threshold,
            args.accuracy_coclustering,
        ]
        # col_names = []

    res = pd.DataFrame(res).T

    # file_name = ['mtrxSize', str(args.n_row_data), '_', str(args.n_col_data), '_llr1Size', str(args.n_row_pattern), '_',
    #             str(args.n_col_pattern),
    #             '_mean', str(args.add_mean_time), 'sd_error', str(args.add_error_time), 'sd_noShffule.csv.gz']
    file_name = "halfAngleTheta" + args.half_angle + "_" + args.test_type + ".csv"
    save_path = args.project_dir + args.res_dir + file_name

    res.to_csv(save_path, mode="a", index=False, header=False)

    print("Save Done!!")

    return True


def save_halfAngleTheta_res(args, hard_info=None):
    res = []
    # col_names=[]
    if args.test_type == "error_test":
        res = [
            args.test_type,
            args.n_row_data,
            args.n_col_data,
            args.n_row_pattern,
            args.n_col_pattern,
            args.add_mean_time,
            args.add_error_time,
            args.n_samples_2_4,
            args.n_samples_8_16,
            args.ith_test,
            args.time_cost,
            args.accuracy_threshold,
            args.accuracy_coclustering,
        ]
        # col_names=['test_type','n_row_data','n_col_data','n_row_pattern','n_col_pattern',
        #           'mean','error','n_samples_2_4','n_samples_8_16','ith_test','time_cost',
        #           'f1']

    elif args.test_type == "mean_test":
        res = [
            args.test_type,
            args.n_row_data,
            args.n_col_data,
            args.n_row_pattern,
            args.n_col_pattern,
            args.add_error_time,
            args.add_mean_time,
            args.n_samples_2_4,
            args.n_samples_8_16,
            args.ith_test,
            args.time_cost,
            args.accuracy_threshold,
            args.accuracy_coclustering,
        ]
        # col_names = []
    elif args.test_type == "size_test" or args.test_type == "small_size_test":
        res = [
            args.test_type,
            args.n_row_data,
            args.n_col_data,
            args.add_error_time,
            args.add_mean_time,
            args.n_row_pattern,
            args.n_col_pattern,
            args.n_samples_2_4,
            args.n_samples_8_16,
            args.ith_test,
            args.time_cost,
            args.accuracy_threshold,
            args.accuracy_coclustering,
        ]
        # col_names = []

    elif args.test_type == "halfAngleTheta_test":
        res = [
            args.test_type,
            args.half_angle,
            hard_info["cos"],
            args.n_row_data,
            args.n_col_data,
            args.n_row_pattern,
            args.n_col_pattern,
            args.n_samples_2_4,
            args.n_samples_2_4,
            args.n_samples_8_16,
            int(args.n_samples_8_16 / 10),
            args.add_error_time,
            args.add_mean_time,
            args.ith_test,
            hard_info["time_cost"],
            hard_info["max_memory"],
            hard_info["f1_threshold"],
            hard_info["accuracy_threshold"],
            hard_info["f1_coclustering"],
            hard_info["accuracy_coclustering"],
        ]
    elif args.test_type == "halfAngleTheta_test_bigSize":
        res = [
            args.test_type,
            args.half_angle,
            hard_info["cos"],
            args.n_row_data,
            args.n_col_data,
            args.n_row_pattern,
            args.n_col_pattern,
            args.n_samples_2_4,
            int(args.n_samples_2_4 / 100),
            int(args.n_samples_8_16 / 10),
            int(args.n_samples_8_16 / 100),
            args.add_error_time,
            args.add_mean_time,
            args.ith_test,
            hard_info["time_cost"],
            hard_info["max_memory"],
            hard_info["f1_threshold"],
            hard_info["accuracy_threshold"],
            hard_info["f1_coclustering"],
            hard_info["accuracy_coclustering"],
        ]
        res = pd.DataFrame(res).T
        file_name = (
            "halfAngleTheta_bigSize"
            + str(args.half_angle)
            + "_"
            + args.test_type
            + ".csv"
        )
        save_path = args.project_dir + args.res_dir + file_name
        res.to_csv(save_path, mode="a", index=False, header=False)
        print("Save Done!!")
        return True
    elif args.test_type == "halfAngleTheta_test_bigSize_200":
        res = [
            args.test_type,
            args.half_angle,
            hard_info["cos"],
            args.n_row_data,
            args.n_col_data,
            args.n_row_pattern,
            args.n_col_pattern,
            args.n_samples_2_4,
            int(args.n_samples_2_4 / 100),
            int(args.n_samples_8_16 / 10),
            int(args.n_samples_8_16 / 100),
            args.add_error_time,
            args.add_mean_time,
            args.ith_test,
            hard_info["time_cost"],
            hard_info["max_memory"],
            hard_info["f1_threshold"],
            hard_info["accuracy_threshold"],
            hard_info["f1_coclustering"],
            hard_info["accuracy_coclustering"],
        ]
        res = pd.DataFrame(res).T
        file_name = (
            "halfAngleTheta_bigSize_pattern200_"
            + str(args.half_angle)
            + "_"
            + args.test_type
            + ".csv"
        )
        save_path = args.project_dir + args.res_dir + file_name
        res.to_csv(save_path, mode="a", index=False, header=False)
        print("Save Done!!")
        return True

    res = pd.DataFrame(res).T

    # file_name = ['mtrxSize', str(args.n_row_data), '_', str(args.n_col_data), '_llr1Size', str(args.n_row_pattern), '_',
    #             str(args.n_col_pattern),
    #             '_mean', str(args.add_mean_time), 'sd_error', str(args.add_error_time), 'sd_noShffule.csv.gz']
    file_name = "halfAngleTheta" + str(args.half_angle) + "_" + args.test_type + ".csv"
    save_path = args.project_dir + args.res_dir + file_name

    res.to_csv(save_path, mode="a", index=False, header=False)

    print("Save Done!!")

    return True
