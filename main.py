# -*- coding: utf8 -*


import argparse, sys, time, os
from multiprocessing import Pool, cpu_count
from functools import reduce
import pandas as pd
import numpy as np

# my libs
# sys.path.append(os.getcwd())
# sys.path.append(os.path.abspath("../"))
# sys.path.append(os.path.abspath("/N/slate/pdang/myProjectsTest/20200113PLLRM"))


from data_interface import generate_simulation_data

# from data_interface import load_data
from data_interface import shuffle_data

# from data_interface import normalize_matrix
from data_interface import normalize_dataframe

# from data_interface import random_sampling
from data_interface import random_sampling_slow
from data_interface import generate_probes
from data_interface import inner_square_sum
from data_interface import orthogonal_max_plooing
from data_interface import get_samples_weights_by_inner_sigmaRate
from data_interface import get_number_probes

# from data_interface import get_samples_weights_by_svd_sigmaRate
from data_interface import get_score_matrix
from data_interface import weighted_sampling_on_samples

# from data_interface import weighted_sampling_on_samples_slow
from model_interface import spectral_coclustering

sep_sign = "*" * 100


def rpsp():
    # score_matrix=np.ones((10,10))
    # score_matrix=pd.DataFrame(score_matrix)
    # return score_matrix

    rdm_rowIdx_colIdx, score_matrix = None, ""

    try:
        for qi in q_dims:
            print(sep_sign)
            print("\n The curent sample size is:{0}X{0}\n".format(qi))
            print(sep_sign)

            samples_3d = None

            if qi == 2:
                # random_sampling
                rdm_rowIdx_colIdx, samples_3d = random_sampling_slow(
                    qi, qDim_nSamples[qi], data_np
                )
            else:
                # qi=4,8,16
                # weighted sampling
                rdm_rowIdx_colIdx, samples_3d = weighted_sampling_on_samples(
                    qi, qDim_nSamples[qi], weights, rdm_rowIdx_colIdx, data_np
                )

            # get probes
            probes = generate_probes(qi, qDim_nProbe[qi])

            # inner product
            singular_values_3d = inner_square_sum(samples_3d, probes)

            # orthogonal_max_plooing
            singular_values_3d_max = orthogonal_max_plooing(qi, singular_values_3d)

            # get weights of each sample
            weights = get_samples_weights_by_inner_sigmaRate(singular_values_3d_max)

            score_matrix = get_score_matrix(
                args, qi, score_matrix, weights, rdm_rowIdx_colIdx
            )

    except:
        print("Error!")
        return 0

    return score_matrix


def main(args):
    args.n_row_pattern = args.n_pattern
    args.n_col_pattern = args.n_pattern

    print(sep_sign)
    print("\n Current Parameters:\n{0}\n".format(args))
    print(sep_sign)

    data = generate_simulation_data(args)  # generate synthetic data
    data = shuffle_data(data)  # shuffle rows and then columns
    data = normalize_dataframe(data)
    row_ids = data.index.map(lambda x: int(x.split("Id")[1]))
    col_ids = data.columns.map(lambda x: int(x.split("Id")[1]))

    # return False

    global data_np
    data_np = data.to_numpy().copy()

    time_cost = 0.0
    time_s = time.time()

    # get q_dim and their number of probes
    global qDim_nProbe
    qDim_nProbe = get_number_probes()

    # sample sizes
    global q_dims
    q_dims = None
    n_row, n_col = data_np.shape
    if n_row <= 5:
        q_dims = [2]
    elif n_row <= 9:
        q_dims = [2, 4]
    elif n_row <= 17:
        q_dims = [2, 4, 8]
    else:
        q_dims = [2, 4, 8, 16]

    global qDim_nSamples
    qDim_nSamples = {
        2: args.n_samples_2_4,
        4: args.n_samples_2_4,
        8: args.n_samples_8_16,
        16: int(args.n_samples_8_16),
    }

    time_s = time.time()

    # parallel version
    # score_matrix_res = []
    # """
    # parallel version
    n_cpu = cpu_count()
    print("\n cpu_count is {0}\n".format(n_cpu))
    print(
        f"Ihe algorithm will run on {n_cpu} CPU/GPU cores in parallel. So you will see many logs in the console."
    )
    # sleep for 3 seconds
    time.sleep(3)

    with Pool(n_cpu) as p:
        score_matrix_res = p.starmap(rpsp, [() for _ in range(n_cpu)])
    # print(score_matrix_res)
    score_matrix = reduce(lambda a, b: a + b, score_matrix_res)
    # """
    # score_matrix = rpsp()

    print("\n The sum of all score_matries:\n {0} \n".format(score_matrix))

    time_e = time.time()
    time_cost = time_e - time_s
    time_cost = np.round(time_cost, decimals=2)
    args.time_cost = time_cost
    print("\n total time is {0}\n".format(time_cost))

    try:
        label_rowColId = spectral_coclustering(score_matrix + 0.001, row_ids, col_ids)
        print(f"Local Low Rank Pattern Recognition:\n{label_rowColId}")
    except:
        # set threshold to 0.5
        score_matrix = score_matrix - np.mean(score_matrix)
        score_matrix = 1 / (1 + np.exp(-score_matrix))
        label_rowColId = np.where(score_matrix + 0.001 > 0.5)
        print(f"Local Low Rank Pattern Recognition:\n{label_rowColId}")

    return True


def parse_arguments(parser):
    # parameters
    parser = argparse.ArgumentParser()

    # data parameters
    parser.add_argument("--seed_col", type=int, default=-1)
    parser.add_argument("--seed_row", type=int, default=-1)

    # data parameters
    parser.add_argument(
        "--project_dir",
        type=str,
        default="./inputs/",
    )
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--res_dir", type=str, default="./outputs/")
    parser.add_argument("--data_source", type=str, default="")
    parser.add_argument("--n_row_data", type=int, default=10)
    parser.add_argument("--n_col_data", type=int, default=10)
    parser.add_argument("--n_row_pattern", type=int, default=5)
    parser.add_argument("--n_col_pattern", type=int, default=5)
    parser.add_argument("--n_pattern", type=int, default=5)
    parser.add_argument("--lk", type=int, default=1)
    parser.add_argument("--add_mean_time", type=float, default=0.0)
    parser.add_argument("--add_error_time", type=float, default=0.0)
    parser.add_argument("--n_samples_2_4", type=int, default=1000)
    parser.add_argument("--n_samples_8_16", type=int, default=10000)
    parser.add_argument("--time_cost", type=float, default=0.0)
    parser.add_argument("--accuracy", type=float, default=0.0)
    parser.add_argument("--test_type", type=str, default="NaN")
    parser.add_argument("--ith_test", type=str, default=1)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RPSP")

    # global args
    args = parse_arguments(parser)

    main(args)
