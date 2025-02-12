import numpy as np
import re
import subprocess

from argparse import ArgumentParser
from os import makedirs
from random import Random
from tqdm import tqdm
import json


def generate_tree_file(filename):
    regex = r"\d+ -> \d+"
    with open(filename + ".hist", "r") as file:
        history = [line.strip() for line in file.readlines() if re.search(regex, line)]
    with open(filename + ".tree", "w") as file:
        for line in history:
            parts = line.split(' ')
            file.write(f" N{int(parts[0])} -> N{int(parts[2])}\n")

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("-ds", "--dataset", type=str, help="Dataset name", default="mnist")
    args.add_argument("-v", "--verbose", action="store_true", help="Suppresses all output if False")
    args.add_argument("-s", "--seed", type=int, help="Sets seed (random if not set)", default=None)
    args.add_argument("-r", "--radius", type=float, nargs=3, help="Radius start, stop, step", default=None)
    args.add_argument("-i", "--iteration", type=int, nargs=2, help="Iteration start, stop, step", default=None)
    args.add_argument("-sub", "--subset", type=int, nargs=2, help="Number of samples, i.e. number of nodes in the trees.", default=None)
    args.add_argument("-bin", "--binarize_features", default=1,
                      help="Binarize the features True. Otherwise pixel values [0, 1]", type=int)
    args.add_argument("-nc", "--num_classes", type=str, help="Comma-separated list of classes to include",
                      default="0,1,2,4,5,6,7,8,9")
    # args.add_argument("-b", "--balance", type=bool, default=True, help="Balance classes in the dataset (only needed it the dataset is built with key --fraction)."),
    args.add_argument("-t", "--timeout", type=int, help="Boom.out process timeout in seconds, stops invalid traversal states", default=120)
    arguments = args.parse_args()

    print(arguments)

    # Iteration
    iter_start = 0 if not arguments.iteration else arguments.iteration[0]
    iter_stop = 15 if not arguments.iteration else arguments.iteration[1]

    # Grid search searches over 3 parameters, R, num_classes and class_fraction
    r_start = 0.05 if not arguments.radius else arguments.radius[0]
    r_stop = 2.50 if not arguments.radius else arguments.radius[1]
    r_step = 0.05 if not arguments.radius else arguments.radius[2]

    # Subset
    s_train = 2000 if not arguments.subset else arguments.subset[0]
    s_test = 2000 if not arguments.subset else arguments.subset[1]

    # Parameters
    subset_r = np.arange(r_start, r_stop + r_step, r_step)
    # Set of classes to be included
    # Convert the string list to an actual list of integers
    num_classes = [list(map(int, arguments.num_classes.split(",")))]
    print(f"Using classes: {num_classes}")

    subset_numbers = [[s_train, s_test]]
    iterations = int(iter_stop - iter_start) #15

    dataset = arguments.dataset
    # dataset = "mnist" # or "fashionmnist"
    # dataset = "fashionmnist"

    timeout = arguments.timeout

    folder_name = "boom-{}-k={}-nc={}-f={}-s={}"
    progress_bar = tqdm(total=len(subset_r) * len(num_classes) * len(subset_numbers) * iterations)
    
    print(f"Running with R range ({r_start}, {r_stop}, {r_step}), {iterations} iterations")
    
    for nc in num_classes:
        for subset in subset_numbers:
            print(f"Subset: {subset}")
            # For each number of classes / fraction we make a single dataset
            seed = arguments.seed if arguments.seed else Random().randint(0, 2**32-1)
            folder = folder_name.format(dataset, 5, ",".join([str(c) for c in nc]), ",".join([str(n) for n in subset]), seed)
            makedirs(folder + "/output", exist_ok=True)

            with open(f'{folder}/arguments_gridsearch.json', 'w') as fp:
                json.dump(vars(arguments), fp)

            gen_data_cmd = ["python", "datafile_generator.py",
                            # "-b", arguments.balance,
                            "-s", str(seed),
                            "-d", dataset,
                            "-i", "data",
                            "-o", folder,
                            "--binarize_features", str(arguments.binarize_features),
                            "-c",
                            ] + [str(c) for c in nc] + ["-n"] + [str(n) for n in subset]
            subprocess.run(gen_data_cmd, capture_output=not arguments.verbose)

            # And run that specific dataset through multiple iterations / values for R
            for r in subset_r:
                # for it in range(iterations):
                for it in range(iter_start, iter_stop):
                    try:
                        run_boom_cmd = ["python", "runner.py",
                                            "-r", str(r),
                                            "-f", folder,
                                            "-i", str(it)
                                        ]
                        output = subprocess.run(run_boom_cmd, capture_output=True, timeout=timeout)
                        with open(f"{folder}/output/boom-r={r:.6f}-iter={it}.log", "w") as file:
                            file.write(output.stderr.decode("utf-8"))
                            file.write(output.stdout.decode("utf-8"))
                            
                        generate_tree_file(f"{folder}/output/boom-r={r:.6f}-iter={it}")

                        if arguments.verbose:
                            print(output.stderr.decode("utf-8"))
                            output.stdout.decode("utf-8")
                    except Exception as e:
                        print(f"Caught exception {e}, skipping iteration")
                    progress_bar.update(1)

    progress_bar.close()