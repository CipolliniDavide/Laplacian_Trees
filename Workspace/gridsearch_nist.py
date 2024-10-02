import numpy as np
import re
import subprocess

from argparse import ArgumentParser
from os import makedirs
from random import Random
from tqdm import tqdm

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
    args.add_argument("-v", "--verbose", action="store_true", help="Suppresses all output if False")
    args.add_argument("-s", "--seed", type=int, help="Sets seed (random if not set)", default=None)
    args.add_argument("-r", "--radius", type=float, nargs=3, help="Radius start, stop, step", default=None)
    args.add_argument("-i", "--iteration", type=int, nargs=2, help="Iteration start, stop, step", default=None)
    args.add_argument("-t", "--timeout", type=int, help="Boom.out process timeout in seconds, stops invalid traversal states", default=120)
    arguments = args.parse_args()

    # Iteration
    iter_start = 0 if not arguments.iteration else arguments.iteration[0]
    iter_stop = 15 if not arguments.iteration else arguments.iteration[1]

    # Grid search searches over 3 parameters, R, num_classes and class_fraction
    r_start = 0.05 if not arguments.radius else arguments.radius[0]
    r_stop = 2.50 if not arguments.radius else arguments.radius[1]
    r_step = 0.05 if not arguments.radius else arguments.radius[2]

    # Parameters
    subset_r = np.arange(r_start, r_stop + r_step, r_step)
    num_classes = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]] # Set of classes to be included
    subset_numbers = [[2000, 2000]]
    iterations = int(iter_stop - iter_start) #15

    dataset = "nist" # or "fashionmnist"
    # dataset = "fashionmnist"

    timeout = arguments.timeout

    folder_name = "boom-{}-k={}-nc={}-f={}-s={}"
    progress_bar = tqdm(total=len(subset_r) * len(num_classes) * len(subset_numbers) * iterations)
    
    print(f"Running with R range ({r_start}, {r_stop}, {r_step}), {iterations} iterations")
    
    for nc in num_classes:
        for subset in subset_numbers:
            print(f"Subset: {subset}")
            # For each number of classes / fraction we make a single dataset
            # seed = arguments.seed if arguments.seed else Random().randint(0, 2**32-1)
            seed=1
            folder = folder_name.format(dataset, 5, ",".join([str(c) for c in nc]), ",".join([str(n) for n in subset]), seed)
            makedirs(folder + "/output", exist_ok=True)

            gen_data_cmd = ["python", "datafile_generator.py",
                                "-s", str(seed),
                                "-d", dataset,
                                "-i", "data",
                                "-o", folder,
                                "-c",
                            ] + [str(c) for c in nc] + ["-n"] + [str(n) for n in subset]
            # subprocess.run(gen_data_cmd, capture_output=not arguments.verbose)

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