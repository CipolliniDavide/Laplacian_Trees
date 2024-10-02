import subprocess
import random

from argparse import ArgumentParser
from os import chdir, makedirs

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("-r", "--radius", type=float, help="R parameter for portegys insert", default=0.7)
    args.add_argument("-f", "--folder", type=str, help="Folder containing data & output", default="output")
    args.add_argument("-i", "--iteration", type=int, help="Iteration number for file out", default=0)
    arguments = args.parse_args()

    ## List of args
    radius = str(arguments.radius)
    seed = str(random.randint(0, 2_147_483_647)) # 0 -> integer max
    train_data_file = "train.dat" # "nist-dig-train-2000.dat"
    logging = "0" # 0 is fastest
    test_data_file = "test.dat"
    method = "2" # 1 = abs, 2 = rel
    k = "5"
    kmin = "3"
    kmax = "100"
    nmin = "50"
    folder = arguments.folder
    iteration = arguments.iteration

    ## Extended options (assignment Jos addition)
    # if True will run through debugging tool
    debugging = 'none' # can also be 'gdb' or 'valgrind'

    source_file = "boom.c"
    program_name = "boom.out"
    compile_command = ["gcc", f"../{source_file}", "-lm", "-ggdb", "-o", f"../{program_name}"]

    run_command = [f"../{program_name}", radius, seed, train_data_file, logging, test_data_file, method, k, kmin, kmax, nmin, str(iteration)]

    if debugging == 'gdb':
        run_command = ["gdb", "--args"] + run_command
    elif debugging == 'valgrind':
        run_command = ["valgrind"] + run_command

    makedirs(folder, exist_ok=True)
    chdir(folder)

    compile_out = subprocess.run(compile_command, capture_output=True)
    if compile_out.returncode != 0:
        print("Compilation failed:")
        print(compile_out.stdout.decode('utf-8'))
        print(compile_out.stderr.decode('utf-8'))
        exit()

    run_out = subprocess.run(run_command, capture_output=debugging == 'none')
    if debugging == 'none':
        print(run_out.stdout.decode('utf-8'))
        print(run_out.stderr.decode('utf-8'))