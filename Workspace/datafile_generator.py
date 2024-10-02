import numpy as np

import mnist
import cifar10
from argparse import ArgumentParser
from mnist import Sample
from os import path
from random import Random
import json


def split_samples_per_class(samples: list[Sample]) -> dict[int, Sample]:
    split = {}

    for sample in samples:
        if sample.label not in split:
              split[sample.label] = []
        split[sample.label].append(sample)
    
    return split

def take_shuffled_fraction(samples: list[Sample], fraction: float, random: Random) -> list[Sample]:
    random.shuffle(samples)
    return samples[:int(fraction * len(samples))]

def take_shuffled_subset(samples: list[Sample], number: int, random: Random) -> list[Sample]:
    print(f"Taking shuffled subset: {number}")
    random.shuffle(samples)
    return samples[:number]

def write_data_file(filename: str, samples: dict[int, Sample], binarize: bool=True) -> None:
    total_samples = 0
    num_features = 0
    for value in samples.values():
        total_samples += len(value)
        num_features = len(value[0].image)

    print(f"Outputting {total_samples} samples, {num_features} features each into {filename}")

    if binarize:
        print("Binarizing")
        with open(filename, "w") as file:
            # write header
            file.write(f"NAMED DATA {total_samples} {num_features}\n")

            for class_samples in samples.values():
                for sample in class_samples:
                    file.write(f"{sample.label}/{sample.index} ")
                    for pixel in sample.image:
                        file.write(f"{1.0 if pixel > 127.0 else 0.0:.1f} ")
                    file.write("\n")
    else:
        print("Not Binarizing")
        with open(filename, "w") as file:
            # write header
            file.write(f"NAMED DATA {total_samples} {num_features}\n")

            for class_samples in samples.values():
                for sample in class_samples:
                    file.write(f"{sample.label}/{sample.index} ")
                    for pixel in sample.image:
                        file.write(f"{pixel:.8f} ")
                    file.write("\n")


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("-s", "--seed", type=int, help="Sets the seed for reproducable outputs")
    args.add_argument("-c", "--classes", nargs="+", type=int, help="The classes to select include, includes all if not set")
    args.add_argument("-f", "--fraction", type=float, help="How large a fraction of samples to take from each class", default=1.0)
    args.add_argument("-n", "--number", type=int, nargs=2,
                      help="Specifies the ammount if images to take from train and test set respectively, has precedense over fraction", default=None)
    args.add_argument("-b", "--balance", type=bool, help="Whether to balance the classes (clamps to lowest amount)", default=False)
    args.add_argument("-d", "--dataset", type=str, help="Which dataset to parse, can be Mnist or FashionMnist (case insensitive)",
                      default="cifar10")
    args.add_argument("-i", "--input", type=str, help="Dataset folder", default="data")
    args.add_argument("-o", "--output", type=str, help="Output folder", default=".")
    args.add_argument("-bin", "--binarize_features", default=1, help="Binarize the features True. Otherwise pixel values [0, 1]", type=int)
    args.add_argument("-l", "--limit-features", type=int, help="Sets a limit to the # of features outputted", default=0)
    arguments = args.parse_args()

    rng = Random(arguments.seed) if arguments.seed else Random()
    classes = arguments.classes or [i for i in range(10)]
    fraction = arguments.fraction or 1.0
    balance = arguments.balance or False
    dataset = arguments.dataset.lower()
    input = arguments.input
    output = arguments.output
    limit = arguments.limit_features

    print(f"seed: {arguments.seed}, classes: {classes}, fraction: {fraction}, balance: {balance}")

    if path.exists(f"{output}/train.dat") and path.exists(f"{output}/test.dat"):
        print("Train & Test file already exist, exiting...")
        exit(0)

    # Load the dataset
    if dataset == "mnist":
        train_samples = mnist.load_dataset(f"{input}/mnist/train-images.idx3-ubyte", f"{input}/mnist/train-labels.idx1-ubyte")
        test_samples = mnist.load_dataset(f"{input}/mnist/t10k-images.idx3-ubyte", f"{input}/mnist/t10k-labels.idx1-ubyte")
    elif dataset == "fashionmnist":
        train_samples = mnist.load_dataset(f"{input}/fashionmnist/train-images-idx3-ubyte", f"{input}/fashionmnist/train-labels-idx1-ubyte")
        test_samples = mnist.load_dataset(f"{input}/fashionmnist/t10k-images-idx3-ubyte", f"{input}/fashionmnist/t10k-labels-idx1-ubyte")
    elif dataset == "cifar10":
        train_samples, test_samples = cifar10.load_dataset(path=f"{input}/cifar10/")

    # Split samples per class and filter out classes that aren't selected
    train_samples = split_samples_per_class(train_samples)
    train_samples = {k: train_samples[k] for k in classes}
    test_samples = split_samples_per_class(test_samples)
    test_samples = {k: test_samples[k] for k in classes}

    # Reduce to fraction if fraction is not 1.0
    if not arguments.number and fraction < 1.0 and fraction > 0.0:
        for cls in train_samples.keys():
            train_samples[cls] = take_shuffled_fraction(train_samples[cls], fraction, rng)
        for cls in test_samples.keys():
            test_samples[cls] = take_shuffled_fraction(test_samples[cls], fraction, rng)
    elif arguments.number:
        train_samples_per_set = arguments.number[0] // len(train_samples)
        test_samples_per_set = arguments.number[1] // len(test_samples)

        if train_samples_per_set < 1 or test_samples_per_set < 1:
            print("Cannot take less samples then sample classes")
            exit(-1)
        if train_samples_per_set * len(train_samples) != arguments.number[0]:
            print("Warning! Required of samples is not divisible by number of classes, the result will include slightly less samples.")

        for cls in train_samples.keys():
            train_samples[cls] = take_shuffled_subset(train_samples[cls], train_samples_per_set, rng)
        for cls in test_samples.keys():
            test_samples[cls] = take_shuffled_subset(test_samples[cls], test_samples_per_set, rng)
    else:
        print("No valid fraction or sample count defined, using full sample set")
    
    # Balance the dataset if required, only does it for train data currently (needed only if --fraction is used)
    if balance:
        cutoff = 1000000
        for samples in train_samples.values():
            cutoff = min(cutoff, len(samples))

        for key in train_samples.keys():
            train_samples[key] = train_samples[key][:cutoff]
        # for key in test_samples.keys():
        #     test_samples[key] = test_samples[key][:cutoff]

    # We limit the number of features for each sample if requested (both test & train)
    if limit != 0: # needs work
        for key in train_samples.keys():
            for sample in train_samples[key]:
                sample.image = sample.image[:limit]
        
        for key in test_samples.keys():
            for sample in test_samples[key]:
                sample.image = sample.image[:limit]

    # Now we should have the proper subset of images & features ready
    write_data_file(f"{output}/train.dat", train_samples, binarize=arguments.binarize_features)
    write_data_file(f"{output}/test.dat", test_samples, binarize=arguments.binarize_features)

    # Save arguments
    with open(f'{output}/arguments_datafile.json', 'w') as fp:
        json.dump(vars(arguments), fp)
