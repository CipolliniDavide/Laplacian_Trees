import matplotlib.pyplot as plt
import re

from glob import glob
from sklearn import metrics
from sys import argv

def parse_det_file(filename: str) -> tuple[list[int], list[int]]:
    labels = []
    predictions = []

    with open(filename, "r") as file:
        header_skipped = False
        for line in file.readlines():
            if not header_skipped:
                header_skipped = True
                continue

            parts = line.split(",")
            labels.append(int(parts[1]))
            predictions.append(int(parts[2]))
    
    return labels, predictions

def group_runs(files: list[str]) -> dict[str, list[int]]:
    regex = r"boom-r=(\d+.\d+)-iter=(\d+)\.det"

    groups = {}

    for file in files:
        filename = file[file.find('/') + 1:]
        matches = re.search(regex, filename)
        
        base = matches[1]
        iteration = matches[2]
        if not base in groups:
            groups[base] = []
        groups[base].append(iteration)

    return groups

def group_filenames(r: str, iters: list[str], folder: str) -> list[str]:
    filenames = []

    for iter in iters:
        filenames.append(f"{folder}boom-r={r}-iter={iter}.det")
    
    return filenames

def combine_matches(group_filenames: list[str]) -> tuple[list[int], list[int]]:
    labels = []
    predictions = []

    for filename in group_filenames:
        l, p = parse_det_file(filename)
        labels += l
        predictions += p
    
    return labels, predictions

if __name__ == "__main__":
    folder = argv[1] + "/output/"

    files = glob(folder + "**.det")
    groups = group_runs(files)

    for r, iters in groups.items():
        print(f"--- GROUP {r} ---")
        files = group_filenames(r, iters, folder)
        labels, predictions = combine_matches(files)
        if len(labels) == 0:
            continue

        conf_mat = metrics.confusion_matrix(labels, predictions)
        display = metrics.ConfusionMatrixDisplay(conf_mat)
        # display.figure_.savefig(f"{argv[1]}/conf_mat-r={r}-iter=all.png")
        display.plot()
        plt.savefig(f"{argv[1]}/conf_mat-r={r}-iter=all.png")
        plt.close()