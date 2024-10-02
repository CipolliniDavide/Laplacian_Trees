import re
import matplotlib as plt

from os import getcwd
from pathlib import Path
from sys import argv

file_in = Path(getcwd() + "/" + argv[1])
regex = r"N(\d+) -> N(\d+)"

# 1 = parent, 0 = child
edges = []

with open(file_in, "r") as file:
    for line in file.readlines():
        groups = re.search(regex, line)
        edges.append([int(groups[1]), int(groups[2])])

# We now have a list of edges in the form of [[child, parent], [child, parent] ... etc]

# Here we keep track of the nodes, [parent] = [list of children]
nodes = {}

# We preallocate a list, the amount of edges is index-mapped
hist = [0 for _ in range(len(edges))]

# Insertion logic
def insert(nodes, parent, child):
    # If a parent is unknown we insert it without children
    if not parent in nodes:
        nodes[parent] = []
    # Add child to its parent
    nodes[parent].append(child)
    # We return the degree of the parent BEFORE insertion
    return len(nodes[parent]) - 1

# Increment the index-mapped list entry with the degree of the parent before insertion
for edge in edges:
    hist[insert(nodes, edge[1], edge[0])] += 1

# Print the histogram
print(hist)

sum = 0
cumulative = []
for entry in hist:
    sum += entry
    cumulative.append(sum)
print(cumulative)