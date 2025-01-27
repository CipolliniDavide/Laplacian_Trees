#!/bin/bash

# Define the Python script to run
python_script="Portegys.py"


# Number of times the experiment is replicated
max_iter=50

fig_format="pdf"
# Define the list of output directories
output_directories=(
#"boom-fashionmnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=500,500-s=1"
#"boom-fashionmnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=1000,1000-s=1"
"boom-fashionmnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=3566897019"
"boom-fashionmnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=4000,4000-s=1"
"boom-mnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=64186134"
"boom-mnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=4000,4000-s=1"
"boom-nist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=1"
"boom-cifar10-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=1"
#"boom-cifar10-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=4000,4000-s=1"
#"boom-mnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=20,20-s=1"
)


# Loop over each output directory
for dir in "${output_directories[@]}"; do
    echo "Processing directory: $dir"
    # Call the Python script with the current output directory as an argument
    python "$python_script" -ds "./Dataset/$dir"  --max_iter ${max_iter} -fig_form ${fig_format}
done
