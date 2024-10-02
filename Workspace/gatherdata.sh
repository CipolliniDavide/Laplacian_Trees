#!/bin/bash

# Directory containing the log files
pathdir="/home/davide/PycharmProjects/LaplacianTree/Dataset/boom-mnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=64186134/output/"
# Directory to save the CSV file
pathsavedir="/home/davide/PycharmProjects/LaplacianTree/Dataset/boom-mnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=64186134/"

pathdir="/home/davide/PycharmProjects/LaplacianTree/Dataset/boom-fashionmnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=3566897019/output/"
pathsavedir="/home/davide/PycharmProjects/LaplacianTree/Dataset/boom-fashionmnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=3566897019/"

#pathdir="/home/davide/PycharmProjects/LaplacianTree/Dataset/boom-nist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=1/output/"
#pathsavedir="/home/davide/PycharmProjects/LaplacianTree/Dataset/boom-nist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=1/"

#pathdir="/home/davide/PycharmProjects/LaplacianTree/Dataset/boom-mnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=4000,4000-s=1/output/"
#pathsavedir="/home/davide/PycharmProjects/LaplacianTree/Dataset/boom-mnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=4000,4000-s=1/"

# Remove old dataset
rm ${pathsavedir}dataset.csv

# Create a CSV file
#echo "Accu,Ndist" > "$pathsavedir/dataset.csv"

# Iterate through log files
for iter in {0..9}; do
  for f in "$pathdir"/*iter=${iter}*.log; do
    # Extract lines containing 'Accu', replace spaces with commas, and modify Ndist key
    fgrep Accu "$f" | tr ' ' ',' | sed 's/,%,/,/' | sed "s/Ndist=/Ndist=$iter,/" >> "$pathsavedir/dataset.csv"
  done
done