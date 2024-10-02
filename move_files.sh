#!/bin/bash

# Set the folder path
folder_path="./Workspace/boom-mnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=4000,4000-s=1/output/"
folder_path_new="./Workspace/boom-mnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=4000,4000-s=1/output-iter=0-50/"

mkdir folder_path_new

# Iterate over each file in the folder
for file in "$folder_path"/*-iter=*.*
do
#    echo $file
    # Extract the value of "iter" from the file name
    iter=$(echo "$file" | grep -oP '(?<=-iter=)\d+')
#    echo $iter
    # Check if iter is greater than 20
    if [ "$iter" -gt 50 ]; then
        # move the file
        mv "$file" $folder_path_new
        echo "Deleted: $file"
    fi
done


