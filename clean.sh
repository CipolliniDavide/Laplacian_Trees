#!/bin/bash

# Set the folder path
folder_path="./Dataset/boom-nist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=1/prop/"

# Iterate over each file in the folder
for file in "$folder_path"/*-iter=*.*
do
#    echo $file
    # Extract the value of "iter" from the file name
    iter=$(echo "$file" | grep -oP '(?<=-iter=)\d+')
#    echo $iter
    # Check if iter is greater than 20
    if [ "$iter" -gt 20 ]; then
        # Delete the file
        rm "$file"
        echo "Deleted: $file"
    fi
done
