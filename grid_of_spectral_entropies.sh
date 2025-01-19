#!/bin/bash

root=$(pwd)
save_fold="${root}/Output/"

#ds=fashionmnist
############################################## Figure 2c ###############################################################
# Define the list of output directories
output_directories=(
#"boom-fashionmnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=500,500-s=1"
"boom-cifar10-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=1"
"boom-fashionmnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=3566897019"
"boom-fashionmnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=4000,4000-s=1"
"boom-mnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=64186134"
"boom-mnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=4000,4000-s=1"
"boom-nist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=1"
)

# Loop over each output directory
for dir in "${output_directories[@]}"; do
    echo "Processing directory: $dir"
    # Call the Python script with the current output directory as an argument
#    python "$python_script" -ds "./Dataset/$dir"  --max_iter ${max_iter} -fig_form "png"

  # Create figure S1a depicting a grid 2x4 of manually annotated graphs
  python grid_of_spectral_entropy.py \
  -lp_img "${root}/OutputTest/png/${dir}/spectrum/vne/" \
  --fig_name "${dir}_fig_vne" \
  -svp "$save_fold" \
  -nc 4 -nr 4 \
  --fig_format png
done
#python grid_of_spectral_entropy.py \
#-lp_img "$save_fold/Degree/" \
#--fig_name "fig_degree" \
#-svp "$save_fold" \
#-nc 5 -nr 4 \
#--fig_format png
