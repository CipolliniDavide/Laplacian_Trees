#!/bin/bash

root=$(pwd)
save_fold="${root}/"

#ds=fashionmnist
############################################## Figure 2c ###############################################################
# Define the list of output directories
output_directories=("plots_for_appendix_thesis")

# Loop over each output directory
for dir in "${output_directories[@]}"; do
    echo "Processing directory: $dir"
    # Call the Python script with the current output directory as an argument
#    python "$python_script" -ds "./Dataset/$dir"  --max_iter ${max_iter} -fig_form "png"

  # Create figure S1a depicting a grid 2x4 of manually annotated graphs
  python grid_of_spectral_entropy.py \
  -lp_img "${root}/${dir}/" \
  --fig_name "${dir}_fig" \
  -svp "$save_fold" \
  -nc 10 -nr 3 \
  --fig_format png
done
#python grid_of_spectral_entropy.py \
#-lp_img "$save_fold/Degree/" \
#--fig_name "fig_degree" \
#-svp "$save_fold" \
#-nc 5 -nr 4 \
#--fig_format png
