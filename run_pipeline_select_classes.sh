#!/bin/bash

############################### Variables unspecific of the datasets ###################################################
root=$(pwd)
# Number of times the experiment is replicated
max_iter=25
# Number of nodes/samples
subset=2000
# Desired classes
nc=(0 1 2 3 5 6 7 8 9)
nc_str=$(IFS=,; echo "${nc[*]}")  # Converts the array in a string separated by commas
# Maximum value of the control parameter
r_max=.85


############################### Variables specific for the datasets ####################################################
#dataset="nist"
#seed=1

dataset="mnist"
#seed=64186134
seed=1

#dataset="fashionmnist"
#seed=3566897019
#seed=1
binarize_features=1

#dataset='cifar10'
#seed=1
#binarize_features=0

original_nc=(0 1 2 3 4 5 6 7 8 9)

for ((i = 0; i < ${#original_nc[@]}; i++)); do
    nc=("${original_nc[@]:0:$i}" "${original_nc[@]:$((i + 1))}")  # Remove one element
    nc_str=$(IFS=,; echo "${nc[*]}")  # Convert array to comma-separated string

    echo "Running experiment with classes: ${nc_str}"

################################################ Create Dataset ########################################################
#cd ./Workspace
#python gridsearch.py --seed ${seed}\
# -r 0.05 ${r_max} 0.1 \
#--iteration 10 ${max_iter} \
#-t 360 \
#--subset ${subset} ${subset} \
#-ds ${dataset} \
#--binarize_features ${binarize_features} \
#--num_classes "${nc_str}" \
#-v
#
### Uncomment the following line only in case of NIST
###python gridsearch_nist.py --seed ${seed} -r 0.05 ${r_max} .05 --iteration 0 ${max_iter} -t 360 -v
#
#cd ..

network_path="${root}/Workspace/boom-${dataset}-k=5-nc=${nc_str}-f=${subset},${subset}-s=${seed}/output/"
dataset_path="${root}/Dataset/boom-${dataset}-k=5-nc=${nc_str}-f=${subset},${subset}-s=${seed}/"

#python create_network_dataset.py -ds "${network_path}" -crt_prop
#
#mkdir -p "${dataset_path}/output"
#mkdir -p "${dataset_path}/prop"
#
#mv "${network_path}"* "${dataset_path}/output/"
#mv "${root}/Workspace/boom-${dataset}-k=5-nc=${nc_str}-f=${subset},${subset}-s=${seed}/prop"/* "${dataset_path}/prop/"

########################################### Gather data ################################################################
pathdir="${dataset_path}/output/"
pathsavedir="${dataset_path}/"

# Remove old dataset
rm -f "${pathsavedir}dataset.csv"

# Iterate through log files
for ((iter=0; iter<max_iter; iter++)); do
  for f in "$pathdir"/*iter=${iter}*.log; do
    fgrep Accu "$f" | tr ' ' ',' | sed 's/,%,/,/' | sed "s/Ndist=/Ndist=$iter,/" >> "${pathsavedir}dataset.csv"
  done
done

############################################# Make plots ##################################################################
python Portegys.py -ds "${dataset_path}" --max_iter ${max_iter} -fig_form "pdf" --r_max ${r_max}
#python Portegys_plot_over_all_r.py -ds ${root}/Dataset/boom-${dataset}-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=${subset},${subset}-s=${seed} \
# --max_iter ${max_iter} -fig_form "png"

done