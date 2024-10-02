#!/bin/bash


root=$(pwd)
# Number of times the experiment is replicated
max_iter=5
subset=100

#for subset in 20 50 100
#do
#dataset="nist"
#seed=1
#
#dataset="mnist"
#seed=64186134

#dataset="fashionmnist"
#seed=3566897019
#seed=1
#binarize_features=1


dataset='cifar10'
seed=1
binarize_features=0

################################################ Create Dataset ########################################################
cd ./Workspace
python gridsearch.py --seed ${seed}\
 -r 0.05 .85 0.05 \
--iteration 0 ${max_iter} \
-t 360 \
--subset ${subset} ${subset} \
-ds ${dataset} \
--binarize_features ${binarize_features} \
-v
#python gridsearch_nist.py --seed ${seed} -r 0.05 .9 .05 --iteration 0 ${max_iter} -t 360 -v
cd ..

python create_network_dataset.py \
-ds ${root}/Workspace/boom-${dataset}-k\=5-nc\=0\,1\,2\,3\,4\,5\,6\,7\,8\,9-f\=${subset}\,${subset}-s\=${seed}/output/ \
-crt_prop
##
mkdir -p ${root}/Dataset/boom-${dataset}-k\=5-nc\=0\,1\,2\,3\,4\,5\,6\,7\,8\,9-f\=${subset}\,${subset}-s\=${seed}/output
mkdir -p ${root}/Dataset/boom-${dataset}-k\=5-nc\=0\,1\,2\,3\,4\,5\,6\,7\,8\,9-f\=${subset}\,${subset}-s\=${seed}/prop
#
mv ${root}/Workspace/boom-${dataset}-k\=5-nc\=0\,1\,2\,3\,4\,5\,6\,7\,8\,9-f\=${subset}\,${subset}-s\=${seed}/output/* \
${root}/Dataset/boom-${dataset}-k\=5-nc\=0\,1\,2\,3\,4\,5\,6\,7\,8\,9-f\=${subset}\,${subset}-s\=${seed}/output/
#
mv ${root}/Workspace/boom-${dataset}-k\=5-nc\=0\,1\,2\,3\,4\,5\,6\,7\,8\,9-f\=${subset}\,${subset}-s\=${seed}/prop/* \
${root}/Dataset/boom-${dataset}-k\=5-nc\=0\,1\,2\,3\,4\,5\,6\,7\,8\,9-f\=${subset}\,${subset}-s\=${seed}/prop/

############################################ Gather data ################################################################
pathdir="${root}/Dataset/boom-${dataset}-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=${subset},${subset}-s=${seed}/output/"
pathsavedir="${root}/Dataset/boom-${dataset}-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=${subset},${subset}-s=${seed}/"

# Remove old dataset
rm ${pathsavedir}dataset.csv

# Iterate through log files
for ((iter=0; iter<max_iter; iter++)); do
  for f in "$pathdir"/*iter=${iter}*.log; do
    # Extract lines containing 'Accu', replace spaces with commas, and modify Ndist key
    fgrep Accu "$f" | tr ' ' ',' | sed 's/,%,/,/' | sed "s/Ndist=/Ndist=$iter,/" >> "$pathsavedir/dataset.csv"
  done
done
#
############################################# Make plots ##################################################################
####
python Portegys.py -ds ${root}/Dataset/boom-${dataset}-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=${subset},${subset}-s=${seed} \
 --max_iter ${max_iter} -fig_form "png"
##
#python Portegys_plot_over_all_r.py -ds ${root}/Dataset/boom-${dataset}-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=${subset},${subset}-s=${seed} \
# --max_iter ${max_iter} -fig_form "png"