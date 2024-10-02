#!/bin/bash

#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=d.cipollini@rug.nl
#SBATCH --job-name='laplTree'
#SBATCH --output=laplTree-%j.log
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --array=1-50


source /home3/p304102/virtual_env/LaplacianTree/bin/activate
cd ./LaplacianTree/

root=$(pwd)
max_iter=$((SLURM_ARRAY_TASK_ID))
min_iter=$((SLURM_ARRAY_TASK_ID-1))

echo ${min_iter} ${max_iter}

subset=4000

#for subset in 20 50 100
#do
#dataset="nist"
#seed=1
#
#dataset="mnist"
#seed=64186134

#dataset="fashionmnist"
#seed=3566897019

#binarize_features=1



dataset='cifar10'
seed=1
binarize_features=0



################################################ Create Dataset ########################################################
cd ${root}/Workspace
python gridsearch.py --seed ${seed}\
 -r .05 .35 0.05 \
--iteration ${min_iter} ${max_iter} \
-t 360 \
--subset ${subset} ${subset} \
-ds ${dataset} \
--binarize_features ${binarize_features} \
-v
#python gridsearch_nist.py --seed ${seed} -r 0.05 .9 .05 --iteration 40 50 -t 360 -v
cd ..

dataset_folder=boom-${dataset}-k\=5-nc\=0\,1\,2\,3\,4\,5\,6\,7\,8\,9-f\=${subset}\,${subset}-s\=${seed}
echo ${dataset_folder}
# Remove files
rm -r ${root}/Workspace/${dataset_folder}/output/*-iter=${min_iter}.map
rm -r ${root}/Workspace/${dataset_folder}/output/*-iter=${min_iter}.hist
rm -r ${root}/Workspace/${dataset_folder}/output/*-iter=${min_iter}.det
rm -r ${root}/Workspace/${dataset_folder}/output/*-iter=${min_iter}.out


python create_single_file_network_dataset.py \
-r .05 0.85 0.05 \
-it ${min_iter} \
-ds ${root}/Workspace/${dataset_folder}/output/ \
-crt_prop
#
##python create_network_dataset.py \
##-ds ${root}/Workspace/${dataset_folder}/output/ \
##-crt_prop
##

mkdir -p ${root}/Dataset/boom-${dataset}-k\=5-nc\=0\,1\,2\,3\,4\,5\,6\,7\,8\,9-f\=${subset}\,${subset}-s\=${seed}/output
mkdir -p ${root}/Dataset/boom-${dataset}-k\=5-nc\=0\,1\,2\,3\,4\,5\,6\,7\,8\,9-f\=${subset}\,${subset}-s\=${seed}/prop

mv ${root}/Workspace/${dataset_folder}/output/*-iter=${min_iter}.tree \
${root}/Dataset/boom-${dataset}-k\=5-nc\=0\,1\,2\,3\,4\,5\,6\,7\,8\,9-f\=${subset}\,${subset}-s\=${seed}/output/

mv ${root}/Workspace/${dataset_folder}/output/*-iter=${min_iter}.log \
${root}/Dataset/boom-${dataset}-k\=5-nc\=0\,1\,2\,3\,4\,5\,6\,7\,8\,9-f\=${subset}\,${subset}-s\=${seed}/output/


mv ${root}/Workspace/${dataset_folder}/prop/* \
${root}/Dataset/boom-${dataset}-k\=5-nc\=0\,1\,2\,3\,4\,5\,6\,7\,8\,9-f\=${subset}\,${subset}-s\=${seed}/prop/
##

############################################ Gather data ################################################################
#pathdir="${root}/Dataset/boom-${dataset}-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=${subset},${subset}-s=${seed}/output/"
#pathsavedir="${root}/Dataset/boom-${dataset}-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=${subset},${subset}-s=${seed}/"
#
## Remove old dataset
#rm ${pathsavedir}dataset.csv
#
## Iterate through log files
#for ((iter=0; iter<max_iter; iter++)); do
#  for f in "$pathdir"/*iter=${iter}*.log; do
#    # Extract lines containing 'Accu', replace spaces with commas, and modify Ndist key
#    fgrep Accu "$f" | tr ' ' ',' | sed 's/,%,/,/' | sed "s/Ndist=/Ndist=$iter,/" >> "$pathsavedir/dataset.csv"
#  done
#done
#
############################################# Make plots ##################################################################
####
###python Portegys.py -ds ${root}/Dataset/boom-${dataset}-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=${subset},${subset}-s=${seed} \
### --max_iter ${max_iter} -fig_form "png"
##
#python Portegys_plot_over_all_r.py -ds ${root}/Dataset/boom-${dataset}-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=${subset},${subset}-s=${seed} \
# --max_iter ${max_iter} -fig_form "png"