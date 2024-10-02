
# Tree networks of real-world data: analysis of efficiency and spatiotemporal scales,

This repository contains the code used in the paper _**Cipollini and Schomaker, 2024, Tree networks of real-world data: analysis of efficiency and spatiotemporal scales, https://doi.org/10.48550/arXiv.2404.17829.**_

To replicate the results, run:
```
bash run_pipeline.sh
```

The bash file calls the following scripts: 
- `Workspace/gridsearch.py` or depending on your choice `Workspace/gridsearch_nist.py`
- `create_network_dataset.py`
- `Portegys.py`

Before running the bash file: 
- from the Zenodo download, take the folder `data` and place it in `./Workspace`
- from the Zenodo download, take the files `data\nist\train.dat` and `data\nist\test.dat` and place them in `Workspace\boom-nist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=1`
- uncomment the corresponding lines 
to a desired dataset (MNIST, NIST, FASHIONMNIST) and the seed to select a specific subset of the dataset.
- Set the number of independent tree formation processes and the parameter _r_ range at line 22 (or 23).
- Select the maximum iterations, e.g. max_iter=100, over which you want to compute the 
averaged quantities with the call to `Portegys.py`

To generate the tree structures for the NIST dataset, uncomment line 23 in `run_pipeline.sh`. The seed is irrelevant for the NIST case.  

The script:
- `dataset_complexity_estimation.py` plots figure 2.
- `Portegys_plot_over_all_r.py` produces the same figure and analysis as `Portegys.py`, but addresses the full range of `r` for panel (a) in figure 5.
- `hierarchical_phase_transition.py` plots panels (d) and (e) in figure 5.
- `grid_of_spectral_entropy.sh` calls `grid_of_spectral_entropy.py`: takes as input the path of a directory and plots the .jpg files contained in a tabular structure.
- `plot_network_over_r.py` produces panel (b) in figure 5.
- `run_overDatasets.sh` only runs `Portegys.py` the analysis and produces the plots.
- `clean.sh` and `move.sh` are helper scripts to erase and move files in `Workspace` and `Dataset` (the folder that will contain the generated trees after you run `run_pipeline.sh`)

### Dataset
https://zenodo.org/records/13883206

### Acknowledgments
The script `boom.c` is provided by Prof. Dr. L. Schomaker.
The authors gratefully acknowledge Jos van Goor for providing the scripts contained in the directory `./Workspace` that enabled efficient grid-search across parameters.
Methods for measuring thermodynamic efficiency - `thermo_spectrum` and `thermo_trajectory` - are imported from the script reported in
Ghavasieh, A., De Domenico, M. Diversity of information pathways drives sparsity in real-world networks. Nat. Phys. 20, 512–519 (2024).

