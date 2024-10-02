
import sys
import pandas as pd
import argparse
from argparse import RawTextHelpFormatter
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
from helpers_dataset.helpers_dataset import (order_files_by_r, read_csv_convert_to_dataframe, create_dataset_Portegys,
                                             load_key_by_)


if  __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('-ds', '--dataset_dir',
                        default=f'{os.getcwd()}/Workspace/boom-mnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=64186134/output/',
                        # default=f'{os.getcwd()}/Dataset/boom-mnist-k=5-nc=0,1,2,3,4,5,6,7,8,9-f=2000,2000-s=64186134',
                        # default=f'{os.getcwd()}/Dataset/boom-mnist-temp',
                        type=str, help='Name of dataset directory.')
    # parser.add_argument('-s_dir', '--save_dir', default=f'{os.getcwd()}/Output/PortegysTree', type=str, help='Name of dataset_dir.')
    parser.add_argument('-crt_prop', '--create_properties', action='store_true',
                        help='Create properties file. This must be true if it is the first time this script is called.')
    parser.add_argument('-fig_form', '--fig_format', default='png', type=str, help='')
    args = parser.parse_args()
    print(args)

    args.dataset_dir = args.dataset_dir + '/'
    save_fold = args.dataset_dir.replace('Dataset', 'Output').replace('output/', '')

    # Load networks:
    if 'output' not in args.dataset_dir:
        raise Exception("Sorry, your dataset directory doesn't contain 'output' directory.")

    load_path = f'{args.dataset_dir}/'
    print('Load_path', load_path)
    file_list = sorted(glob('{:s}/*.tree'.format(load_path)))

    if not file_list:
        raise Exception("Sorry, the folder is empty:\nCheck you dataset directory and try again.")

    dict_files = order_files_by_r(file_list=file_list)
    # Create and load dataset_dir
    save_prop_dir = load_path.replace('output', 'prop')

    create_dataset_Portegys(save_fold_ds=save_fold,
                            dict_files=dict_files,
                            save_fold_prop=save_prop_dir,
                            crt_properties=args.create_properties)

    # df_nw = pd.read_hdf(f'{save_fold}/dataset_nw.h5', key='data')
    # df_nw = df_nw[df_nw['R'] == .85]
    # spectrum = df_nw['spectrum'].to_numpy()
    # for s in spectrum:
    #     print(s.min(), s.max())
    #     plt.plot(s)
    #     plt.show()