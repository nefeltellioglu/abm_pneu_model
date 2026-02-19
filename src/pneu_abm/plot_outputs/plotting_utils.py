#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 12:29:30 2023

@author: ntellioglu
"""

############
#plotting

import sys,os
#from math import ceil, sqrt, log
import h5py    
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import polars as pl
import json
import re
#os.chdir(os.path.join(repo_path))

def add_headers(
    fig,
    *,
    row_headers=None,
    col_headers=None,
    row_pad=1,
    col_pad=5,
    rotate_row_headers=True,
    **text_kwargs
):
    # Based on https://stackoverflow.com/a/25814386

    axes = fig.get_axes()

    for ax in axes:
        sbs = ax.get_subplotspec()

        # Putting headers on cols
        if (col_headers is not None) and sbs.is_first_row():
            ax.annotate(
                col_headers[sbs.colspan.start],
                xy=(0.5, 1),
                xytext=(0, col_pad),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",
                **text_kwargs,
            )

        # Putting headers on rows
        if (row_headers is not None) and sbs.is_first_col():
            ax.annotate(
                row_headers[sbs.rowspan.start],
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - row_pad, 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
                rotation=rotate_row_headers * 90,
                **text_kwargs,
            )

def get_prev_data(directory,data_dir, filename, year, isIndigenous = False):
    
    full_file_address = os.path.join('%s/%s'%(directory,filename))
    
    
    strain_fname = os.path.join(data_dir,'disease/strain_list.csv')
    strains_df = pl.read_csv(
            strain_fname,
            comment_prefix="#",
            has_header=True)
    strain_list = np.sort(strains_df["serotype"].to_list())
    
    file_addresses = os.path.join(directory)
    filenames = os.listdir(file_addresses)#os.listdir(directory)
    #display(filenames)
    file_addresses = filenames.copy()
    ind_file_addresses = []
    """for item in filenames:
        if (not item.startswith("params")) or \
            (not item.endswith(".dat")):
            file_addresses.remove(item)"""
    #for folder in os.path.join(directory, file_addresses):
    for cur_address in file_addresses:
        for path, subdirs, files in os.walk(os.path.join(directory, 
                                                         cur_address)):
            for name in files:
                if name.endswith("hd5"):
                    ind_file_addresses.append(os.path.join(path, name))
    savingaddress = os.path.join(directory)#, "output")
    
    all_df = pl.DataFrame()
    for filename in ind_file_addresses:#[vaccination_file_addresses[0],vaccination_file_addresses[1]]:
        #filename = ind_file_addresses[0]
        try:
            eligible = None
            with h5py.File(filename, "r") as f:
                #extract t_per_year 
                os.path.dirname(filename)
                if "seed" in filename:
                    run_no = int(filename.split("/")[-2].split("_")[1])
            
                else:
                    run_no = 0
                    
                params = dict(f['params'].attrs.items())
                t_per_year = 365#params['t_per_year']
                
                #
                group_name = 'prevalence'
                key = 'base_data'
                raw_data = f[group_name][key][()].tolist()
                names = f[group_name][key][()].dtype.names
                df = pl.from_numpy(f[group_name][key][()],
                    schema=f[group_name][key][()].dtype.names, orient="col")
                df = df.with_columns(
                        pl.lit(run_no).alias("run_no"),
                        )
            f.close()
            all_df = pl.concat([all_df, df], how="diagonal", rechunk=True)
            print(filename)
        except:
            print("here")
            pass

    return (all_df, cap_ipd_rates, ipd_by_vacc_age, 
                unique_ipd_strains, 
                strain_list)
    
def get_disease_data(directory,data_dir, filename, year):
    
    strain_fname = os.path.join(data_dir,'disease/strain_list.csv')
    strains_df = pl.read_csv(
            strain_fname,
            comment_prefix="#",
            has_header=True)
    strain_list = np.sort(strains_df["serotype"].to_list())
    
    file_addresses = os.path.join(directory)
    filenames = os.listdir(file_addresses)#os.listdir(directory)
    #display(filenames)
    file_addresses = filenames.copy()
    ind_file_addresses = []
    
    
    #for folder in os.path.join(directory, file_addresses):
    for cur_address in file_addresses:
        for path, subdirs, files in os.walk(os.path.join(directory, 
                                                         cur_address)):
            for name in files:
                if name.endswith("hd5"):
                    ind_file_addresses.append(os.path.join(path, name))

    savingaddress = os.path.join(directory, "output")
    
    all_df = pl.DataFrame()
    #all_sum = pd.DataFrame()
    for filename in ind_file_addresses:#[vaccination_file_addresses[0],vaccination_file_addresses[1]]:
        try:
            eligible = None
            
            with h5py.File(filename, "r") as f:
                #extract t_per_year 
                os.path.dirname(filename)

                if "seed" in filename:
                    run_no = int(filename.split("/")[-2].split("_")[1])
                    
                else:
                    run_no = 0
                    
                params = dict(f['params'].attrs.items())
                t_per_year = 365
                
                #
                group_name = 'disease_byage'
                key = 'base_data'
                raw_data = f[group_name][key][()].tolist()
                names = f[group_name][key][()].dtype.names
                df = pl.from_numpy(f[group_name][key][()],
                    schema=f[group_name][key][()].dtype.names, orient="col")
                df = df.with_columns(
                        pl.lit(run_no).alias("run_no"),
                        )
            f.close()
            all_df = pl.concat([all_df, df], how="diagonal", rechunk=True)
            print("disease")
            print(filename)
        except:
            print("Here")
            pass

    return (all_df)
    
def get_vaccine_data(directory,data_dir, filename, year):
    
    strain_fname = os.path.join(data_dir,'disease/strain_list.csv')
    strains_df = pl.read_csv(
            strain_fname,
            comment_prefix="#",
            has_header=True)
    strain_list = np.sort(strains_df["serotype"].to_list())
    
    file_addresses = os.path.join(directory)
    filenames = os.listdir(file_addresses)#os.listdir(directory)
    #display(filenames)
    file_addresses = filenames.copy()
    ind_file_addresses = []
    
    #for folder in os.path.join(directory, file_addresses):
    for cur_address in file_addresses:
        for path, subdirs, files in os.walk(os.path.join(directory, 
                                                         cur_address)):
            for name in files:
                if name.endswith("hd5"):
                    ind_file_addresses.append(os.path.join(path, name))

    savingaddress = os.path.join(directory, "output")
    
    all_df = pl.DataFrame()
    #all_sum = pd.DataFrame()
    for filename in ind_file_addresses:#[vaccination_file_addresses[0],vaccination_file_addresses[1]]:
        try:
            eligible = None
            
            with h5py.File(filename, "r") as f:
                #extract t_per_year 
                os.path.dirname(filename)
                
                    
                if "seed" in filename:
                    run_no = int(filename.split("/")[-2].split("_")[1])

                else:
                    run_no = 0
                    
                params = dict(f['params'].attrs.items())
                t_per_year = 365
                
                #
                group_name = 'vaccination'
                key = 'base_data'
                a = f[group_name][key][()]
                a['vaccine_type'] = [str(v,'utf-8') for v in a['vaccine_type']]
                a['on_time'] = [str(o,'utf-8') == "True" for o in a['on_time']]
                a = a.astype([('t', '<u4'), 
                              ('vaccine_type', 'U20'), 
                              ('dose', '<u4'), ('age', '<u4'), 
                              ('on_time', 'U20'), ('fraction', '<f8'), 
                              ('birth_cohort', '<u4')])

                df = pl.from_numpy(a,orient="col")
                df = df.with_columns(
                        pl.lit(run_no).alias("run_no"),
                        )
            f.close()
            all_df = pl.concat([all_df, df], how="diagonal", rechunk=True)
            print(filename)
        except:
            pass

    return (all_df)
   
def get_vaccine_by_type_data(directory,data_dir, filename, year):
    
    strain_fname = os.path.join(data_dir,'disease/strain_list.csv')
    strains_df = pl.read_csv(
            strain_fname,
            comment_prefix="#",
            has_header=True)
    strain_list = np.sort(strains_df["serotype"].to_list())
    
    file_addresses = os.path.join(directory)
    filenames = os.listdir(file_addresses)#os.listdir(directory)
    #display(filenames)
    file_addresses = filenames.copy()
    ind_file_addresses = []
    
    for cur_address in file_addresses:
        for path, subdirs, files in os.walk(os.path.join(directory, 
                                                         cur_address)):
            for name in files:
                if name.endswith("hd5"):
                    ind_file_addresses.append(os.path.join(path, name))

    savingaddress = os.path.join(directory, "output")
    
    all_df = pl.DataFrame()
    #all_sum = pd.DataFrame()
    for filename in ind_file_addresses:#[vaccination_file_addresses[0],vaccination_file_addresses[1]]:
        try:
            eligible = None
            
            with h5py.File(filename, "r") as f:
                #extract t_per_year 
                os.path.dirname(filename)
                
                if "seed" in filename:
                    run_no = int(filename.split("/")[-2].split("_")[1])
                    
                else:
                    run_no = 0
                    
                params = dict(f['params'].attrs.items())
                t_per_year = 365
                
                #
                group_name = 'vacc_byagebyproduct'
                key = 'base_data'
                a = f[group_name][key][()]
                a['individual_type'] = [str(v,'utf-8') for v in a['individual_type']]
                a['vaccine_type'] = [str(v,'utf-8') for v in a['vaccine_type']]
                a['disease'] = [str(v,'utf-8') for v in a['disease']]
                a['vaccine_delivered'] = [str(v,'utf-8') for v in a['vaccine_delivered']]
                a = a.astype([('t', '<u4'), 
                              ('individual_type', 'U30'), 
                              ('vaccine_type', 'U30'), 
                              ('disease', 'U20'), 
                              ('vaccine_delivered', 'U20'), 
                              ('inds_per_age_group', '<u4', (106,))
                              ])
                df = pl.from_numpy(a,orient="col")
                
                df = df.with_columns(
                        pl.lit(run_no).alias("run_no"),
                        )
            f.close()
            all_df = pl.concat([all_df, df], how="diagonal", rechunk=True)
        except:
            print("here")
            pass

    return (all_df)
   
def get_disease_by_type_data(directory,data_dir, filename, year):
    
    strain_fname = os.path.join(data_dir,'disease/strain_list.csv')
    strains_df = pl.read_csv(
            strain_fname,
            comment_prefix="#",
            has_header=True)
    strain_list = np.sort(strains_df["serotype"].to_list())
    
    file_addresses = os.path.join(directory)
    filenames = os.listdir(file_addresses)#os.listdir(directory)
    #display(filenames)
    file_addresses = filenames.copy()
    ind_file_addresses = []
    
    #for folder in os.path.join(directory, file_addresses):
    for cur_address in file_addresses:
        for path, subdirs, files in os.walk(os.path.join(directory, 
                                                         cur_address)):
            for name in files:
                if name.endswith("hd5"):
                    ind_file_addresses.append(os.path.join(path, name))

    savingaddress = os.path.join(directory, "output")
    
    all_df = pl.DataFrame()
    #all_sum = pd.DataFrame()
    for filename in ind_file_addresses:#[vaccination_file_addresses[0],vaccination_file_addresses[1]]:
        print(filename)
        try:
            eligible = None
            
            with h5py.File(filename, "r") as f:
                #extract t_per_year 
                os.path.dirname(filename)
                
                if "seed" in filename:
                    run_no = int(filename.split("/")[-2].split("_")[1])
                    
                else:
                    run_no = 0
                    
                params = dict(f['params'].attrs.items())
                t_per_year = 365
                
                #
                group_name = 'disease_byagebyproduct'
                key = 'base_data'
                a = f[group_name][key][()]
                a['individual_type'] = [str(v,'utf-8') for v in a['individual_type']]
                a['vaccine_type'] = [str(v,'utf-8') for v in a['vaccine_type']]
                a['disease'] = [str(v,'utf-8') for v in a['disease']]
                a['when_vaccine_delivered'] = [str(v,'utf-8') for v in a['when_vaccine_delivered']]
                a['serotype_group'] = [str(v,'utf-8') for v in a['serotype_group']]
                a = a.astype([('t', '<u4'), 
                              ('individual_type', 'U30'), 
                              ('vaccine_type', 'U30'), 
                              ('disease', 'U20'), 
                              ('when_vaccine_delivered', 'U20'), 
                              ('inds_per_age_group', '<u4', (106,)),
                              ('serotype_group', 'U20'), 
                              ])
                df = pl.from_numpy(a,orient="col")
                
                df = df.with_columns(
                        pl.lit(run_no).alias("run_no"),
                        )
            f.close()
            all_df = pl.concat([all_df, df], how="diagonal", rechunk=True)
        except:
            print("here")
            pass

    return (all_df)
   


def get_vaccine_delivered(directory,data_dir, filename, year):
    
    strain_fname = os.path.join(data_dir,'disease/strain_list.csv')
    strains_df = pl.read_csv(
            strain_fname,
            comment_prefix="#",
            has_header=True)
    strain_list = np.sort(strains_df["serotype"].to_list())
    
    file_addresses = os.path.join(directory)
    filenames = os.listdir(file_addresses)#os.listdir(directory)
    #display(filenames)
    file_addresses = filenames.copy()
    ind_file_addresses = []
    
    #for folder in os.path.join(directory, file_addresses):
    for name in file_addresses:
        if name.endswith("hd5"):
            ind_file_addresses.append(os.path.join(directory, name))

    savingaddress = os.path.join(directory, "output")
    
    all_df = pl.DataFrame()
    #all_sum = pd.DataFrame()
    for filename in ind_file_addresses:#[vaccination_file_addresses[0],vaccination_file_addresses[1]]:
        try:
            eligible = None
            
            with h5py.File(filename, "r") as f:
                #extract t_per_year 
                os.path.dirname(filename)
                
                if "seed" in filename:
                    run_no = int(filename.split("/")[-2].split("_")[1])
                    
                else:
                    run_no = 0
                    
                params = dict(f['params'].attrs.items())
                t_per_year = 365
                
                #
                group_name = 'vaccination_delivered'
                key = 'base_data'
                a = f[group_name][key][()]
                #a['vaccine_type'] = [str(v,'utf-16') for v in a['vaccine_type']]
                a['vaccine_type'] = [str(v,'latin1') for v in a['vaccine_type']]
                
                a = a.astype([('t', '<u4'), 
                              ('vaccine_type', 'U40'), 
                              ('no_delivered', '<u4'), ('total_inds', '<u4')])
                df = pl.from_numpy(a,orient="col")
                df = df.with_columns(
                        pl.lit(run_no).alias("run_no"),
                        )
            f.close()
            all_df = pl.concat([all_df, df], how="diagonal", rechunk=True)
        except:
            pass

    return (all_df)
   
    

if __name__ == '__main__':
    
    repo_path = ".."
    repo_path = os.path.abspath(os.path.join(repo_path))
    
    
    
    
    