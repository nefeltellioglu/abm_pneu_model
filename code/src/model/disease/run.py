#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 15:17:24 2023

@author: ntellioglu
"""

import os

#from random import Random
import numpy as np
from .disease_simulation import DisSimulation
from ..population.utils import create_path
import tables as tb
from ..observers.obs_pop import PopulationObserver

def go_single(p, disease_type, cmatrix, cur_seed, sim_type=DisSimulation, 
                          verbose=False):
    """
    Run a single simulation (or load if previously run).
    
    It takes a parameter dictionary, disease class, contact matrix, a random
    seed, a simulation type and then runs a single simulation.
    
    :param p: The simulation parameters
    :type p: dict
    :param disease_type: The disease model to simulate.
    :type disease_type: :class:`DiseaseBase`
    :param cmatrix: The population contact matrix.
    :type cmatrix: :class:`ContactMatrix`
    :param cur_seed: Random seed for current experiment.
    :type cur_seed: int
    :param sim_type: Simulation class to use (current options are SimEpi or SimBDI).
    :type sim_type: :class:`SimEpi`
    :param verbose: Flag to indicate whether to write output to terminal.
    :type verbose: bool
    """

    create_path(p['prefix'])
    
    rng = np.random.RandomState(cur_seed)
    
    start_year = 0 #p['burn_in'] + p['epi_burn_in']
    year_list = [(start_year + x, start_year + y) for x, y in zip(p['years'][:-1], p['years'][1:])]

    disease_fname = os.path.join(p['prefix'], 'disease_%s_%s.hd5' % (year_list[-1]))

    # a check to remove invalid files; NB: will not remove files from partial/incomplete runs (use x for that)
    if os.path.isfile(disease_fname) and not tb.is_pytables_file(disease_fname):
        os.remove(disease_fname)
        
    # load disease if output file already exists, otherwise run simulation
    if os.path.isfile(disease_fname) and not p['overwrite']:
        print("@@_go_single: loading existing disease (%sseed=%d)..." % (
            disease_type,  cur_seed))
        print("NB: to overwrite existing disease output, rerun with 'x' switch (eg, 'python main.py s x')")
        disease = None
        try:
            disease = disease_type(p, cmatrix, rng, disease_fname, mode='r')
            
        except tb.exceptions.NoSuchNodeError:
            # this is thrown when requested observers don't exist in disease file...
            if disease:
                disease.done(False)
            print("existing disease not complete... rerunning")
        else:
            if disease.is_complete():
                disease.done()
                return
            else:
                disease.done(False)
                print("existing disease not complete... rerunning")
    if (verbose):
        print("@@_go_single: running simulation (seed=%d)..." % (cur_seed))
    # create and set up simulation object
    disease = disease_type(p, cmatrix, rng, disease_fname, mode='w')
    sim = sim_type(p, disease, rng)
    #sim.add_observers(PopulationObserver(sim.h5file))
    
    if 'fake' not in p:
        sim.run(verbose)
        if (verbose):
            print("\t... simulation DONE! (seed=%d)..." % (cur_seed))
        disease.done(True)
        #sim.done(complete=True)
        
        