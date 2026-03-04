#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 15:30:25 2023

@author: ntellioglu
"""

import sys,os, time

from pneu_abm.model.disease.disease_simulation import \
                                            DisSimulation
from pneu_abm.model.observers.obs_pop import PopulationObserver
from pneu_abm.model.observers.obs_prevalence import PrevalenceObserver
#from pneu_abm.model.observers.obs_vacc_rollout import VaccinationObserver
#from pneu_abm.model.observers.obs_disease import DiseaseObserver
from pneu_abm.model.observers.obs_vacc_rollout_scenarios import VaccinationObserver
#from pneu_abm.model.observers.obs_disease_scenarios import DiseaseObserver
from pneu_abm.model.observers.obs_disease_by_age import DiseaseObserverByAge
from pneu_abm.model.observers.obs_disease_by_vaccine import DiseaseObserverByVaccine
from pneu_abm.model.observers.obs_prevalence_by_age import PrevalenceByAgeObserver

from pneu_abm.model.observers.obs_vacc_delivered import VaccinationDeliveredObserver
from pneu_abm.model.disease.disease import Disease



from pneu_abm.model.disease.contact_matrix import ContactMatrix
from pneu_abm.model.disease.run import go_single


from pathlib import Path


import pandas as pd
#import numpy as np
#np.set_printoptions(threshold=sys.maxsize)
import polars as pl
pl.Config.set_tbl_rows(50)
pl.Config.set_tbl_cols(100)
#from pathos.multiprocessing import ProcessingPool as Pool

    
class DiseaseModel(Disease):
    """
    Local version of SIR disease, adding observers and vaccines specific 
    to this set of experiments.
    """

    def __init__(self, p, cmatrix, fname, mode='w'):
        super(DiseaseModel, self).__init__(p, cmatrix, fname, mode)
        #print(fname)
        self.add_observers(PopulationObserver(h5file = self.h5file, 
                                              interval = p['t_per_year']),
                           PrevalenceObserver( h5file= self.h5file, 
                                              strains = self.strains),
                           VaccinationObserver(h5file = self.h5file, 
                                              vaccines = self.vaccines, 
                                              interval = p['t_per_year'] ),
                           DiseaseObserverByAge(h5file = self.h5file, 
                                            vaccines = self.vaccines, 
                                              interval = p['t_per_year'],
                                              strains_df = self.strains_df),
                           DiseaseObserverByVaccine(h5file = self.h5file, 
                                            vaccines = self.vaccines, 
                                              interval = p['t_per_year'],
                                              strains_df = self.strains_df),
                           VaccinationDeliveredObserver(h5file = self.h5file, 
                                              vaccines = self.vaccines),
                           PrevalenceByAgeObserver(h5file = self.h5file, 
                                            vaccines = self.vaccines, 
                                              interval = p['t_per_year']),
                           )
        

from pathlib import Path

# (basic usage) run simulation
# sweep parameters
# run single simulation
#go_single(cur_params, DiseaseModel, KnownContactMatrix(givenmatrix = knowncmatrix), 
#                      cur_params['seed'], verbose=True)

def make_contact_matrix(parameters):
    """Return a ContactMatrix instance built from the parameter set.

    This mirrors the logic used in ``new_go_single`` and is exposed so that
    other modules (e.g. interactive plotting utilities) can obtain the same
    contact matrix without duplicating file‑reading code.

    Args:
        parameters (dict): scenario parameter dictionary containing at least
            ``cmatrix`` (relative path) and ``age_classes``.
    """
    data_dir = Path('src/pneu_abm/data')
    if not data_dir.exists():
        raise ValueError('data directory is missing')

    cm_array = pd.read_csv(f'{data_dir}/{parameters["cmatrix"]}',
                           header=None).to_numpy()
    return ContactMatrix(givenmatrix=cm_array,
                         age_classes=parameters['age_classes'])


def new_go_single(parameters):
    cmatrix = make_contact_matrix(parameters)
    go_single(parameters, DiseaseModel, cmatrix)

