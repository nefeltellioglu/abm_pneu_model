#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 15:30:25 2023

@author: ntellioglu
"""

import sys,os, time

from model.disease.at_risk_varying_disease_simulation import \
                                            AtRiskDisSimulation
from model.observers.obs_pop import PopulationObserver
from model.observers.obs_prevalence import PrevalenceObserver
#from model.observers.obs_vacc_rollout import VaccinationObserver
#from model.observers.obs_disease import DiseaseObserver
from model.observers.obs_vacc_rollout_scenarios import VaccinationObserver
#from model.observers.obs_disease_scenarios import DiseaseObserver
from model.observers.obs_disease_by_age import DiseaseObserverByAge
from model.observers.obs_vacc_delivered import VaccinationDeliveredObserver
from model.disease.at_risk_varying_disease import AtRiskDisease
from model.observers.obs_disease_by_age_at_risk import \
                                                DiseaseObserverByAgeAtRisk

from model.observers.obs_disease_by_vaccine import \
                                                DiseaseObserverByVaccineAtRisk

from model.observers.obs_vaccine_by_age_by_product_at_risk import \
                                                VaccineObserverByAgeByProduct
from model.observers.obs_disease_by_age_by_product_at_risk import \
                                                DiseaseObserverByAgeByProduct


from model.disease.contact_matrix import KnownContactMatrix
from model.disease.at_risk_varying_run import go_single


import pandas as pd
#import numpy as np
#np.set_printoptions(threshold=sys.maxsize)
import polars as pl
pl.Config.set_tbl_rows(50)
pl.Config.set_tbl_cols(100)
#from pathos.multiprocessing import ProcessingPool as Pool

from pathlib import Path
data_dir = Path('data')
if not data_dir.exists():
    print(os.getcwd())
    raise ValueError('data directory is missing')
    
class DiseaseModel(AtRiskDisease):
    """
    Local version of SIR disease, adding observers and vaccines specific 
    to this set of experiments.
    """

    def __init__(self, p, cmatrix, rng, fname, mode='w'):
        super(DiseaseModel, self).__init__(p, cmatrix, rng, fname, mode)
        #print(fname)
        self.add_observers(PopulationObserver(h5file = self.h5file, 
                                              interval = p['t_per_year']),
                           PrevalenceObserver( h5file= self.h5file, 
                                              strains = self.strains),
                           VaccinationObserver(h5file = self.h5file, 
                                              vaccines = self.vaccines, 
                                              interval = p['t_per_year'] ),
                           #DiseaseObserverByAge(h5file = self.h5file, 
                           #                 vaccines = self.vaccines, 
                           #                   interval = p['t_per_year']),
                           DiseaseObserverByAgeAtRisk(h5file = self.h5file, 
                                            vaccines = self.vaccines, 
                                              interval = p['t_per_year']),
                           DiseaseObserverByVaccineAtRisk(h5file = self.h5file, 
                                            vaccines = self.vaccines, 
                                              interval = p['t_per_year']),
                           VaccinationDeliveredObserver(h5file = self.h5file, 
                                              vaccines = self.vaccines),
                           VaccineObserverByAgeByProduct(h5file = self.h5file, 
                                              vaccines = self.vaccines,
                                              interval = p['t_per_year'] ),
                           DiseaseObserverByAgeByProduct(h5file = self.h5file, 
                                              vaccines = self.vaccines,
                                              interval = p['t_per_year'],
                                              data_directory = data_dir),
                            )
        


knowncmatrix = pd.read_csv(
    'data/population/all_contact_matrix_Australia_prem_2017.csv',
    header=None
).to_numpy()
# (basic usage) run simulation
# sweep parameters
# run single simulation
#go_single(cur_params, DiseaseModel, KnownContactMatrix(givenmatrix = knowncmatrix), 
#                      cur_params['seed'], verbose=True)

def new_go_single(parameters):
    go_single(parameters, DiseaseModel, 
              KnownContactMatrix(givenmatrix = knowncmatrix),
              parameters['seed'] )

