"""
Observer object for carriage percentages for circulating strains.
"""


from .obs_base import Observer
import tables as tb
import numpy as np
import os
import matplotlib.pyplot as plt
import polars as pl

class VaccinationEffAcquisitionObserver(Observer):
    def __init__(self, h5file, vaccines, interval):
        #interval is in time ticks not days
        self.vaccines = vaccines
        self.interval = interval
        desc = {'t': tb.UInt32Col(pos=0),
                'age': tb.UInt32Col(pos=1),
                'quantile': tb.Float64Col(pos=2),
                'final_vaccine_time':tb.UInt32Col(pos=3),
                'vaccine_type': tb.StringCol(pos=4, itemsize = 30),
                'no_of_doses': tb.UInt32Col(pos=5),
                'meanlog': tb.Float64Col(pos=6),
                'sdlog': tb.Float64Col(pos=7),
                'exposed_strains': tb.StringCol(pos=8, itemsize = 4),
                'log_antibodies': tb.Float64Col(pos=9),
                'prob_of_transmission': tb.Float64Col(pos=10),
                'will_infected': tb.UInt32Col(pos=11),
                }              
        super(VaccinationEffAcquisitionObserver, self).__init__(h5file =h5file, \
                        label = 'vaccination_eff_acq', description = desc,
                            title = 'Vaccination Eff Acquisition Observer')
        

    
    def update(self, t, day, pop, **kwargs):
        
        if t % self.interval == 0:    
        
            if len(pop.vaccinated_acq_pop):
                #todo: will infected
                for row in pop.vaccinated_acq_pop.rows(named=True):
                    self.row['t'] = day
                    self.row['age'] = row['age']
                    self.row['quantile'] = row['quantile']
                    self.row['final_vaccine_time'] = row['final_vaccine_time']
                    self.row['vaccine_type'] = row['vaccine_type']
                    self.row['no_of_doses'] = row['no_of_doses']
                    self.row['meanlog'] = row['meanlog']
                    self.row['sdlog'] = row['sdlog']
                    self.row['exposed_strains'] = row['exposed_strains']
                    self.row['log_antibodies'] = \
                                            row["log_antibodies"]
                    self.row['prob_of_transmission'] = \
                                            row['prob_of_transmission']
                    self.row['will_infected'] = row['will_infected']  * 1                       
                    self.row.append()
                    self.h5file.flush()
                

   

    
            
    