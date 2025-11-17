#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 17:46:28 2023

@author: ntellioglu
"""

import os
import tables as tb
from random import Random
from math import exp
from collections import defaultdict
import numpy as np
import polars as pl
import time
import matplotlib.pyplot as plt
import csv

from .disease_simulation import DisSimulation

from ..population.utils import will_live

from .disease_utils import gen_age_structured_pop
from .disease_simulation import DisSimulation
from ..population.disease_population import DisPopulation
from ..population.simulation import Simulation, _adjust_prob


class VaryingTransDisSimulation(DisSimulation):
    """
    Basic demographic simulation object.

    Handles updating of births, deaths, aging, immigration

    :param p: dictionary of simulation parameters.
    :type p: dict
    :param create_pop: If `True` (default), create a random population; 
        otherwise, this will need to be done later.
    :type create_pop: bool

    """
    def __init__(self, p, disease, rng):
        self.disease = disease
        self.rng = rng
        #self.nprng = np.random.RandomState(self.rng.randint(0, 99999999))
        super(VaryingTransDisSimulation, self).__init__(p, disease, rng)
            
        create_pop = True
        if create_pop:
            self.create_population()
            
    
     
         
    
    
    