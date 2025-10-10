#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 12:38:36 2023

@author: ntellioglu
"""

from math import pi, exp, sqrt

import numpy as np
import polars as pl

age_classes = list(range(0, 80, 5))

#givencmatrix
import pandas as pd
#aus_contact_mat = pd.read_csv('/Users/ntellioglu/Documents/pneumonia/code_v5/data/all_contact_matrix_Australia_prem_2017.csv',header=None).to_numpy()


class KnownContactMatrix(object):
    def __init__(self, givenmatrix,#givenmatrix = aus_contact_mat,
                 age_classes = age_classes):
        self.C = givenmatrix
        self.EC = None
        self.max_age = 100
        self.age_classes = age_classes
        self.age_class_ids = {}
        self._create_age_maps(age_classes)

    
    def _create_age_maps(self, cutoffs):
        """
        Create maps from :
            (a) age (years) to age_classes (index); and
            (b) age (years) to age_class_ids (left edge of age class)
        """

        # an age map that maps ages to age class indices
        self.age_map = []
        prev_cutoff = 0
        for index, cur_cutoff in enumerate(cutoffs[1:] + [self.max_age + 2]):
            self.age_map.extend([index] * (cur_cutoff - prev_cutoff))
            prev_cutoff = cur_cutoff

        # an age map that maps ages to left edge of matching age class
        self.age_ids_map = []
        prev_cutoff = 0
        for index, cur_cutoff in enumerate(cutoffs[1:] + [self.max_age + 2]):
            self.age_ids_map.extend([prev_cutoff] * (cur_cutoff - prev_cutoff))
            prev_cutoff = cur_cutoff
            
    def init_a_levels(self, P):
        pass

    def init_age_classes(self, P):
        for age, next_age in zip(age_classes, age_classes[1:] + [self.max_age + 1]):
            self.age_class_ids[age] = P.I.filter(
                                        pl.col("age")
                                        .is_in(range(age,next_age)))
            #print("age %s"%age)
            
            
    def init_contact_matrix_gaussian(self, epsilon, sigma_2):
        #self.C = givenmatrix
        self._expand_contact_matrix()
    
    def init_contact_matrix(self):
        """
        Initialise the contact matrix.
        
        Epsilon in the range [0,1] is a convex combination parameter:
        epsilon = 1 ---> proportionate mixing
        epsilon = 0 ---> preferred mixing (i.e., on the diagonal only)
        """
        #self.C = givenmatrix
        #init_mat=self.C
        #self.C = np.zeros((self.max_age + 1, self.max_age + 1,))
        #for i in range(self.max_age + 1):
        #    ci = self.age_map[i]
        #    for j in range(len(self.age_classes)):
        #        cj = self.age_map[j]
        #        self.C[i][j] = init_mat[ci][cj]/5
        #self._expand_contact_matrix()
        self.EC=self.C

    def _expand_contact_matrix(self):
        """
        Create an expanded version of the contact matrix, with a row for each individual
        age.  This speeds up calculation of FOI slightly, with only a very small cost in
        storage redundancy.
        """
        
        self.EC = np.zeros((self.max_age + 1, len(self.age_classes)))
        for i in range(self.max_age + 1):
            ci = self.age_map[i]
            for j in range(len(self.age_classes)):
                #cj = self.age_map[j]
                self.EC[i][j] = self.C[ci][j]

    def update_age_classes(self, births, deaths, immigrants, birthdays):
        pass