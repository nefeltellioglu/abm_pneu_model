#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 12:38:36 2023

@author: ntellioglu
"""

class ContactMatrix(object):
    def __init__(self, givenmatrix, age_classes):
        self.C = givenmatrix.T
        self.EC = None
        self.max_age = age_classes[-1]
        self.yearly_ages = age_classes[1] - age_classes[0]
        self.age_classes =  age_classes
        self.age_class_ids = {}
        