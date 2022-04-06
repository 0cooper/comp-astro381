#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 3 10:01:15 2022

@author: 0cooper

"""
import numpy as np


def random_number(N, seed, lo=0, hi=1):
    """
    Random number generator
    
    Parameters
    ----------
    N : int 
        array length of random numbers to generate
    seed : float
        seed for random number generator
    lo : float, default = 0 
        min for range of random numbers to generate from
    hi : float, default = 1 
        max for range of random numbers to generate from
  
    Returns
    ----------
    randarr : array
        array of N random numbers between lo and hi
        
    seed : float
        output seed for random number generator iterations
    """
    a = 1664525
    c = 1013904223
    m = 4294967296
    randarr = np.empty(N)
    for i in range(N):
        seed = (a*seed+c)%m
        rand = seed/m*(hi-lo)+lo
        randarr[i] = rand
    
    return randarr, seed