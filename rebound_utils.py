#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 15:06:15 2022

@author: 0cooper
"""

# the basics
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import rebound

def rebound_orbit(dt,intg='ias15',bodies=['Sun','Jupiter','Saturn','Churyumov-Gerasimenko'],\
                       particle1='Jupiter',particle2='Churyumov-Gerasimenko',Noutputs=10000,plot=True):
    """
    Integrates orbits of solar system bodies using rebound, 
        loosely follows example in https://rebound.readthedocs.io/en/latest/ipython_examples/Churyumov-Gerasimenko/

    Parameters
    ----------
    dt : float
        Timestep for integration.
    intg : str, optional
        Integration scheme. The default is 'ias15', see https://rebound.readthedocs.io/en/latest/integrators/ for more options.
    bodies : str, optional
        List of particles to include in simulation. The default is ['Sun','Jupiter','Saturn','Churyumov-Gerasimenko'].
    particle1 : str, optional
        First particle to calculate distance from. The default is 'Jupiter'.
    particle2 : str, optional
        Second particle to calculate distance from. The default is 'Churyumov-Gerasimenko'.
    Noutputs : int
        Number of outputs for integration.
    plot : boolean, optional
        Plot orbits and distance. The default is True.

    Returns
    -------
    results : astropy table
        Table of orbital positions (x,y,z) and particle names generated from integration.

    """
    # set up the simulation with Sun, Jupiter, Saturn, C-G
    sim = rebound.Simulation()
    sim.add(bodies)
    nb = len(bodies)
    
    # set up integration params
    sim.dt = dt # set negative time step to evolve orbit backwards
    year = 2.*np.pi # One year in units where G=1
    times = np.linspace(0.,-70.*year, Noutputs) # time array
    x = np.zeros((nb,Noutputs))
    y = np.zeros((nb,Noutputs))
    z = np.zeros((nb,Noutputs))
    
    sim.integrator = intg    # IAS15 is the default integrator
    sim.move_to_com()        # We always move to the center of momentum frame before an integration
    ps = sim.particles       # ps is now an array of pointers and will change as the simulation runs

    for i,time in enumerate(times):
        sim.integrate(time)
        for j in range(nb):
            x[j][i] = ps[j].x
            y[j][i] = ps[j].y
            z[j][i] = ps[j].z
            
    p1 = np.where(np.array(bodies)==particle1)[0][0]
    p2 = np.where(np.array(bodies)==particle2)[0][0]
    distance = np.sqrt(np.square(x[p1]-x[p2])+np.square(y[p1]-y[p2])+np.square(z[p1]-z[p2]))   
    tyear = times/year
    
    if plot==True:    
        # plot orbits over last 70 years
        fig, ax = plt.subplots(2,figsize=(8,12))
        for j in range(nb):
            if j == 0:
                pass
            else:
                ax[0].plot(x[j], y[j], label=bodies[j])    
        ax[0].legend()

        # plot distance from particle1 to particle2
        ax[1].set_xlabel("time [yrs]")
        ax[1].set_ylabel("distance [AU]")
        ax[1].plot(tyear, distance)
    else:
        pass
    
    # dump orbits into table
    results = Table([bodies,x,y,z],names=('Name','x','y','z'))
    
    return results