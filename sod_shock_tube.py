#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 10:52:15 2022

@author: 0cooper
analytical solution adapted from @author: ibackus
from their sod-shocktube code hosted at: https://github.com/ibackus/sod-shocktube

"""

import numpy as np
import scipy
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib
import astropy
from astropy.table import Table
import sod_shock_tube
plt.style.use('cooper-paper.mplstyle')


def cs(gamma, pressure, density):
    """
    Calculate the sound speed for given gamma, pressure, density

    """

    return np.sqrt(gamma * pressure / density)
    


def shock_fx(P4, P1, P5, rho1, rho5, gamma):
    """
    Calculate shock tube function, analytical solution

    """
    cs1 = cs(gamma, P1, rho1)
    cs5 = cs(gamma, P5, rho5)

    term = (gamma - 1.) / (2. * gamma) * (cs5 / cs1) * (P4 / P5 - 1.) / np.sqrt(1. + (gamma + 1.) / (2. * gamma) * (P4 / P5 - 1.))
    coeff = (1. - term) ** ((2. * gamma )/ (gamma - 1.))

    return P1 * coeff - P4


def calc_regions(PL, uL, rhoL, PR, uR, rhoR, gamma=1.4):
    """
    Compute regions for the shock tube analytical solution

    """
    # true for PL > PR
    rho1 = rhoL
    P1 = PL
    u1 = uL
    rho5 = rhoR
    P5 = PR
    u5 = uR

    # solve for post-shock pressure
    P4 = scipy.optimize.fsolve(shock_fx, P1, (P1, P5, rho1, rho5, gamma))[0]

    # compute post-shock  velocity
    cs5 = cs(gamma, P5, rho5)

    coeff = np.sqrt(1. + 0.5 * (gamma + 1.) / gamma * (P4 / P5 - 1.))

    u4 = cs5 * (P4 / P5 - 1.) / (gamma * coeff)
    rho4 = rho5 * (1. + (0.5 * (gamma + 1.) / gamma) * (P4 / P5 - 1.)) / (1. + (0.5 * (gamma - 1.)/ gamma) * (P4 / P5 - 1.))

    # shock speed
    w = cs5 * coeff

    # compute values at foot of rarefaction
    P3 = P4
    u3 = u4
    rho3 = rho1 * (P3 / P1) ** (1. / gamma)
    return (P1, rho1, u1), (P3, rho3, u3), (P4, rho4, u4), (P5, rho5, u5), w


def calc_positions(PL, PR, region1, region3, w, xi, t, gamma):
    """
    Find shock tube positions in the following order: Head of Rarefaction, Foot of Rarefaction,
            Contact Discontinuity, Shock
    """
    P1, rho1 = region1[:2]  # don't need velocity
    P3, rho3, u3 = region3
    cs1 = cs(gamma, P1, rho1)
    cs3 = cs(gamma, P3, rho3)

    xsh = xi + w * t
    xcd = xi + u3 * t
    xft = xi + (u3 - cs3) * t
    xhd = xi - cs1 * t

    return xhd, xft, xcd, xsh


def region_states(PL, PR, region1, region3, region4, region5):
    """
    Dictionary (region no.: p, rho, u), except for rarefaction region
 
    """
    return {'Region 1': region1,
            'Region 2': 'RAREFACTION',
            'Region 3': region3,
            'Region 4': region4,
            'Region 5': region5}



def create_arr(PL, PR, xL, xR, positions, state1, state3, state4, state5,
                  npts, gamma, t, xi):
    """
    Create arrays of x, p, rho and u values across the domain of interest
    
    """
    xhd, xft, xcd, xsh = positions
    P1, rho1, u1 = state1
    P3, rho3, u3 = state3
    P4, rho4, u4 = state4
    P5, rho5, u5 = state5

    x_arr = np.linspace(xL, xR, npts)
    rho = np.zeros(npts, dtype=float)
    P = np.zeros(npts, dtype=float)
    u = np.zeros(npts, dtype=float)
    cs1 = cs(gamma, P1, rho1)
    for i, x in enumerate(x_arr):
        if x < xhd:
            rho[i] = rho1
            P[i] = P1
            u[i] = u1
        elif x < xft:
            u[i] = 2. / (gamma + 1.) * (cs1 + (x - xi) / t)
            coeff = 1. - 0.5 * (gamma - 1.) * u[i] / cs1
            rho[i] = rho1 * coeff ** (2. / (gamma - 1.))
            P[i] = P1 * coeff ** (2. * gamma / (gamma - 1.))
        elif x < xcd:
            rho[i] = rho3
            P[i] = P3
            u[i] = u3
        elif x < xsh:
            rho[i] = rho4
            P[i] = P4
            u[i] = u4
        else:
            rho[i] = rho5
            P[i] = P5
            u[i] = u5
    

    return x_arr, P, rho, u


def solve(PL, rhoL, uL, PR, rhoR, uR, xL, xR, xi, t, gamma=1.4, N=200):
    """
    Solves the Sod shock tube problem analytically (i.e. riemann problem) of discontinuity
    across an interface.
    
    Parameters
    ----------
    PL, rhoL, uL, PR, rhoR, uR, xL, xR, xi: floats
        State (pressure P, density rho, velocity u, position x) on left/right (R/L) side of the
        shocktube barrier for the initial conditions, and xi position of shock interface
    t: float
        Time to calculate the solution at
    gamma: float
        Adiabatic index for the gas
    N: int
        Number of points for arrays of pressure, density, velocity, energy
        
    Returns
    -------
    result: dict
        Arrays of position, pressure, density, velocity, and internal energy
        
    """


    # calculate regions
    region1, region3, region4, region5, w = \
        calc_regions(PL, uL, rhoL, PR, uR, rhoR, gamma)

    regions = region_states(PL, PR, region1, region3, region4, region5)

    # calculate positions
    x_pos = calc_positions(PL, PR, region1, region3, w, xi, t, gamma)

    pos_description = ('Head of Rarefaction', 'Foot of Rarefaction',
                       'Contact Discontinuity', 'Shock')
    positions = dict(zip(pos_description, x_pos))

    # create arrays
    x, P, rho, u = create_arr(PL, PR, xL, xR, x_pos,
                                 region1, region3, region4, region5,
                                 N, gamma, t, xi)

    e = P / (rho * (gamma - 1.0))
    result = {'x': x, 'P': P, 'rho': rho, 'u': u, 'energy': e}

    return result

def numerical(gamma=1.4,rhoL=1.,rhoR=0.125,prL=1.,prR=0.1,velL=0.,velR=0.,\
                         xL=0.,xi=0.75,xR=2.,CFL=0.5,N=200,q0=4,q1=0.5,tmax=0.245,verbose=True):
    
    """
    Solves the Sod shock tube problem numerically 
    
    Parameters
    ----------
    gamma: float
        Adiabatic index for the gas
    rhoL, rhoR, prL, prR, velL, velR, xL, xi, xR : floats
        State (pressure pr, density rho, velocity vel, position x) on left/right (R/L) side of the
        shocktube barrier for the initial conditions, and xi position of shock interface
    CFL : float
        Courant number (stability criterion)
    N: int
        Number of points for arrays of pressure, density, velocity, etc 
    q0, q1: floats
        Artificial viscosity parameters
    tmax: float
        Time to integrate numerical solution out to and calculate the solution at 
    verbose: boolean
        True will output 1) plot of the initial conditions 2) prints counter and time for numerical routine
        
    Returns
    -------
    Plots position vs pressure, density, velocity, and internal energy on separate subplots with the analytical
        solution as a black line and each numerical solution as colorful stars
        
    """

    # initialize old arrays
    x_old = np.linspace(start=xL, stop=xR, num=N) # position, includes init cond for x
    rho_old = np.zeros(N-1) # density
    vel_old = np.zeros(N) # velocity
    pr_old = np.zeros(N-1) # pressure
    qvisc_old = np.zeros(N-1) # artificial viscosity
    dt = 0.0001 # initial time step

    # apply initial conditions to the other arrays
    left = np.where(x_old[:-1] <= xi) # left side from shock interface 
    right = np.where(x_old[:-1] > xi) # right side from shock interface 

    rho_old[left] = rhoL; rho_old[right] = rhoR # density init cond
    pr_old[left] = prL; pr_old[right] = prR # pressure init cond
    en_old = pr_old/(rho_old*(gamma-1)) # energy init cond
    cs_old = np.sqrt(gamma*pr_old/rho_old) # sound speed init cond
    dx_old = x_old[1:] - x_old[:-1] # grid spacing dx based on x array

    # set cell mass
    dm_i2 = 2*rho_old/N # dm i1/2
    dm_i = 0.5*(dm_i2[:-1]+dm_i2[1:]) # dm i
    dm_i = np.pad(dm_i, (1,1), 'symmetric') # pad dm array for array length

    # plot the initial conditions we set as a sanity check
    if verbose==True:
        f, ax = plt.subplots(3, sharex=True, figsize=(8,8))

        ax[0].plot(x_old[1:], pr_old, 'm')
        ax[0].set_ylabel('pressure')
        ax[0].set_ylim(0, 1.1)
        ax[0].set_xlim(xL, xR)

        ax[1].plot(x_old[1:], rho_old, 'c')
        ax[1].set_ylabel('density')
        ax[1].set_ylim(0, 1.1)
        ax[1].set_xlim(xL, xR)

        ax[2].plot(x_old, vel_old, 'b')
        ax[2].set_ylabel('velocity')
        ax[2].set_xlim(xL, xR)

        plt.show()
    else:
        pass

    # initialize time and counter
    time = 0.0 # time
    ct = 1 # counter
    
    # begin while loop, stop at tmax
    while time < tmax:

        # boundary conditions, set edge velocities = 0
        vel_old[0], vel_old[-1] = 0., 0.

        # counter print out
        if verbose==True:
            print("counter = ",ct)
        else:
            pass

        # solve for new velocity
        prpad = np.pad(pr_old, (1,1), 'symmetric') # pad edges to match array size to vel array
        qpad = np.pad(qvisc_old, (1,1), 'symmetric') # pad edges to match array size to vel array
        pr_diff = prpad[1:] - prpad[:-1] # delta pressure
        q_diff = qpad[1:] - qpad[:-1] # delta artificial viscosity
        vel_new = vel_old - dt*(pr_diff + q_diff)/dm_i # velocity

        # compute new grid positions
        x_new = x_old + dt*vel_new

        # compute new grid widths
        dx_new = x_new[1:] - x_new[:-1]

        # compute new density
        rho_new = dm_i2/dx_new

        # compute sound speed
        cs_new = np.sqrt(gamma*pr_old/rho_old)

        # compute artificial viscosity
        rho_avg = 0.5*(1/rho_old + 1/rho_new) # average density
        vel_diff = vel_new[1:] - vel_new[:-1] # delta velocity
        qvisc_new = np.zeros((N-1)) # set q = 0 initially
        idx = np.where(vel_diff/dx_new < 0)[0] # condition to replace qvisc with non zero value
        qvisc_new[idx] = (q0*vel_diff[idx]**2 - q1*vel_diff[idx])*(cs_new[idx]/rho_avg[idx]) # q array

        # compute new energy
        en_new = en_old - (pr_old+qvisc_old)*(1/rho_new - 1/rho_old) 

        # compute new pressure
        pr_new = en_new*rho_new*(gamma-1)

        # update all arrays
        vel_old, x_old, dx_old, rho_old, cs_old, qvisc_old, en_old, pr_old = vel_new, x_new, dx_new, rho_new, cs_new, qvisc_new, en_new, pr_new

        # print out time at each round
        if verbose==True:
            print("at time = ",t)    
        else:
            pass
        
        # update time
        time += dt

        # compute next time step
        dt = np.min((CFL * dx_old) / (cs_old + vel_old[:-1]))

        # update counter for verbose output
        ct += 1

    # exact analytical result
    result = solve(prL, rhoL, velL, prR, rhoR, velR, xL, xR, xi, t=tmax, gamma=gamma, N=N)
    # start figure
    f, ax = plt.subplots(4, sharex=True, figsize=(8,8))

    # pressure
    ax[0].plot(result['x'], result['P'], 'k')
    ax[0].plot(x_new[1:], pr_new, 'm*')
    ax[0].set_ylabel('pressure')
    ax[0].set_ylim(0, 1.1)

    # density
    ax[1].plot(result['x'], result['rho'], 'k')
    ax[1].plot(x_new[1:], rho_new, 'c*')
    ax[1].set_ylabel('density')
    ax[1].set_ylim(0, 1.1)

    # velocity
    ax[2].plot(result['x'], result['u'], 'k')
    ax[2].plot(x_new, vel_new, 'b*')
    ax[2].set_ylabel('velocity')
    ax[2].set_ylim(-3, 3)

    # energy
    ax[3].plot(result['x'], result['energy'], 'k')
    ax[3].plot(x_new[1:], en_new, '*', color='coral')
    ax[3].set_ylabel('energy')
    ax[3].set_ylim(-1, 5)

    # limit to domain
    plt.xlim(xL,xR)
    plt.show()