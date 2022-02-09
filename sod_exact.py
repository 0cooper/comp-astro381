#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 10:52:15 2022

@author: 0cooper
adapted from @author: ibackus
from their sod-shocktube code hosted at: https://github.com/ibackus/sod-shocktube
"""

import numpy as np
import scipy
import scipy.optimize


def cs(gamma, pressure, density):
    """
    Calculate the sound speed

    Parameters
    ----------
    gamma : TYPE
        DESCRIPTION.
    pressure : TYPE
        DESCRIPTION.
    density : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    return np.sqrt(gamma * pressure / density)
    


def shock_fx(P4, P1, P5, rho1, rho5, gamma):
    """
    Calculate shock tube function

    Parameters
    ----------
    P4 : TYPE
        DESCRIPTION.
    P1 : TYPE
        DESCRIPTION.
    P5 : TYPE
        DESCRIPTION.
    rho1 : TYPE
        DESCRIPTION.
    rho5 : TYPE
        DESCRIPTION.
    gamma : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    cs1 = cs(gamma, P1, rho1)
    cs5 = cs(gamma, P5, rho5)

    term = (gamma - 1.) / (2. * gamma) * (cs5 / cs1) * (P4 / P5 - 1.) / np.sqrt(1. + (gamma + 1.) / (2. * gamma) * (P4 / P5 - 1.))
    coeff = (1. - term) ** ((2. * gamma )/ (gamma - 1.))

    return P1 * coeff - P4


def calc_regions(PL, uL, rhoL, PR, uR, rhoR, gamma=1.4):
    """
    Compute regions
    :rtype : tuple
    :return: returns p, rho and u for regions 1,3,4,5 as well as the shock speed
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
    :return: tuple of positions in the following order ->
            Head of Rarefaction: xhd,  Foot of Rarefaction: xft,
            Contact Discontinuity: xcd, Shock: xsh
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
    :return: dictionary (region no.: p, rho, u), except for rarefaction region
    where the value is a string, obviously
    """
    return {'Region 1': region1,
            'Region 2': 'RAREFACTION',
            'Region 3': region3,
            'Region 4': region4,
            'Region 5': region5}



def create_arr(PL, PR, xL, xR, positions, state1, state3, state4, state5,
                  npts, gamma, t, xi):
    """
    :return: tuple of x, p, rho and u values across the domain of interest
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
    Solves the Sod shock tube problem (i.e. riemann problem) of discontinuity
    across an interface.
    Parameters
    ----------
    left_state, right_state: tuple
        A tuple of the state (pressure, density, velocity) on each side of the
        shocktube barrier for the ICs.  In the case of a dusty-gas, the density
        should be the gas density.
    geometry: tuple
        A tuple of positions for (left boundary, right boundary, barrier)
    t: float
        Time to calculate the solution at
    gamma: float
        Adiabatic index for the gas.
    npts: int
        number of points for array of pressure, density and velocity
    dustFrac: float
        Uniform fraction for the gas, between 0 and 1.
    Returns
    -------
    positions: dict
        Locations of the important places (rarefaction wave, shock, etc...)
    regions: dict
        constant pressure, density and velocity states in distinct regions
    values: dict
        Arrays of pressure, density, and velocity as a function of position.
        The density ('rho') is the gas density, which may differ from the
        total density in a dusty-gas.
        Also calculates the specific internal energy
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

    return positions, regions, result