################################################################################
###### This is an example script to generate HDF5-format ICs for GIZMO
######  The specific example below is obviously arbitrary, but could be generalized
######  to whatever IC you need. 
################################################################################
################################################################################

## load libraries we will use 
import numpy as np
import h5py as h5py

# the main routine. this specific example builds an N-dimensional box of gas plus 
#   a collisionless particle species, with a specified mass ratio. the initial 
#   gas particles are distributed in a uniform lattice; the initial collisionless 
#   particles laid down randomly according to a uniform probability distribution 
#   with a specified random velocity dispersion
#
def make_IC():
    '''
    Built from the example subroutine provided to demonstrate how to make HDF5-format
    ICs for GIZMO. The specific example here is arbitrary, but can be generalized
    to whatever IC you need
    '''

    N = int(200) # 200 particles as in hw 2
    fname='sod_shock_hw2.hdf5'; # output filename 
    
    # ICs as in hw 2, right (R) and left (L)
    rhoL=1.
    rhoR=0.125
    prL=1.
    prR=0.1
    velL=0.
    velR=0.
    xL=0.
    xi=0.75
    xR=2.

    Lbox = 2.0 # box side length
    dust_to_gas_ratio = 0.01 # mass ratio of collisionless particles to gas
    gamma_eos = 5./3. # polytropic index of ideal equation of state the run will assume
    
    xarr = np.linspace(start=xL, stop=xR, num=N) # position, includes init cond for x
    yarr = 0.0*xarr; zarr = 0.0*xarr # position for 1D problem
    rhoarr = np.zeros(N) # density
    vxarr = np.zeros(N); vyarr = np.zeros(N); vzarr = np.zeros(N) # velocity
    Parr = np.zeros(N) # pressure

    # apply initial conditions to the other arrays
    left = np.where(xarr <= xi) # left side from shock interface 
    right = np.where(xarr > xi) # right side from shock interface 

    rhoarr[left] = rhoL; rhoarr[right] = rhoR # density init cond
    Parr[left] = prL; Parr[right] = prR # pressure init cond
    enarr = Parr/(rhoarr*(gamma_eos-1)) # energy init cond
    
    mL = rhoL * xi
    mR = rhoR * (Lbox - xi)
    mtot = mL + mR
    delta_m = mtot / N

    # for MFM/equal mass
    marr = delta_m * np.ones(N)
    
    # gas ids
    idarr = np.arange(1, N + 1)

    # now we get ready to actually write this out
    #  first - open the hdf5 ics file, with the desired filename
    file = h5py.File(fname,'w')

    npart = np.array([N, 0, 0, 0, 0, 0])

    h = file.create_group("Header");

    h.attrs['BoxSize'] = Lbox
    h.attrs['Flag_Cooling'] = 0
    h.attrs['Flag_Feedback'] = 0
    h.attrs['Flag_IC_Info'] = 0
    h.attrs['Flag_Metals'] = 0
    h.attrs['Flag_Sfr'] = 0 
    h.attrs['Flag_StellarAge'] = 0 

    h.attrs['NumPart_ThisFile'] = npart;
    h.attrs['NumPart_Total'] = npart; 
    h.attrs['NumPart_Total_HighWord'] = 0 * npart;

    h.attrs['HubbleParam'] = 1.0
    h.attrs['Omega0'] = 0.0
    h.attrs['OmegaLambda'] = 0.0
    h.attrs['Redshift'] = 0.0

    h.attrs['MassTable'] = np.zeros(6); 
    h.attrs['Time'] = 0.0; 
    h.attrs['NumFilesPerSnapshot'] = 1; 
    h.attrs['Flag_DoublePrecision'] = 1;

    p = file.create_group("PartType0")
    q = np.zeros((N, 3)); q[:, 0] = xarr; q[:, 1] = yarr; q[:, 2] = zarr
    p.create_dataset("Coordinates", data=q)

    q = np.zeros((N, 3)); q[:, 0] = vxarr; q[:, 1] = vyarr; q[:, 2] = vzarr
    p.create_dataset("Velocities", data=q)
    p.create_dataset("ParticleIDs", data=idarr)
    p.create_dataset("Masses", data=marr)

    p.create_dataset("InternalEnergy", data=enarr)

    # close the HDF5 file, which saves these outputs
    file.close()
    
    # all done!
    print("yeehaw")
    
