import numpy as np
from hyperion.model import AnalyticalYSOModel
from hyperion.util.constants import *


Nph = 1e4 # number of photons for model
m = AnalyticalYSOModel()

# Set the stellar parameters
m.star.radius = 2.0 * rsun
m.star.temperature = 6200
m.star.luminosity = 5.0 * lsun
m.star.mass = 0.5 * msun

# Add a flared disk
disk = m.add_flared_disk()
disk.mass = 0.01 * msun
disk.rmin = 10 * rsun
disk.rmax = 200 * au
disk.r_0 = m.star.radius
disk.h_0 = 0.01 * disk.r_0
disk.p = -1.0
disk.beta = 1.25
disk.dust = 'kmh_lite.hdf5'

# Add an Ulrich envelope
envelope = m.add_ulrich_envelope()
envelope.mass = 0.4 * msun
envelope.rc = disk.rmax
envelope.mdot = 5.e-6 * msun / yr
envelope.rmin = 200 * au
envelope.rmax = 10000 * au
envelope.p = -2.0
envelope.dust = 'kmh_lite.hdf5'

# Add a bipolar cavity
cavity = envelope.add_bipolar_cavity()
cavity.power = 1.5
cavity.theta_0 = 20
cavity.r_0 = envelope.rmax
cavity.rho_0 = 5e4 * 3.32e-24
cavity.rho_exp = 0.
cavity.dust = 'kmh_lite.hdf5'

# Use raytracing to improve s/n of thermal/source emission
m.set_raytracing(True)

# Use the modified random walk
m.set_mrw(True, gamma=2.)

# Set up grid
m.set_spherical_polar_grid_auto(299, 199, 10)

# Set up SED
sed = m.add_peeled_images(sed=True, image=False)
sed.set_viewing_angles([45.], [45.])
sed.set_wavelength_range(150, 0.02, 2000.)

# Set number of photons
m.set_n_photons(initial=Nph, imaging=Nph,
                raytracing_sources=Nph, raytracing_dust=Nph)

# Set number of temperature iterations and convergence criterion
m.set_n_initial_iterations(10)
m.set_convergence(True, percentile=99.0, absolute=2.0, relative=1.1)

# Write out file
m.write('yso_N1e4_grid0.rtin',overwrite=True)
m.run('yso_N1e4_grid0.rtout',mpi=False, overwrite=True)
