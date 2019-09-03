"""
This example uses the Merrifield model to calculate the effects of fluence
on population dynamics.
"""

from sfmodelling.spin import SpinHamiltonian
from sfmodelling.models import Merrifield

from matplotlib import pyplot as plt
import numpy as np


# magnetic stuff
sh = SpinHamiltonian()
sh.D = -5.8e-6
sh.E = sh.D/3
sh.X = 1e-10

model = Merrifield()

# rate constants
model.kGEN = 2
model.kSF = 5
model.k_SF = 2
model.kDISS = 0.4
model.kTTA = 1e-19
model.kRELAX = 0
model.kSNR = 0.1
model.kSSA = 0
model.kTTNR = 0.09
model.kTNR = 1e-5


# specify the initial species (singlet or triplet sensitization)
model.initial_species = 'triplet'


# euler angles defining the orientations of the molecular pairs within the unit cell
euler_angles = (0, np.pi/8, 0)


# time at which to normalise the population dynamics
t_norm = 4


# specifiy initial exciton densities (per cm3)
ieds = [1e16, 1e17, 1e18]


# perform the simulation, assuming a single crystal and no magnetic field
simulations = {}
cslsq = sh.calculate_overlap_specific_orientation(0, 0, 0, *euler_angles)
for i, ied in enumerate(ieds):
    model.GS_0 = ied
    model.simulate(cslsq)
    t, S1 = model.normalise_population_at(model.S1, t_norm)
    simulations[ied] = (t, S1)


# plot the results        
fig, axes = plt.subplots(figsize=(5, 4))
alphas = [0.3, 0.65, 1]
for i, ied in enumerate(ieds):
    axes.loglog(simulations[ied][0], simulations[ied][1], 'b-', alpha=alphas[i], label=r'10$^{{{0}}}$ cm$^{{-3}}$'.format(str(ied)[-2:]))
axes.legend(frameon=False)
axes.set_ylabel('singlet population (norm.)')
axes.set_xlabel('time (ns)')

