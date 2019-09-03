"""
This example uses the Bardeen model to calculate the effects of magnetic field
on photoluminescence at various times following photoexcitation.
"""

from sfmodelling.spin import SpinHamiltonian
from sfmodelling.models import Bardeen

from matplotlib import pyplot as plt
import numpy as np


# magnetic stuff
sh = SpinHamiltonian()
sh.D = -5.8e-6
sh.E = sh.D/3
sh.X = 1e-10


model = Bardeen()

# rate constants
model.kGEN = 2
model.kSF = 1/0.05
model.k_SF = 1/35
model.kHOP = 1/15
model.k_HOP = 1/4000
model.kRELAX = 1/30
model.kSNR = 1/10
model.kSSA = 0
model.kTTNR = 1/15
model.kSPIN = 1/4000


# initial condition (arbitrary because the Bardeen model is linear)
model.GS_0 = 1


# magnetic field strengths to sample (units: Tesla)
Bs = np.linspace(0, 0.2, 201)


# time ranges in whuch to calculate the effect
timeranges = [(0, 2), (20, 30), (200, 400)]


# euler angles defining the orientations of the molecular pairs within the unit cell
euler_angles = [(0, 0, 0), (0, np.pi/4, 0)]


# perform the simulation, assuming a polycrystalline morphology
simulations = np.zeros((len(Bs), len(timeranges)))
for i, B in enumerate(Bs):
    for a, angles in enumerate(euler_angles):
        cslsq = sh.calculate_overlap_semirandom_orientation(B, *angles, tofile=False)
        model.simulate(cslsq)
        for j, tr in enumerate(timeranges):
            S1 = model.get_population_between('S1', tr[0], tr[1])
            simulations[i, j] = (simulations[i, j]+S1)/(a+1)


# plot the results        
fig, axes = plt.subplots(nrows=1, ncols=len(timeranges), sharex=True, sharey=True, gridspec_kw={'hspace': 0, 'wspace': 0.1}, figsize=(4*len(timeranges), 3))
for j, tr in enumerate(timeranges):
    axes[j].plot(1000*Bs, 100*(simulations[:, j]-simulations[0, j])/simulations[0, j], 'b-')
    axes[j].set_xlim([0, 200])
    axes[j].set_ylim([-50, 50])
    axes[j].axhline(0, color='0.5', linewidth=1)
    axes[j].set_xlabel('Magnetic Field Strength (mT)')
    axes[j].text(190, 40, '{0}-{1} ns'.format(tr[0], tr[1]), ha='right')
axes[0].set_ylabel(r'$\Delta$PL/PL (%)')
