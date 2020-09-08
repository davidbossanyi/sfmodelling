import os

from sfmodelling.spin import SpinHamiltonian
from sfmodelling.diftes import MerrifieldExplicit1TT

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import usefulfunctions as uf


# Figure 4b (or S16)
###############################################################################

# choose the temperature to simulate
temperature = '100K'

# import the rate constants
rates = pd.read_csv(os.path.join('rates', 'diftes_rates_fig4b_figS16.csv'), header=0, index_col=0)
rates = rates[temperature]

# time at which to normlise the kinetics
normalise_time = 4

# set up the spin Hamiltonian and calculate the overlap factors
sh = SpinHamiltonian()
sh.D = 1.08e-6  # zero field splitting D parameter
sh.E = 2.98e-6  # zero field splitting E parameter
sh.X = 6e-8  # intermolecular triplet dipole-dipole interaction
sh.rAB = (0.7636, -0.4460, 0.4669)  # vector joining molecule A to molecule B, in molecular coordinates of A
cslsq = sh.calculate_overlap_specific_orientation(0, 0, 0, 0, 0, 0)  # Euler angles are all zero (parallel molecules) and there is no magnetic field

model = MerrifieldExplicit1TT()

model.initial_species = 'singlet'  # photoexcited singlet to begin with

model.TTA_channel = 1  # triplet-triplet annihilation produces (T..T)

# set the rate constants
model.kGEN = rates['kGEN']
model.kSF = rates['kSF']
model.k_SF = rates['k_SF']
model.kHOP = rates['kHOP']
model.k_HOP = rates['k_HOP']
model.kRELAX = rates['kRELAX']
model.kHOP2 = rates['kHOP2']
model.kTTA = rates['kTTA']
model.kSNR = rates['kSNR']
model.kSSA = rates['kSSA']
model.kTTNR = rates['kTTNR']
model.kTNR = rates['kTNR']

# plot the simulations
fig, ax = plt.subplots()
alphas = np.linspace(0.2, 1, 3)
N0s = [1e17, 1e18, 1e19]
for i, N0 in enumerate(N0s):
    model.GS_0 = N0
    
    # do the simulation
    model.simulate(cslsq)
    
    # normalise the kinetics
    ttmodel, factor = model.normalise_population_at(model.TT, normalise_time)
    
    # plot simulation
    ax.loglog(model.t, ttmodel, 'b-', alpha=alphas[i], label=uf.fsci(N0)+r' cm$^{-3}$')
    ax.set_xlim([1, 1e5])
    ax.set_ylim([1e-7, 2])

ax.set_xlabel('Time (ns)', fontsize=18)
ax.set_ylabel('PL intensity (norm.)', fontsize=18)
ax = uf.format_axes(ax, labelsize=18)
ax = uf.logticks(ax)
ax.legend(frameon=False, fontsize=18)


# Figure 4c
###############################################################################

# import the rate constants
rates = pd.read_csv(os.path.join('rates', 'diftes_rates_fig4c.csv'), header=None, index_col=0, squeeze=True)

# D and E parameters tweaked within reported errors (Yong 2017)
sh.D = 0.74e-6
sh.E = 3.1e-6

# angles of B-field vector in molecular coordinates
phi = 2.364
theta = -0.936

# B-field
Bs = np.linspace(0, 0.26, 61)  # units of Tesla

model = MerrifieldExplicit1TT()
model.GS_0 = 7.2e17  # calculated from measured power, spot size, diftes absorbance spectrum etc.

model.initial_species = 'singlet'  # photoexcited singlet to begin with

model.TTA_channel = 1  # triplet-triplet annihilation produces (T..T)

# set the rate constants
model.kGEN = rates['kGEN']
model.kSF = rates['kSF']
model.k_SF = rates['k_SF']
model.kHOP = rates['kHOP']
model.k_HOP = rates['k_HOP']
model.kRELAX = rates['kRELAX']
model.kHOP2 = rates['kHOP2']
model.kTTA = rates['kTTA']
model.kSNR = rates['kSNR']
model.kSSA = rates['kSSA']
model.kTTNR = rates['kTTNR']
model.kTNR = rates['kTNR']

# do the MFE simulation
simulation1 = np.zeros_like(Bs)
simulation2 = np.zeros_like(Bs)
for i, B in enumerate(Bs):
    cslsq = sh.calculate_overlap_specific_orientation(B, theta, phi, 0, 0, 0)
    model.simulate(cslsq)
    PL = model.S1+(1/40)*model.TT  # ratio of radiative rates is 40:1 (Yong 2017)
    PL1 = model.get_population_between(PL, 20, 30)  # 20-30ns window
    PL2 = model.get_population_between(PL, 100, 200)  # 100-200ns window
    simulation1[i] = PL1
    simulation2[i] = PL2
# calculate delta PL / PL
simulation1 = 100*(simulation1-simulation1[0])/simulation1[0]
simulation2 = 100*(simulation2-simulation2[0])/simulation2[0]

# plot the results
fig, ax = plt.subplots()

ax.plot(1000*Bs, simulation1, color='dodgerblue', label='20-30ns')
ax.plot(1000*Bs, simulation2, color='darkorange', label='100-200ns')

ax.set_xlabel('Magnetic field strength (mT)', fontsize=18)
ax.set_ylabel(r'$\Delta$PL/PL (%)', fontsize=18, labelpad=-10)
ax = uf.format_axes(ax, labelsize=18)
ax = uf.linearticks(ax, xmajorsep=50, xminorsep=10, ymajorsep=10, yminorsep=1)
ax.legend(frameon=False, fontsize=18, loc=(0.2, 0.7))
ax.axhline(0, color='0.5', linewidth=1)

