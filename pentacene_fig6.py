from tripletpairs.toolkit import convolve_irf, integrate_between
from sfmodelling.pentacene import PentaceneModel

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import usefulfunctions as uf


ptd = 8e15  # pulse energy to excitation density conversion factor

m = PentaceneModel()

# set the rate constants
m.kSF = 1e4  # singlet fission
m.k_SF = 0  # triplet-pair fusion
m.kSEP = 1e3  # triplet-pair separation
m.kTTA = 1.2e-19  # triplet-triplet annihilation
m.kTTNR = 0  # triplet-pair decay
m.kTNR = 2e-3  # triplet decay
m.kSNR = 0.011  # singlet decay
m.fTTA = 0.1  # fraction of TTA events that result in a triplet-pair

# set the initial condition
m.initial_weighting = {'S1': 1}  # start with a photoexcited singlet state


# Figure 6a
###############################################################################

pulses = ['4.63', '24.8', '46.3', '74.5', '91', '115']  # pulse energies

fig, ax = plt.subplots()
for i, pulse in enumerate(pulses):
    
    N0 = float(pulse)*ptd  # convert pulse energy to excitation density
    
    # do the simulation
    m.G = N0
    m.simulate()
    S1, TT, T1 = m.simulation_results['S1'], m.simulation_results['TT'], m.simulation_results['T1']
    
    triplet = TT+T1
    # convolve the simulation with the IRF
    t, triplet = convolve_irf(m.t, triplet, 1e-4)
    
    ax.plot(t*1000, triplet/max(triplet), color=plt.cm.tab10(i))
    
    if i == 0:
        simulation = pd.DataFrame(index=1000*t, columns=pulses)
        simulation[pulse] = triplet/max(triplet)
    else:
        simulation[pulse] = triplet/max(triplet)    
    
ax.set_xlim([-50, 1500])
ax.set_ylim([0.4, 1.1])

ax = uf.format_axes(ax, labelsize=18)
ax.set_xlabel('Time (ps)', fontsize=18)
ax.set_ylabel(r'$\Delta$T/T (norm.)', fontsize=18)
ax = uf.linearticks(ax, xmajorsep=400, xminorsep=100, ymajorsep=0.1, yminorsep=0.05)


# Figure 6b
###############################################################################

pulses = ['0.54', '1.7', '5.4', '17', '54', '130']  # pulse energies

fig, ax = plt.subplots()
for i, pulse in enumerate(pulses):

    N0 = float(pulse)*ptd  # convert pulse energy to excitation density
    
    # do the simulation
    m.G = N0
    m.simulate()
    TT, T1 = m.simulation_results['TT'], m.simulation_results['T1']
    
    triplet = TT+T1
    # convolve the simulation with the IRF
    t, triplet = convolve_irf(m.t, triplet, 1.5)
    

    normf = 7.256006182604989e-21  # factor to convert excitation density to delta T / T signal
    ax.loglog(t, triplet*normf, color=plt.cm.tab10(i))
    
    if i == 0:
        simulation = pd.DataFrame(index=t, columns=pulses)
        simulation[pulse] = triplet*normf
    else:
        simulation[pulse] = triplet*normf
    
ax.set_xlim([0.2, 800])
ax.set_ylim([1e-6, 2e-2])

ax = uf.format_axes(ax, labelsize=18)
ax.set_xlabel('Time (ns)', fontsize=18)
ax.set_ylabel(r'$\Delta$T/T', fontsize=18)
ax = uf.logticks(ax)


# Figure 6e
###############################################################################

timeranges = ['2_4', '5_10', '10_20', '20_50']

fig, axes = plt.subplots(nrows=1, ncols=4, gridspec_kw={'hspace': 0.1, 'wspace': 0.22}, figsize=(14, 3))

pulses = np.linspace(7.957747154594766, 138.15533254504803, 50)  # pulse energies
simulation = pd.DataFrame(index=pulses, columns=timeranges)    
for i, pulse in enumerate(pulses):
    
    # do the simulation
    m.G = ptd*pulse
    m.simulate()
    TT = m.simulation_results['TT']
    
    # convolve the simulation with the IRF
    t, TT = convolve_irf(m.t, TT, 1.5)
    
    for j, tr in enumerate(timeranges):
        simulation.loc[pulse, tr] = integrate_between(t, TT, float(tr.split('_')[0]), float(tr.split('_')[1]))

normfs = [
    14.369720908374074,
    18.813948059853974,
    9.150441284406645,
    6.645232230358948]  # normalisation factors to turn excitation density into measured PL counts

for i, tr in enumerate(timeranges):
    ax = axes[i]
    ax = uf.format_axes(ax, labelsize=18)
    ax.plot(simulation.index, normfs[i]*simulation[tr]/simulation[tr].max(), 'k--')
    ax.text(0.95, 0.05, '{0}-{1}ns'.format(tr.split('_')[0], tr.split('_')[1]), fontsize=18, transform=ax.transAxes, va='bottom', ha='right')
    ax = uf.format_axes(ax, labelsize=18)
    if i == 0:
        ax = uf.linearticks(ax, xmajorsep=40, xminorsep=10, ymajorsep=4, yminorsep=2)
        ax.set_ylabel(r'PL (counts/10$^4$)', fontsize=18)
    elif i == 1:
        ax = uf.linearticks(ax, xmajorsep=40, xminorsep=10, ymajorsep=4, yminorsep=2)
    elif i == 2:
        ax = uf.linearticks(ax, xmajorsep=40, xminorsep=10, ymajorsep=2, yminorsep=1)
    elif i == 3:
        ax = uf.linearticks(ax, xmajorsep=40, xminorsep=10, ymajorsep=2, yminorsep=1)
    ax.set_xlabel(r'Pulse Energy ($\mu$Jcm$^{-2}$)', fontsize=18)


# Figure 6d
###############################################################################

# do the simulation
pulse = 20  # measured power at 1kHz in uW
ptope = 2.2104853207207684  # factor to convert measured power (uW) to pulse energy (uJ/cm2) based on measured spot size and rep rate
m.G = ptd*pulse*ptope
m.simulate()
TT = m.simulation_results['TT']

# convolve the simulation with the IRF
t, TT = convolve_irf(m.t, TT, 1.5)

fig, ax = plt.subplots()

TT = TT*3.96229693e-09  # normalise the simulation to PL counts
ax.plot(t, TT, 'k--')

ax.set_yscale('log')
ax.set_xscale('symlog', linthreshx=4)

ax.set_ylim([1e0, 2e5])
ax.set_xlim([-4, 300])

ax = uf.format_axes(ax, labelsize=18)
ax.xaxis.set_major_locator(mticker.FixedLocator([-4, 0, 4, 10, 100]))
ax.xaxis.set_minor_locator(mticker.FixedLocator([-3, -2, -1, 1, 2, 3, 5, 6, 7, 8, 9, 20, 30, 40, 50, 60, 70, 80, 90, 200, 300]))
ax.set_xticklabels([-4, 0, 4, 10, 100])
ax = uf.logticks(ax, which='y')

ax.set_ylabel('PL (counts)', fontsize=18)
ax.set_xlabel('Time (ns)', fontsize=18)