# sfmodelling
Kinetic modelling and magnetic field effect for singlet fission systems.

### Example usage
```python
from sfmodelling.spin import SpinHamiltonian
from sfmodelling.models import Merrifield

# magnetic stuff
sh = SpinHamiltonian()
sh.D = 5.4e-6
sh.E = 1.2e-7
sh.X = 1e-8
sh.rAB = (0, 0, 1)

mag_field_strength = 0.1  # 100mT
mag_field_orientation = (0.1, -0.5)  # (theta, phi)
euler_angles = (0, 0, 0)  # for example, one molecule per unit cell

cslsq = sh.calculate_overlap_specific_orientation(mag_field_strength, *mag_field_orientation, *euler_angles)

# kinetic stuff
model = Merrifield()

# set rate constants (units per ns, except kTTA which is cm^3/ns)
model.kGEN = 1e4
model.kSF = 1e4
model.k_SF = 1e-1
model.kRELAX = 0
model.kSNR = 1e-1
model.kSSA = 0
model.kTTNR = 1e-2
model.kDISS = 1e-2
model.kTTA = 1e-20
model.kTNR = 1e-6

model.GS_0 = 1e17  # initial excitation density in number cm3

# simulate and plot the dynamics
from matplotlib import pyplot as plt

model.simulate(cslsq)

singlet_population, normalisation_factor = model.normalise_population_at(model.S1, 1)  # normalise singlet to 1ns
triplet_pair_population = model.TT_total / normalisation_factor
free_triplet_population = model.T1 / normalisation_factor

plt.figure()
plt.loglog(model.t, singlet_population, 'k-', label=r'S$_1$')
plt.loglog(model.t, triplet_pair_population, 'b-', label=r'T..T')
plt.loglog(model.t, free_triplet_population, 'r-', label=r'T$_1$')
plt.legend(frameon=False)
plt.xlim([1, max(model.t)])
plt.ylim([1e-7, 1e6])
plt.xlabel('Time (ns)')
plt.ylabel('Population (norm.)')

```
