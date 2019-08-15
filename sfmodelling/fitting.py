import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import least_squares
from scipy.interpolate import UnivariateSpline

from spin import SpinHamiltonian
from models import Merrifield, Bardeen, MerrifieldBardeen



class MerrifieldKineticFitting(object):
    '''
    description
    '''
    def __init__(self):
        self._rates_str = np.array(['kGEN', 'kSF', 'k_SF', 'kDISS', 'kTTA', 'kRELAX', 'kSNR', 'kSSA', 'kTTNR', 'kTNR'], dtype=str)
        self.normalise_time = 4
        
    def compute_cslsq_random_orientation(self, B, D, E, X):
        sh = SpinHamiltonian()
        sh.D, sh.E, sh.X = D, E, X
        self.cslsq = sh.calculate_overlap_random_orientation(B)
        
    def input_data(self, tdata, ydata):
        self.tdata = tdata
        self.ydata = ydata
        
    def input_rates(self, kGEN, kSF, k_SF, kDISS, kTTA, kRELAX, kSNR, kSSA, kTTNR, kTNR):
        self.rates = np.array([kGEN, kSF, k_SF, kDISS, kTTA, kRELAX, kSNR, kSSA, kTTNR, kTNR], dtype=np.float64)
        
    def simulate_exciton_population(self, exciton, rates, initial_species, initial_population):
        model = Merrifield()
        model.kGEN, model.kSF, model.k_SF, model.kDISS, model.kTTA, model.kRELAX, model.kSNR, model.kSSA, model.kTTNR, model.kTNR = rates
        model.initial_species, model.GS_0 = initial_species, initial_population
        model.simulate(self.cslsq)
        if exciton == 'S1':
            t, pop = model.normalise_population_at(model.S1, self.normalise_time)
        elif exciton == 'TT_bright':
            t, pop = model.normalise_population_at(model.TT_bright, self.normalise_time)
        elif exciton == 'TT_dark':
            t, pop = model.normalise_population_at(model.TT_dark, self.normalise_time)
        elif exciton == 'TT_total':
            t, pop = model.normalise_population_at(model.TT_total, self.normalise_time)
        elif exciton == 'T1':
            t, pop = model.normalise_population_at(model.T1, self.normalise_time)
        else:
            raise ValueError('invalid exciton')
        return t, pop
    
    def spline_simulation(self, exciton, rates, initial_species, initial_population):
        tmodel, ymodel = self.simulate_exciton_population(exciton, rates, initial_species, initial_population)
        ymodelspl = UnivariateSpline(tmodel, ymodel, s=0)
        ymodel = ymodelspl(self.tdata)
        return ymodel
    
    def log_residuals(self, exciton, rates, initial_species, initial_population):
        ymodel = self.spline_simulation(exciton, rates, initial_species, initial_population)
        return np.log(self.ydata)-np.log(ymodel)
    
    def _least_squares_function(self, exciton, fitted_rates, held_rates, mask, initial_species, initial_population):
        rates = self._reconstruct_rates_array(fitted_rates, held_rates, mask)
        residuals = self.log_residuals(exciton, rates, initial_species, initial_population)
        return residuals
    
    @staticmethod
    def _reconstruct_rates_array(fitted_rates, held_rates, mask):
        rates = np.zeros_like(mask, dtype=np.float64)
        f, h = 0, 0
        for i, b in enumerate(mask):
            if b:
                rates[i] = fitted_rates[f]
                f += 1
            else:
                rates[i] = held_rates[h]
                h += 1
        return rates
    
    def fit(self, exciton, initial_species, initial_population, rates2fit):
        mask = np.array([True if (i in rates2fit) else False for i in self._rates_str], dtype=bool)
        fitted_rates = least_squares(lambda x: self._least_squares_function(exciton, x, self.rates[np.invert(mask)], mask, initial_species, initial_population), self.rates[mask], bounds=(0, 1000), loss='linear').x
        self.rates = self._reconstruct_rates_array(fitted_rates, self.rates[np.invert(mask)], mask)
        self.tmodel, self.model = self.simulate_exciton_population(exciton, self.rates, initial_species, initial_population)
        self.rates2fit = rates2fit
        
    def print_rates(self, tofile=False):
        if tofile:
            f = open('Mrates.txt', 'w')
            stream = f
        else:
            print('\n')
            stream = None  # sys.stdout
        print('rates in per ns (* - fitted)\n', file=stream)
        for i, rate in enumerate(self._rates_str):
            print('{0}: {1:.3e}{2}'.format(rate, self.rates[i], '*' if rate in self.rates2fit else ''), file=stream)
        print('\n')
        if tofile:
            f.close()
            
    def plot(self):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.loglog(self.tdata, self.ydata, 'ro', mfc='None')
        ax.loglog(self.tmodel, self.model, 'k-')
        return fig, ax
        
    def save(self):
        table = np.vstack((self.tmodel, self.model))
        np.savetxt('Mfit.csv', np.transpose(table), delimiter=',', header='time (ns), model')
        
        
        
class BardeenKineticFitting(object):
    '''
    description
    '''
    def __init__(self):
        self._rates_str = np.array(['kGEN', 'kSF', 'k_SF', 'kHOP', 'k_HOP', 'kRELAX', 'kSNR', 'kSSA', 'kTTNR', 'kSPIN'], dtype=str)
        self.normalise_time = 4
        
    def compute_cslsq_random_orientation(self, B, D, E, X):
        sh = SpinHamiltonian()
        sh.D, sh.E, sh.X = D, E, X
        self.cslsq = sh.calculate_overlap_random_orientation(B)
        
    def input_data(self, tdata, ydata):
        self.tdata = tdata
        self.ydata = ydata
        
    def input_rates(self, kGEN, kSF, k_SF, kHOP, k_HOP, kRELAX, kSNR, kSSA, kTTNR, kSPIN):
        self.rates = np.array([kGEN, kSF, k_SF, kHOP, k_HOP, kRELAX, kSNR, kSSA, kTTNR, kSPIN], dtype=np.float64)
        
    def simulate_exciton_population(self, exciton, rates, initial_population):
        model = Bardeen()
        model.kGEN, model.kSF, model.k_SF, model.kHOP, model._HOP, model.kRELAX, model.kSNR, model.kSSA, model.kTTNR, model.SPIN = rates
        model.GS_0 = initial_population
        model.simulate(self.cslsq)
        if exciton == 'S1':
            t, pop = model.normalise_population_at(model.S1, self.normalise_time)
        elif exciton == 'TT_bright':
            t, pop = model.normalise_population_at(model.TT_bright, self.normalise_time)
        elif exciton == 'TT_dark':
            t, pop = model.normalise_population_at(model.TT_dark, self.normalise_time)
        elif exciton == 'TT_total':
            t, pop = model.normalise_population_at(model.TT_total, self.normalise_time)
        elif exciton == 'T_T_total':
            t, pop = model.normalise_population_at(model.T_T_total, self.normalise_time)
        else:
            raise ValueError('invalid exciton')
        return t, pop
    
    def spline_simulation(self, exciton, rates, initial_population):
        tmodel, ymodel = self.simulate_exciton_population(exciton, rates, initial_population)
        ymodelspl = UnivariateSpline(tmodel, ymodel, s=0)
        ymodel = ymodelspl(self.tdata)
        return ymodel
    
    def log_residuals(self, exciton, rates, initial_population):
        ymodel = self.spline_simulation(exciton, rates, initial_population)
        return np.log(self.ydata)-np.log(ymodel)
    
    def _least_squares_function(self, exciton, fitted_rates, held_rates, mask, initial_population):
        rates = self._reconstruct_rates_array(fitted_rates, held_rates, mask)
        residuals = self.log_residuals(exciton, rates, initial_population)
        return residuals
    
    @staticmethod
    def _reconstruct_rates_array(fitted_rates, held_rates, mask):
        rates = np.zeros_like(mask, dtype=np.float64)
        f, h = 0, 0
        for i, b in enumerate(mask):
            if b:
                rates[i] = fitted_rates[f]
                f += 1
            else:
                rates[i] = held_rates[h]
                h += 1
        return rates
    
    def fit(self, exciton, initial_population, rates2fit):
        mask = np.array([True if (i in rates2fit) else False for i in self._rates_str], dtype=bool)
        fitted_rates = least_squares(lambda x: self._least_squares_function(exciton, x, self.rates[np.invert(mask)], mask, initial_population), self.rates[mask], bounds=(0, 1000), loss='linear').x
        self.rates = self._reconstruct_rates_array(fitted_rates, self.rates[np.invert(mask)], mask)
        self.tmodel, self.model = self.simulate_exciton_population(exciton, self.rates, initial_population)
        self.rates2fit = rates2fit
        
    def print_rates(self, tofile=False):
        if tofile:
            f = open('Brates.txt', 'w')
            stream = f
        else:
            print('\n')
            stream = None  # sys.stdout
        print('rates in per ns (* - fitted)\n', file=stream)
        for i, rate in enumerate(self._rates_str):
            print('{0}: {1:.3e}{2}'.format(rate, self.rates[i], '*' if rate in self.rates2fit else ''), file=stream)
        print('\n')
        if tofile:
            f.close()
            
    def plot(self):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.loglog(self.tdata, self.ydata, 'ro', mfc='None')
        ax.loglog(self.tmodel, self.model, 'k-')
        return fig, ax
        
    def save(self):
        table = np.vstack((self.tmodel, self.model))
        np.savetxt('Bfit.csv', np.transpose(table), delimiter=',', header='time (ns), model')
        
        


class MerrifieldBardeenKineticFitting(object):
    '''
    description
    '''
    def __init__(self):
        self._rates_str = np.array(['kGEN', 'kSF', 'k_SF', 'kHOP', 'k_HOP', 'kDISS', 'kTTA', 'kRELAX', 'kSNR', 'kSSA', 'kTTNR', 'kSPIN', 'kTNR'], dtype=str)
        self.normalise_time = 4
        
    def compute_cslsq_random_orientation(self, B, D, E, X):
        sh = SpinHamiltonian()
        sh.D, sh.E, sh.X = D, E, X
        self.cslsq = sh.calculate_overlap_random_orientation(B)
        
    def input_data(self, tdata, ydata):
        self.tdata = tdata
        self.ydata = ydata
        
    def input_rates(self, kGEN, kSF, k_SF, kHOP, k_HOP, kDISS, kTTA, kRELAX, kSNR, kSSA, kTTNR, kSPIN, kTNR):
        self.rates = np.array([kGEN, kSF, k_SF, kHOP, k_HOP, kDISS, kTTA, kRELAX, kSNR, kSSA, kTTNR, kSPIN, kTNR], dtype=np.float64)
        
    def simulate_exciton_population(self, exciton, rates, initial_species, initial_population):
        model = MerrifieldBardeen()
        model.kGEN, model.kSF, model.k_SF, model.kHOP, model.k_HOP, model.kDISS, model.kTTA, model.kRELAX, model.kSNR, model.kSSA, model.kTTNR, model.kSPIN, model.kTNR = rates
        model.initial_species, model.GS_0 = initial_species, initial_population
        model.simulate(self.cslsq)
        if exciton == 'S1':
            t, pop = model.normalise_population_at(model.S1, self.normalise_time)
        elif exciton == 'TT_bright':
            t, pop = model.normalise_population_at(model.TT_bright, self.normalise_time)
        elif exciton == 'TT_dark':
            t, pop = model.normalise_population_at(model.TT_dark, self.normalise_time)
        elif exciton == 'TT_total':
            t, pop = model.normalise_population_at(model.TT_total, self.normalise_time)
        elif exciton == 'T_T_total':
            t, pop = model.normalise_population_at(model.T_T_total, self.normalise_time)
        elif exciton == 'T1':
            t, pop = model.normalise_population_at(model.T1, self.normalise_time)
        else:
            raise ValueError('invalid exciton')
        return t, pop
    
    def spline_simulation(self, exciton, rates, initial_species, initial_population):
        tmodel, ymodel = self.simulate_exciton_population(exciton, rates, initial_species, initial_population)
        ymodelspl = UnivariateSpline(tmodel, ymodel, s=0)
        ymodel = ymodelspl(self.tdata)
        return ymodel
    
    def log_residuals(self, exciton, rates, initial_species, initial_population):
        ymodel = self.spline_simulation(exciton, rates, initial_species, initial_population)
        return np.log(self.ydata)-np.log(ymodel)
    
    def _least_squares_function(self, exciton, fitted_rates, held_rates, mask, initial_species, initial_population):
        rates = self._reconstruct_rates_array(fitted_rates, held_rates, mask)
        residuals = self.log_residuals(exciton, rates, initial_species, initial_population)
        return residuals
    
    @staticmethod
    def _reconstruct_rates_array(fitted_rates, held_rates, mask):
        rates = np.zeros_like(mask, dtype=np.float64)
        f, h = 0, 0
        for i, b in enumerate(mask):
            if b:
                rates[i] = fitted_rates[f]
                f += 1
            else:
                rates[i] = held_rates[h]
                h += 1
        return rates
    
    def fit(self, exciton, initial_species, initial_population, rates2fit):
        mask = np.array([True if (i in rates2fit) else False for i in self._rates_str], dtype=bool)
        fitted_rates = least_squares(lambda x: self._least_squares_function(exciton, x, self.rates[np.invert(mask)], mask, initial_species, initial_population), self.rates[mask], bounds=(0, 1000), loss='linear').x
        self.rates = self._reconstruct_rates_array(fitted_rates, self.rates[np.invert(mask)], mask)
        self.tmodel, self.ymodel = self.simulate_exciton_population(exciton, self.rates, initial_species, initial_population)
        self.rates2fit = rates2fit
        
    def print_rates(self, tofile=False):
        if tofile:
            f = open('MBrates.txt', 'w')
            stream = f
        else:
            print('\n')
            stream = None  # sys.stdout
        print('rates in per ns (* - fitted)\n', file=stream)
        for i, rate in enumerate(self._rates_str):
            print('{0}: {1:.3e}{2}'.format(rate, self.rates[i], '*' if rate in self.rates2fit else ''), file=stream)
        print('\n')
        if tofile:
            f.close()
            
    def plot(self):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.loglog(self.tdata, self.ydata, 'ro', mfc='None')
        ax.loglog(self.tmodel, self.ymodel, 'k-')
        return fig, ax
        
    def save(self):
        table = np.vstack((self.tmodel, self.model))
        np.savetxt('MBfit.csv', np.transpose(table), delimiter=',', header='time (ns), model')       