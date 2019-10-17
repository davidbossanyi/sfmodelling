'''
Contains classes for simulating excited state dynamics of singlet fission
systems. So far, there are 3 classes: Merrifield, Bardeen and a kind of
combination of the two MerrifieldBardeen.

Version 1.1
David Bossanyi 17/10/2019
dgbossanyi1@sheffield.ac.uk
'''

import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt


class Merrifield(object):
    '''
    Solves the Merrifield model for a given set of cslsq values.
    '''
    def __init__(self):
        self._define_default_parameters()
    
    def _define_default_parameters(self):
        # generation of photoexcited singlet to model IRF
        self.kGEN = 0.25
        # rates between excited states
        self.kSF = 20.0
        self.k_SF = 0.03
        self.kDISS = 0.067
        self.kTTA = 1e-18
        # spin relaxation (Bardeen addition - not in original Merrifield)
        self.kRELAX = 0
        # rates of decay
        self.kSNR = 0.1
        self.kSSA = 0
        self.kTTNR = 0.067
        self.kTNR = 1e-5
        # initial population (per cm-3)
        self.GS_0 = 8e17
        self.initial_species = 'singlet'
        # bright threshold: if |Cs|^2 > threshold TT state is considered bright
        self.bright_threshold = 1e-4
        # time axis
        self.t = np.logspace(-5, 5, 10000)
        
    def _set_generation_rates(self):
        if self.initial_species == 'singlet':
            self.kGENS = self.kGEN
            self.kGENT = 0
        elif self.initial_species == 'triplet':
            self.kGENS = 0
            self.kGENT = self.kGEN
        else:
            raise ValueError('initial_species attribute must be either \'singlet\' or \'triplet\'')

    def _rate_equations(self, t, y, Cslsq):
        GS, S1, TT_1, TT_2, TT_3, TT_4, TT_5, TT_6, TT_7, TT_8, TT_9, T1 = y
        dydt = np.zeros(len(y))
        # GS
        dydt[0] = -(self.kGENS+self.kGENT)*GS
        # S1
        dydt[1] = self.kGENS*GS - (self.kSNR+self.kSF*np.sum(Cslsq))*S1 -self.kSSA*S1*S1+ self.k_SF*(Cslsq[0]*TT_1+Cslsq[1]*TT_2+Cslsq[2]*TT_3+Cslsq[3]*TT_4+Cslsq[4]*TT_5+Cslsq[5]*TT_6+Cslsq[6]*TT_7+Cslsq[7]*TT_8+Cslsq[8]*TT_9)
        # TT_1
        dydt[2] = self.kSF*Cslsq[0]*S1 - (self.k_SF*Cslsq[0]+self.kDISS+self.kTTNR+self.kRELAX)*TT_1 + (1/9)*self.kTTA*T1*T1 + (1/8)*self.kRELAX*(TT_2+TT_3+TT_4+TT_5+TT_6+TT_7+TT_8+TT_9)
        # TT_2
        dydt[3] = self.kSF*Cslsq[1]*S1 - (self.k_SF*Cslsq[1]+self.kDISS+self.kTTNR+self.kRELAX)*TT_2 + (1/9)*self.kTTA*T1*T1 + (1/8)*self.kRELAX*(TT_1+TT_3+TT_4+TT_5+TT_6+TT_7+TT_8+TT_9)
        # TT_3
        dydt[4] = self.kSF*Cslsq[2]*S1 - (self.k_SF*Cslsq[2]+self.kDISS+self.kTTNR+self.kRELAX)*TT_3 + (1/9)*self.kTTA*T1*T1 + (1/8)*self.kRELAX*(TT_1+TT_2+TT_4+TT_5+TT_6+TT_7+TT_8+TT_9)
        # TT_4
        dydt[5] = self.kSF*Cslsq[3]*S1 - (self.k_SF*Cslsq[3]+self.kDISS+self.kTTNR+self.kRELAX)*TT_4 + (1/9)*self.kTTA*T1*T1 + (1/8)*self.kRELAX*(TT_1+TT_2+TT_3+TT_5+TT_6+TT_7+TT_8+TT_9)
        # TT_5
        dydt[6] = self.kSF*Cslsq[4]*S1 - (self.k_SF*Cslsq[4]+self.kDISS+self.kTTNR+self.kRELAX)*TT_5 + (1/9)*self.kTTA*T1*T1 + (1/8)*self.kRELAX*(TT_1+TT_2+TT_3+TT_4+TT_6+TT_7+TT_8+TT_9)
        # TT_6
        dydt[7] = self.kSF*Cslsq[5]*S1 - (self.k_SF*Cslsq[5]+self.kDISS+self.kTTNR+self.kRELAX)*TT_6 + (1/9)*self.kTTA*T1*T1 + (1/8)*self.kRELAX*(TT_1+TT_2+TT_3+TT_4+TT_5+TT_7+TT_8+TT_9)
        # TT_7
        dydt[8] = self.kSF*Cslsq[6]*S1 - (self.k_SF*Cslsq[6]+self.kDISS+self.kTTNR+self.kRELAX)*TT_7 + (1/9)*self.kTTA*T1*T1 + (1/8)*self.kRELAX*(TT_1+TT_2+TT_3+TT_4+TT_5+TT_6+TT_8+TT_9)
        # TT_8
        dydt[9] = self.kSF*Cslsq[7]*S1 - (self.k_SF*Cslsq[7]+self.kDISS+self.kTTNR+self.kRELAX)*TT_8 + (1/9)*self.kTTA*T1*T1 + (1/8)*self.kRELAX*(TT_1+TT_2+TT_3+TT_4+TT_5+TT_6+TT_7+TT_9)
        # TT_9
        dydt[10] = self.kSF*Cslsq[8]*S1 - (self.k_SF*Cslsq[8]+self.kDISS+self.kTTNR+self.kRELAX)*TT_9 + (1/9)*self.kTTA*T1*T1 + (1/8)*self.kRELAX*(TT_1+TT_2+TT_3+TT_4+TT_5+TT_6+TT_7+TT_8)
        # T1
        dydt[11] = self.kGENT*GS + 2.0*self.kDISS*(TT_1+TT_2+TT_3+TT_4+TT_5+TT_6+TT_7+TT_8+TT_9) - 2.0*self.kTTA*T1*T1 - self.kTNR*T1
        #
        return dydt
    
    def simulate(self, Cslsq):
        self._set_generation_rates()
        y0 = np.array([self.GS_0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        y = odeint(lambda y, t: self._rate_equations(t, y, Cslsq), y0, self.t)
        self._unpack_simulation(y, Cslsq)

    def _unpack_simulation(self, y, Cslsq):
        self.GS = y[:, 0]
        self.S1 = y[:, 1]
        self.TT_bright = np.sum(y[:, np.where(Cslsq > self.bright_threshold)[0]+2], axis=1)
        self.TT_dark = np.sum(y[:, np.where(Cslsq <= self.bright_threshold)[0]+2], axis=1)
        self.TT_total = np.sum(y[:, 2:11], axis=1)
        self.T1 = y[:, 11]
        self.num_TT_bright = len(Cslsq[Cslsq > self.bright_threshold])
        self._integrate_populations()
        
    def plot_simulation(self):
        fig = plt.figure(figsize=(6, 5))
        plt.loglog(self.t, self.S1, 'b-', label=r'S$_1$')
        plt.loglog(self.t, self.TT_bright, 'r-', label=r'TT$_{bright}$')
        plt.loglog(self.t, self.TT_dark, 'k-', label=r'TT$_{dark}$')
        plt.loglog(self.t, self.TT_total, 'm-', label=r'TT$_{total}$')
        plt.loglog(self.t, self.T1, 'g-', label=r'T$_1$')
        plt.xlabel('time (ns)')
        plt.ylabel(r'population (cm$^{-3}$)')
        plt.legend()
        plt.xlim([1e-4, 1e5])
        plt.ylim([1e-8, 1e1])
        return fig
    
    def _integrate_populations(self):
        self.GS_int = np.trapz(self.GS, x=self.t)
        self.S1_int = np.trapz(self.S1, x=self.t)
        self.TT_bright_int = np.trapz(self.TT_bright, x=self.t)
        self.TT_dark_int = np.trapz(self.TT_dark, x=self.t)
        self.TT_total_int = np.trapz(self.TT_total, x=self.t)
        self.T1_int = np.trapz(self.T1, x=self.t)
        
    def get_population_between(self, species, t1, t2):
        mask = ((self.t >= t1) & (self.t <= t2))
        t = self.t[mask]
        if species == 'GS':
            population = np.trapz(self.GS[mask], x=t)/(t2-t1)
        elif species == 'S1':
            population = np.trapz(self.S1[mask], x=t)/(t2-t1)
        elif species == 'TT_bright':
            population = np.trapz(self.TT_bright[mask], x=t)/(t2-t1)
        elif species == 'TT_dark':
            population = np.trapz(self.TT_dark[mask], x=t)/(t2-t1)
        elif species == 'TT_total':
            population = np.trapz(self.TT_total[mask], x=t)/(t2-t1)
        elif species == 'T1':
            population = np.trapz(self.T1[mask], x=t)/(t2-t1)
        else:
            raise ValueError('\'species\' must be one of \'GS\', \'S1\', \'TT_bright\', \'TT_dark\', \'TT_total\', \'T1\'')
        return population
    
    def normalise_population_at(self, species, t):
        idx = np.where((self.t-t)**2 == min((self.t-t)**2))[0][0]
        factor = species[idx]
        species = species/factor
        return species, factor
        
        

class Bardeen(object):
    '''
    Solves the Bardeen model for a given set of cslsq values.
    '''
    def __init__(self):
        self._define_default_parameters()
    
    def _define_default_parameters(self):
        # generation of photoexcited singlet to model IRF
        self.kGEN = 0.25
        # rates between excited states
        self.kSF = 20.0
        self.k_SF = 0.03
        self.kHOP = 0.067
        self.k_HOP = 2.5e-4
        # spin relaxation
        self.kRELAX = 0.033
        # rates of decay
        self.kSNR = 0.1
        self.kSSA =0
        self.kTTNR = 0.067
        self.kSPIN = 2.5e-4
        # initial population (arb.)
        self.GS_0 = 1
        # bright threshold: if |Cs|^2 > threshold TT state is considered bright
        self.bright_threshold = 1e-4
        # time axis
        self.t = np.logspace(-5, 5, 10000)

    def _rate_equations(self, t, y, Cslsq):
        '''
        Note that the TT non-radiative rate is multiplied by Cslsq also here....
        '''
        GS, S1, TT_1, TT_2, TT_3, TT_4, TT_5, TT_6, TT_7, TT_8, TT_9, T_T_1, T_T_2, T_T_3, T_T_4, T_T_5, T_T_6, T_T_7, T_T_8, T_T_9 = y
        dydt = np.zeros(len(y))
        # GS
        dydt[0] = -self.kGEN*GS
        # S1
        dydt[1] = self.kGEN*GS - (self.kSNR+self.kSF*np.sum(Cslsq))*S1 -self.kSSA*S1*S1 + self.k_SF*(Cslsq[0]*TT_1+Cslsq[1]*TT_2+Cslsq[2]*TT_3+Cslsq[3]*TT_4+Cslsq[4]*TT_5+Cslsq[5]*TT_6+Cslsq[6]*TT_7+Cslsq[7]*TT_8+Cslsq[8]*TT_9)
        # TT_1
        dydt[2] = self.kSF*Cslsq[0]*S1 - (self.k_SF+self.kTTNR)*Cslsq[0]*TT_1 - self.kHOP*TT_1 + self.k_HOP*T_T_1
        # TT_2
        dydt[3] = self.kSF*Cslsq[1]*S1 - (self.k_SF+self.kTTNR)*Cslsq[1]*TT_2 - self.kHOP*TT_2 + self.k_HOP*T_T_2
        # TT_3
        dydt[4] = self.kSF*Cslsq[2]*S1 - (self.k_SF+self.kTTNR)*Cslsq[2]*TT_3 - self.kHOP*TT_3 + self.k_HOP*T_T_3
        # TT_4
        dydt[5] = self.kSF*Cslsq[3]*S1 - (self.k_SF+self.kTTNR)*Cslsq[3]*TT_4 - self.kHOP*TT_4 + self.k_HOP*T_T_4
        # TT_5
        dydt[6] = self.kSF*Cslsq[4]*S1 - (self.k_SF+self.kTTNR)*Cslsq[4]*TT_5 - self.kHOP*TT_5 + self.k_HOP*T_T_5
        # TT_6
        dydt[7] = self.kSF*Cslsq[5]*S1 - (self.k_SF+self.kTTNR)*Cslsq[5]*TT_6 - self.kHOP*TT_6 + self.k_HOP*T_T_6
        # TT_7
        dydt[8] = self.kSF*Cslsq[6]*S1 - (self.k_SF+self.kTTNR)*Cslsq[6]*TT_7 - self.kHOP*TT_7 + self.k_HOP*T_T_7
        # TT_8
        dydt[9] = self.kSF*Cslsq[7]*S1 - (self.k_SF+self.kTTNR)*Cslsq[7]*TT_8 - self.kHOP*TT_8 + self.k_HOP*T_T_8
        # TT_9
        dydt[10] = self.kSF*Cslsq[8]*S1 - (self.k_SF+self.kTTNR)*Cslsq[8]*TT_9 - self.kHOP*TT_9 + self.k_HOP*T_T_9
        # T_T_1
        dydt[11] = self.kHOP*TT_1 - (self.k_HOP+self.kSPIN+self.kRELAX)*T_T_1 + (1/8)*self.kRELAX*(T_T_2+T_T_3+T_T_4+T_T_5+T_T_6+T_T_7+T_T_8+T_T_9)
        # T_T_2
        dydt[12] = self.kHOP*TT_2 - (self.k_HOP+self.kSPIN+self.kRELAX)*T_T_2 + (1/8)*self.kRELAX*(T_T_1+T_T_3+T_T_4+T_T_5+T_T_6+T_T_7+T_T_8+T_T_9)
        # T_T_3
        dydt[13] = self.kHOP*TT_3 - (self.k_HOP+self.kSPIN+self.kRELAX)*T_T_3 + (1/8)*self.kRELAX*(T_T_1+T_T_2+T_T_4+T_T_5+T_T_6+T_T_7+T_T_8+T_T_9)
        # T_T_4
        dydt[14] = self.kHOP*TT_4 - (self.k_HOP+self.kSPIN+self.kRELAX)*T_T_4 + (1/8)*self.kRELAX*(T_T_1+T_T_2+T_T_3+T_T_5+T_T_6+T_T_7+T_T_8+T_T_9)
        # T_T_5
        dydt[15] = self.kHOP*TT_5 - (self.k_HOP+self.kSPIN+self.kRELAX)*T_T_5 + (1/8)*self.kRELAX*(T_T_1+T_T_2+T_T_3+T_T_4+T_T_6+T_T_7+T_T_8+T_T_9)
        # T_T_6
        dydt[16] = self.kHOP*TT_6 - (self.k_HOP+self.kSPIN+self.kRELAX)*T_T_6 + (1/8)*self.kRELAX*(T_T_1+T_T_2+T_T_3+T_T_4+T_T_5+T_T_7+T_T_8+T_T_9)
        # T_T_7
        dydt[17] = self.kHOP*TT_7 - (self.k_HOP+self.kSPIN+self.kRELAX)*T_T_7 + (1/8)*self.kRELAX*(T_T_1+T_T_2+T_T_3+T_T_4+T_T_5+T_T_6+T_T_8+T_T_9)
        # T_T_8
        dydt[18] = self.kHOP*TT_8 - (self.k_HOP+self.kSPIN+self.kRELAX)*T_T_8 + (1/8)*self.kRELAX*(T_T_1+T_T_2+T_T_3+T_T_4+T_T_5+T_T_6+T_T_7+T_T_9)
        # T_T_9
        dydt[19] = self.kHOP*TT_9 - (self.k_HOP+self.kSPIN+self.kRELAX)*T_T_9 + (1/8)*self.kRELAX*(T_T_1+T_T_2+T_T_3+T_T_4+T_T_5+T_T_6+T_T_7+T_T_8)
        #
        return dydt
    
    def simulate(self, Cslsq):
        y0 = np.array([self.GS_0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        y = odeint(lambda y, t: self._rate_equations(t, y, Cslsq), y0, self.t)
        self._unpack_simulation(y, Cslsq)

    def _unpack_simulation(self, y, Cslsq):
        self.GS = y[:, 0]
        self.S1 = y[:, 1]
        self.TT_bright = np.sum(y[:, np.where(Cslsq > self.bright_threshold)[0]+2], axis=1)
        self.TT_dark = np.sum(y[:, np.where(Cslsq <= self.bright_threshold)[0]+2], axis=1)
        self.TT_total = np.sum(y[:, 2:11], axis=1)
        self.T_T_total = np.sum(y[:, 11:], axis=1)
        self.num_TT_bright = len(Cslsq[Cslsq > self.bright_threshold])
        self._integrate_populations()
        
    def plot_simulation(self):
        fig = plt.figure(figsize=(6, 5))
        plt.loglog(self.t, self.S1, 'b-', label=r'S$_1$')
        plt.loglog(self.t, self.TT_bright, 'r-', label=r'TT$_{bright}$')
        plt.loglog(self.t, self.TT_dark, 'k-', label=r'TT$_{dark}$')
        plt.loglog(self.t, self.TT_total, 'm-', label=r'TT$_{total}$')
        plt.loglog(self.t, self.T_T_total, 'c-', label=r'T..T$_{total}$')
        plt.xlabel('time (ns)')
        plt.ylabel(r'population (arb.)')
        plt.legend()
        plt.xlim([1e-4, 1e5])
        plt.ylim([1e-8, 1e1])
        return fig
    
    def _integrate_populations(self):
        self.GS_int = np.trapz(self.GS, x=self.t)
        self.S1_int = np.trapz(self.S1, x=self.t)
        self.TT_bright_int = np.trapz(self.TT_bright, x=self.t)
        self.TT_dark_int = np.trapz(self.TT_dark, x=self.t)
        self.TT_total_int = np.trapz(self.TT_total, x=self.t)
        self.T_T_total_int = np.trapz(self.T_T_total, x=self.t)
        
    def get_population_between(self, species, t1, t2):
        mask = ((self.t >= t1) & (self.t <= t2))
        t = self.t[mask]
        if species == 'GS':
            population = np.trapz(self.GS[mask], x=t)/(t2-t1)
        elif species == 'S1':
            population = np.trapz(self.S1[mask], x=t)/(t2-t1)
        elif species == 'TT_bright':
            population = np.trapz(self.TT_bright[mask], x=t)/(t2-t1)
        elif species == 'TT_dark':
            population = np.trapz(self.TT_dark[mask], x=t)/(t2-t1)
        elif species == 'TT_total':
            population = np.trapz(self.TT_total[mask], x=t)/(t2-t1)
        elif species == 'T_T_total':
            population = np.trapz(self.T_T_total[mask], x=t)/(t2-t1)
        else:
            raise ValueError('\'species\' must be one of \'GS\', \'S1\', \'TT_bright\', \'TT_dark\', \'TT_total\', \'T_T_total\'')
        return population
    
    def normalise_population_at(self, species, t):
        idx = np.where((self.t-t)**2 == min((self.t-t)**2))[0][0]
        factor = species[idx]
        species = species/factor
        return species, factor
    
    
class MerrifieldBardeen(object):
    '''
    Solves the Bardeen model for a given set of cslsq values but includes
    free triplets as well, in the same way as Merrifield.
    '''
    def __init__(self):
        self._define_default_parameters()
    
    def _define_default_parameters(self):
        # generation of photoexcited singlet to model IRF
        self.kGEN = 0.25
        # rates between excited states
        self.kSF = 20.0
        self.k_SF = 0.03
        self.kHOP = 0.067
        self.k_HOP = 2.5e-4
        self.kDISS = 0.067
        self.kTTA = 1e-18
        # spin relaxation
        self.kRELAX = 0.033
        # rates of decay
        self.kSNR = 0.1
        self.kSSA =0
        self.kTTNR = 0.067
        self.kSPIN = 2.5e-4
        self.kTNR = 1e-5
        # initial population (arb.)
        self.GS_0 = 8e17
        self.initial_species = 'singlet'
        # bright threshold: if |Cs|^2 > threshold TT state is considered bright
        self.bright_threshold = 1e-4
        # time axis
        self.t = np.logspace(-5, 5, 10000)
        
    def _set_generation_rates(self):
        if self.initial_species == 'singlet':
            self.kGENS = self.kGEN
            self.kGENT = 0
        elif self.initial_species == 'triplet':
            self.kGENS = 0
            self.kGENT = self.kGEN
        else:
            raise ValueError('initial_species attribute must be either \'singlet\' or \'triplet\'')

    def _rate_equations(self, t, y, Cslsq):
        '''
        Note that the TT non-radiative rate is multiplied by Cslsq also here....
        '''
        GS, S1, TT_1, TT_2, TT_3, TT_4, TT_5, TT_6, TT_7, TT_8, TT_9, T_T_1, T_T_2, T_T_3, T_T_4, T_T_5, T_T_6, T_T_7, T_T_8, T_T_9, T1 = y
        dydt = np.zeros(len(y))
        # GS
        dydt[0] = -(self.kGENS+self.kGENT)*GS
        # S1
        dydt[1] = self.kGENS*GS - (self.kSNR+self.kSF*np.sum(Cslsq))*S1 -self.kSSA*S1*S1 + self.k_SF*(Cslsq[0]*TT_1+Cslsq[1]*TT_2+Cslsq[2]*TT_3+Cslsq[3]*TT_4+Cslsq[4]*TT_5+Cslsq[5]*TT_6+Cslsq[6]*TT_7+Cslsq[7]*TT_8+Cslsq[8]*TT_9)
        # TT_1
        dydt[2] = self.kSF*Cslsq[0]*S1 - (self.k_SF+self.kTTNR)*Cslsq[0]*TT_1 - self.kHOP*TT_1 + self.k_HOP*T_T_1
        # TT_2
        dydt[3] = self.kSF*Cslsq[1]*S1 - (self.k_SF+self.kTTNR)*Cslsq[1]*TT_2 - self.kHOP*TT_2 + self.k_HOP*T_T_2
        # TT_3
        dydt[4] = self.kSF*Cslsq[2]*S1 - (self.k_SF+self.kTTNR)*Cslsq[2]*TT_3 - self.kHOP*TT_3 + self.k_HOP*T_T_3
        # TT_4
        dydt[5] = self.kSF*Cslsq[3]*S1 - (self.k_SF+self.kTTNR)*Cslsq[3]*TT_4 - self.kHOP*TT_4 + self.k_HOP*T_T_4
        # TT_5
        dydt[6] = self.kSF*Cslsq[4]*S1 - (self.k_SF+self.kTTNR)*Cslsq[4]*TT_5 - self.kHOP*TT_5 + self.k_HOP*T_T_5
        # TT_6
        dydt[7] = self.kSF*Cslsq[5]*S1 - (self.k_SF+self.kTTNR)*Cslsq[5]*TT_6 - self.kHOP*TT_6 + self.k_HOP*T_T_6
        # TT_7
        dydt[8] = self.kSF*Cslsq[6]*S1 - (self.k_SF+self.kTTNR)*Cslsq[6]*TT_7 - self.kHOP*TT_7 + self.k_HOP*T_T_7
        # TT_8
        dydt[9] = self.kSF*Cslsq[7]*S1 - (self.k_SF+self.kTTNR)*Cslsq[7]*TT_8 - self.kHOP*TT_8 + self.k_HOP*T_T_8
        # TT_9
        dydt[10] = self.kSF*Cslsq[8]*S1 - (self.k_SF+self.kTTNR)*Cslsq[8]*TT_9 - self.kHOP*TT_9 + self.k_HOP*T_T_9
        # T_T_1
        dydt[11] = self.kHOP*TT_1 - (self.k_HOP+self.kSPIN+self.kRELAX+self.kDISS)*T_T_1 + (1/8)*self.kRELAX*(T_T_2+T_T_3+T_T_4+T_T_5+T_T_6+T_T_7+T_T_8+T_T_9) + (1/9)*self.kTTA*T1*T1
        # T_T_2
        dydt[12] = self.kHOP*TT_2 - (self.k_HOP+self.kSPIN+self.kRELAX+self.kDISS)*T_T_2 + (1/8)*self.kRELAX*(T_T_1+T_T_3+T_T_4+T_T_5+T_T_6+T_T_7+T_T_8+T_T_9) + (1/9)*self.kTTA*T1*T1
        # T_T_3
        dydt[13] = self.kHOP*TT_3 - (self.k_HOP+self.kSPIN+self.kRELAX+self.kDISS)*T_T_3 + (1/8)*self.kRELAX*(T_T_1+T_T_2+T_T_4+T_T_5+T_T_6+T_T_7+T_T_8+T_T_9) + (1/9)*self.kTTA*T1*T1
        # T_T_4
        dydt[14] = self.kHOP*TT_4 - (self.k_HOP+self.kSPIN+self.kRELAX+self.kDISS)*T_T_4 + (1/8)*self.kRELAX*(T_T_1+T_T_2+T_T_3+T_T_5+T_T_6+T_T_7+T_T_8+T_T_9) + (1/9)*self.kTTA*T1*T1
        # T_T_5
        dydt[15] = self.kHOP*TT_5 - (self.k_HOP+self.kSPIN+self.kRELAX+self.kDISS)*T_T_5 + (1/8)*self.kRELAX*(T_T_1+T_T_2+T_T_3+T_T_4+T_T_6+T_T_7+T_T_8+T_T_9) + (1/9)*self.kTTA*T1*T1
        # T_T_6
        dydt[16] = self.kHOP*TT_6 - (self.k_HOP+self.kSPIN+self.kRELAX+self.kDISS)*T_T_6 + (1/8)*self.kRELAX*(T_T_1+T_T_2+T_T_3+T_T_4+T_T_5+T_T_7+T_T_8+T_T_9) + (1/9)*self.kTTA*T1*T1
        # T_T_7
        dydt[17] = self.kHOP*TT_7 - (self.k_HOP+self.kSPIN+self.kRELAX+self.kDISS)*T_T_7 + (1/8)*self.kRELAX*(T_T_1+T_T_2+T_T_3+T_T_4+T_T_5+T_T_6+T_T_8+T_T_9) + (1/9)*self.kTTA*T1*T1
        # T_T_8
        dydt[18] = self.kHOP*TT_8 - (self.k_HOP+self.kSPIN+self.kRELAX+self.kDISS)*T_T_8 + (1/8)*self.kRELAX*(T_T_1+T_T_2+T_T_3+T_T_4+T_T_5+T_T_6+T_T_7+T_T_9) + (1/9)*self.kTTA*T1*T1
        # T_T_9
        dydt[19] = self.kHOP*TT_9 - (self.k_HOP+self.kSPIN+self.kRELAX+self.kDISS)*T_T_9 + (1/8)*self.kRELAX*(T_T_1+T_T_2+T_T_3+T_T_4+T_T_5+T_T_6+T_T_7+T_T_8) + (1/9)*self.kTTA*T1*T1
        #
        dydt[20] = self.kGENT*GS + 2.0*self.kDISS*(T_T_1+T_T_2+T_T_3+T_T_4+T_T_5+T_T_6+T_T_7+T_T_8+T_T_9) - 2.0*self.kTTA*T1*T1 - self.kTNR*T1
        #
        return dydt
    
    def simulate(self, Cslsq):
        self._set_generation_rates()
        y0 = np.array([self.GS_0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        y = odeint(lambda y, t: self._rate_equations(t, y, Cslsq), y0, self.t)
        self._unpack_simulation(y, Cslsq)

    def _unpack_simulation(self, y, Cslsq):
        self.GS = y[:, 0]
        self.S1 = y[:, 1]
        self.TT_bright = np.sum(y[:, np.where(Cslsq > self.bright_threshold)[0]+2], axis=1)
        self.TT_dark = np.sum(y[:, np.where(Cslsq <= self.bright_threshold)[0]+2], axis=1)
        self.TT_total = np.sum(y[:, 2:11], axis=1)
        self.T_T_total = np.sum(y[:, 11:-1], axis=1)
        self.T1 = y[:, -1]
        self.num_TT_bright = len(Cslsq[Cslsq > self.bright_threshold])
        self._integrate_populations()
        
    def plot_simulation(self):
        fig = plt.figure(figsize=(6, 5))
        plt.loglog(self.t, self.S1, 'b-', label=r'S$_1$')
        plt.loglog(self.t, self.TT_bright, 'r-', label=r'TT$_{bright}$')
        plt.loglog(self.t, self.TT_dark, 'k-', label=r'TT$_{dark}$')
        plt.loglog(self.t, self.TT_total, 'm-', label=r'TT$_{total}$')
        plt.loglog(self.t, self.T_T_total, 'c-', label=r'T..T$_{total}$')
        plt.loglog(self.t, self.T1, 'g-', label=r'T$_1$')
        plt.xlabel('time (ns)')
        plt.ylabel(r'population (cm$^{-3}$)')
        plt.legend()
        plt.xlim([1e-4, 1e5])
        plt.ylim([1e-8, 1e1])
        return fig
    
    def _integrate_populations(self):
        self.GS_int = np.trapz(self.GS, x=self.t)
        self.S1_int = np.trapz(self.S1, x=self.t)
        self.TT_bright_int = np.trapz(self.TT_bright, x=self.t)
        self.TT_dark_int = np.trapz(self.TT_dark, x=self.t)
        self.TT_total_int = np.trapz(self.TT_total, x=self.t)
        self.T_T_total_int = np.trapz(self.T_T_total, x=self.t)
        
    def get_population_between(self, species, t1, t2):
        mask = ((self.t >= t1) & (self.t <= t2))
        t = self.t[mask]
        if species == 'GS':
            population = np.trapz(self.GS[mask], x=t)/(t2-t1)
        elif species == 'S1':
            population = np.trapz(self.S1[mask], x=t)/(t2-t1)
        elif species == 'TT_bright':
            population = np.trapz(self.TT_bright[mask], x=t)/(t2-t1)
        elif species == 'TT_dark':
            population = np.trapz(self.TT_dark[mask], x=t)/(t2-t1)
        elif species == 'TT_total':
            population = np.trapz(self.TT_total[mask], x=t)/(t2-t1)
        elif species == 'T_T_total':
            population = np.trapz(self.T_T_total[mask], x=t)/(t2-t1)
        elif species == 'T1':
            population = np.trapz(self.T1[mask], x=t)/(t2-t1)
        else:
            raise ValueError('\'species\' must be one of \'GS\', \'S1\', \'TT_bright\', \'TT_dark\', \'TT_total\', \'T_T_total\', \'T1\'')
        return population
    
    def normalise_population_at(self, species, t):
        idx = np.where((self.t-t)**2 == min((self.t-t)**2))[0][0]
        factor = species[idx]
        species = species/factor
        return species, factor
