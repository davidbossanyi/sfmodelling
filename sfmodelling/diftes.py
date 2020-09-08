"""
Contains the clases for several different rate models. The classes can be used 
to simulate the dynamics of the various excited states as a function of
excitation density and magnetic field.

Models included in models.py:
    - Merrifield
    - Bardeen
    - ExtendedBardeen
    - AnnihilatingTripletPairs
    - MerrifieldExplicit1TT
    - Simple3StateModel
"""

import numpy as np
from scipy.integrate import odeint


class MerrifieldExplicit1TT(object):
    """
    This is basically Merrifields model, but explicitly separating 
    the 1(TT) from S1 and (T..T).
    """

    def __init__(self):
        self._define_default_parameters()
    
    def _define_default_parameters(self):
        """
        Set some arbitrary values to begin with.
        """
        # generation of photoexcited singlet to model IRF
        self.kGEN = 0.25
        # rates between excited states
        self.kSF = 20.0
        self.k_SF = 0.03
        self.kHOP = 0.067
        self.k_HOP = 2.5e-4
        self.kHOP2 = 1e-5
        self.kTTA = 1e-18
        # spin relaxation
        self.kRELAX = 0
        # rates of decay
        self.kSNR = 0.1
        self.kSSA = 0
        self.kTTNR = 0.067
        self.kTNR = 1e-5
        # initial population
        self.GS_0 = 8e17
        self.initial_species = 'singlet'
        # time axis
        self.t = np.logspace(-5, 5, 10000)
        # TTA channel
        self.TTA_channel = 1
        
    def _set_tta_rates(self):
        """
        Different options for the recombining triplets.
        """
        if self.TTA_channel == 1:  # this is T1 + T1 -> (T..T)
            self._kTTA_1 = self.kTTA
            self._kTTA_2 = 0
            self._kTTA_3 = 0
        elif self.TTA_channel == 2:  # this is T1 + T1 -> (TT)
            self._kTTA_1 = 0
            self._kTTA_2 = self.kTTA
            self._kTTA_3 = 0
        elif self.TTA_channel == 3:  # this is T1 + T1 -> S1
            self._kTTA_1 = 0
            self._kTTA_2 = 0
            self._kTTA_3 = self.kTTA
        else:
            raise ValueError('TTA channel must be either 1, 2 or 3') 
        
    def _set_generation_rates(self):
        """
        Can either start with population in S1 or T1.
        """
        if self.initial_species == 'singlet':  # start by populating the S1 state (standard experiments)
            self._kGENS = self.kGEN
            self._kGENT = 0
        elif self.initial_species == 'triplet':  # start by populating the T1 state (upconversion experiments)
            self._kGENS = 0
            self._kGENT = self.kGEN
        else:
            raise ValueError('initial_species attribute must be either \'singlet\' or \'triplet\'')           

    def _rate_equations(self, t, y, cslsq):
        """
        This function sets the rate equations in the format required by 
        scipy.integrate.odeint
        
        It takes the Cs overlap values as an argument
        """
        GS, S1, TT, T_T_1, T_T_2, T_T_3, T_T_4, T_T_5, T_T_6, T_T_7, T_T_8, T_T_9, T1 = y
        dydt = np.zeros(len(y))
        # GS
        dydt[0] = -(self._kGENS+self._kGENT)*GS
        # S1
        dydt[1] = self._kGENS*GS - (self.kSNR+self.kSF)*S1 - self.kSSA*S1*S1 + self.k_SF*TT + self._kTTA_3*T1**2
        # TT
        dydt[2] = self.kSF*S1 - (self.k_SF+self.kTTNR+self.kHOP*np.sum(cslsq))*TT + self.k_HOP*(cslsq[0]*T_T_1+cslsq[1]*T_T_2+cslsq[2]*T_T_3+cslsq[3]*T_T_4+cslsq[4]*T_T_5+cslsq[5]*T_T_6+cslsq[6]*T_T_7+cslsq[7]*T_T_8+cslsq[8]*T_T_9) + self._kTTA_2*T1**2
        # T_T_1
        dydt[3] = self.kHOP*cslsq[0]*TT - (self.k_HOP*cslsq[0]+self.kTNR+self.kHOP2+self.kRELAX)*T_T_1 + (1/9)*self._kTTA_1*T1**2 + (1/8)*self.kRELAX*(T_T_2+T_T_3+T_T_4+T_T_5+T_T_6+T_T_7+T_T_8+T_T_9)
        # T_T_2
        dydt[4] = self.kHOP*cslsq[1]*TT - (self.k_HOP*cslsq[1]+self.kTNR+self.kHOP2+self.kRELAX)*T_T_2 + (1/9)*self._kTTA_1*T1**2 + (1/8)*self.kRELAX*(T_T_1+T_T_3+T_T_4+T_T_5+T_T_6+T_T_7+T_T_8+T_T_9)
        # T_T_3
        dydt[5] = self.kHOP*cslsq[2]*TT - (self.k_HOP*cslsq[2]+self.kTNR+self.kHOP2+self.kRELAX)*T_T_3 + (1/9)*self._kTTA_1*T1**2 + (1/8)*self.kRELAX*(T_T_1+T_T_2+T_T_4+T_T_5+T_T_6+T_T_7+T_T_8+T_T_9)
        # T_T_4
        dydt[6] = self.kHOP*cslsq[3]*TT - (self.k_HOP*cslsq[3]+self.kTNR+self.kHOP2+self.kRELAX)*T_T_4 + (1/9)*self._kTTA_1*T1**2 + (1/8)*self.kRELAX*(T_T_1+T_T_2+T_T_3+T_T_5+T_T_6+T_T_7+T_T_8+T_T_9)
        # T_T_5
        dydt[7] = self.kHOP*cslsq[4]*TT - (self.k_HOP*cslsq[4]+self.kTNR+self.kHOP2+self.kRELAX)*T_T_5 + (1/9)*self._kTTA_1*T1**2 + (1/8)*self.kRELAX*(T_T_1+T_T_2+T_T_3+T_T_4+T_T_6+T_T_7+T_T_8+T_T_9)
        # T_T_6
        dydt[8] = self.kHOP*cslsq[5]*TT - (self.k_HOP*cslsq[5]+self.kTNR+self.kHOP2+self.kRELAX)*T_T_6 + (1/9)*self._kTTA_1*T1**2 + (1/8)*self.kRELAX*(T_T_1+T_T_2+T_T_3+T_T_4+T_T_5+T_T_7+T_T_8+T_T_9)
        # T_T_7
        dydt[9] = self.kHOP*cslsq[6]*TT - (self.k_HOP*cslsq[6]+self.kTNR+self.kHOP2+self.kRELAX)*T_T_7 + (1/9)*self._kTTA_1*T1**2 + (1/8)*self.kRELAX*(T_T_1+T_T_2+T_T_3+T_T_4+T_T_5+T_T_6+T_T_8+T_T_9)
        # T_T_8
        dydt[10] = self.kHOP*cslsq[7]*TT - (self.k_HOP*cslsq[7]+self.kTNR+self.kHOP2+self.kRELAX)*T_T_8 + (1/9)*self._kTTA_1*T1**2 + (1/8)*self.kRELAX*(T_T_1+T_T_2+T_T_3+T_T_4+T_T_5+T_T_6+T_T_7+T_T_9)
        # T_T_9
        dydt[11] = self.kHOP*cslsq[8]*TT - (self.k_HOP*cslsq[8]+self.kTNR+self.kHOP2+self.kRELAX)*T_T_9 + (1/9)*self._kTTA_1*T1**2 + (1/8)*self.kRELAX*(T_T_1+T_T_2+T_T_3+T_T_4+T_T_5+T_T_6+T_T_7+T_T_8)
        #
        dydt[12] = self._kGENT*GS + (self.kTNR+(2.0*self.kHOP2))*(T_T_1+T_T_2+T_T_3+T_T_4+T_T_5+T_T_6+T_T_7+T_T_8+T_T_9) - 2*self._kTTA_1*T1**2 - 2*self._kTTA_2*T1**2 - 2*self._kTTA_3*T1**2 - self.kTNR*T1
        #
        return dydt
    
    def simulate(self, cslsq):
        """
        Given an array of 9 cslsq calues (output from SpinHamiltonian class) 
        this simulates all population dynamics
        """
        self._set_tta_rates()
        self._set_generation_rates()
        y0 = np.array([self.GS_0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # the number of zeros needs to match the number of species!
        y = odeint(lambda y, t: self._rate_equations(t, y, cslsq), y0, self.t)
        self._unpack_simulation(y, cslsq)

    def _unpack_simulation(self, y, cslsq):
        """
        This is called by model.simulate - it populates species attributes 
        with numpy arrays which are the population dynamics
        """
        self.GS = y[:, 0]
        self.S1 = y[:, 1]
        self.TT = y[:, 2]
        self.T_T_total = np.sum(y[:, 3:12], axis=1)
        self.T1 = y[:, -1]
        
    def get_population_between(self, species, t1, t2):
        """
        integrates the dynamics of species (e.g. model.S1) 
        between times t1 and t2
        """
        mask = ((self.t >= t1) & (self.t <= t2))
        t = self.t[mask]
        population = np.trapz(species[mask], x=t)/(t2-t1)
        return population
    
    def normalise_population_at(self, species, t):
        """
        normalises the dynamics of species (e.g. model.S1) at time t
        """
        idx = np.where((self.t-t)**2 == min((self.t-t)**2))[0][0]
        factor = species[idx]
        species = species/factor
        return species, factor


class Merrifield(object):
    """
    The standard Merrifield model.
    """
    
    def __init__(self):
        self._define_default_parameters()
    
    def _define_default_parameters(self):
        """
        Set some arbitrary values to begin with.
        """
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
        # time axis
        self.t = np.logspace(-5, 5, 10000)
        
    def _set_generation_rates(self):
        """
        Can either start with population in S1 or T1.
        """
        if self.initial_species == 'singlet':
            self.kGENS = self.kGEN
            self.kGENT = 0
        elif self.initial_species == 'triplet':
            self.kGENS = 0
            self.kGENT = self.kGEN
        else:
            raise ValueError('initial_species attribute must be either \'singlet\' or \'triplet\'')

    def _rate_equations(self, t, y, cslsq):
        """
        This function sets the rate equations in the format required by 
        scipy.integrate.odeint
        
        It takes the Cs overlap values as an argument
        """
        GS, S1, TT_1, TT_2, TT_3, TT_4, TT_5, TT_6, TT_7, TT_8, TT_9, T1 = y
        dydt = np.zeros(len(y))
        # GS
        dydt[0] = -(self.kGENS+self.kGENT)*GS
        # S1
        dydt[1] = self.kGENS*GS - (self.kSNR+self.kSF*np.sum(cslsq))*S1 -self.kSSA*S1*S1+ self.k_SF*(cslsq[0]*TT_1+cslsq[1]*TT_2+cslsq[2]*TT_3+cslsq[3]*TT_4+cslsq[4]*TT_5+cslsq[5]*TT_6+cslsq[6]*TT_7+cslsq[7]*TT_8+cslsq[8]*TT_9)
        # TT_1
        dydt[2] = self.kSF*cslsq[0]*S1 - (self.k_SF*cslsq[0]+self.kDISS+self.kTTNR+self.kRELAX)*TT_1 + (1/9)*self.kTTA*T1*T1 + (1/8)*self.kRELAX*(TT_2+TT_3+TT_4+TT_5+TT_6+TT_7+TT_8+TT_9)
        # TT_2
        dydt[3] = self.kSF*cslsq[1]*S1 - (self.k_SF*cslsq[1]+self.kDISS+self.kTTNR+self.kRELAX)*TT_2 + (1/9)*self.kTTA*T1*T1 + (1/8)*self.kRELAX*(TT_1+TT_3+TT_4+TT_5+TT_6+TT_7+TT_8+TT_9)
        # TT_3
        dydt[4] = self.kSF*cslsq[2]*S1 - (self.k_SF*cslsq[2]+self.kDISS+self.kTTNR+self.kRELAX)*TT_3 + (1/9)*self.kTTA*T1*T1 + (1/8)*self.kRELAX*(TT_1+TT_2+TT_4+TT_5+TT_6+TT_7+TT_8+TT_9)
        # TT_4
        dydt[5] = self.kSF*cslsq[3]*S1 - (self.k_SF*cslsq[3]+self.kDISS+self.kTTNR+self.kRELAX)*TT_4 + (1/9)*self.kTTA*T1*T1 + (1/8)*self.kRELAX*(TT_1+TT_2+TT_3+TT_5+TT_6+TT_7+TT_8+TT_9)
        # TT_5
        dydt[6] = self.kSF*cslsq[4]*S1 - (self.k_SF*cslsq[4]+self.kDISS+self.kTTNR+self.kRELAX)*TT_5 + (1/9)*self.kTTA*T1*T1 + (1/8)*self.kRELAX*(TT_1+TT_2+TT_3+TT_4+TT_6+TT_7+TT_8+TT_9)
        # TT_6
        dydt[7] = self.kSF*cslsq[5]*S1 - (self.k_SF*cslsq[5]+self.kDISS+self.kTTNR+self.kRELAX)*TT_6 + (1/9)*self.kTTA*T1*T1 + (1/8)*self.kRELAX*(TT_1+TT_2+TT_3+TT_4+TT_5+TT_7+TT_8+TT_9)
        # TT_7
        dydt[8] = self.kSF*cslsq[6]*S1 - (self.k_SF*cslsq[6]+self.kDISS+self.kTTNR+self.kRELAX)*TT_7 + (1/9)*self.kTTA*T1*T1 + (1/8)*self.kRELAX*(TT_1+TT_2+TT_3+TT_4+TT_5+TT_6+TT_8+TT_9)
        # TT_8
        dydt[9] = self.kSF*cslsq[7]*S1 - (self.k_SF*cslsq[7]+self.kDISS+self.kTTNR+self.kRELAX)*TT_8 + (1/9)*self.kTTA*T1*T1 + (1/8)*self.kRELAX*(TT_1+TT_2+TT_3+TT_4+TT_5+TT_6+TT_7+TT_9)
        # TT_9
        dydt[10] = self.kSF*cslsq[8]*S1 - (self.k_SF*cslsq[8]+self.kDISS+self.kTTNR+self.kRELAX)*TT_9 + (1/9)*self.kTTA*T1*T1 + (1/8)*self.kRELAX*(TT_1+TT_2+TT_3+TT_4+TT_5+TT_6+TT_7+TT_8)
        # T1
        dydt[11] = self.kGENT*GS + 2.0*self.kDISS*(TT_1+TT_2+TT_3+TT_4+TT_5+TT_6+TT_7+TT_8+TT_9) - 2.0*self.kTTA*T1*T1 - self.kTNR*T1
        #
        return dydt
    
    def simulate(self, cslsq):
        """
        Given an array of 9 cslsq calues (output from SpinHamiltonian class) 
        this simulates all population dynamics
        """
        self._set_generation_rates()
        y0 = np.array([self.GS_0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        y = odeint(lambda y, t: self._rate_equations(t, y, cslsq), y0, self.t)
        self._unpack_simulation(y, cslsq)

    def _unpack_simulation(self, y, cslsq):
        """
        This is called by model.simulate - it populates species attributes 
        with numpy arrays which are the population dynamics
        """
        self.GS = y[:, 0]
        self.S1 = y[:, 1]
        self.TT_bright = cslsq[0]*y[:, 2] + cslsq[1]*y[:, 3] + cslsq[2]*y[:, 4] + cslsq[3]*y[:, 5] + cslsq[4]*y[:, 6] + cslsq[5]*y[:, 7] + cslsq[6]*y[:, 8] + cslsq[7]*y[:, 9] + cslsq[8]*y[:, 10]
        self.TT_total = np.sum(y[:, 2:11], axis=1)
        self.T_T_total = np.sum(y[:, 11:-1], axis=1)
        self.T1 = y[:, -1]
        
    def get_population_between(self, species, t1, t2):
        """
        integrates the dynamics of species (e.g. model.S1) 
        between times t1 and t2
        """
        mask = ((self.t >= t1) & (self.t <= t2))
        t = self.t[mask]
        population = np.trapz(species[mask], x=t)/(t2-t1)
        return population
    
    def normalise_population_at(self, species, t):
        """
        normalises the dynamics of species (e.g. model.S1) at time t
        """
        idx = np.where((self.t-t)**2 == min((self.t-t)**2))[0][0]
        factor = species[idx]
        species = species/factor
        return species, factor
