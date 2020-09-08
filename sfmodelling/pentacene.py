import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class RateModel:

    def __init__(self):
        self._number_of_states = 2
        self.states = ['S1', 'T1']
        self.rates = []
        self.model_name = 'base'
        self._time_resolved = True
        self.G = 1e17
        self._allowed_initial_states = {'S1', 'T1'}
        self._initial_state_mapping = {'S1': 0, 'T1': -1}
        self.initial_weighting = {'S1': 1}
        
    def _check_initial_weighting(self):
        for starting_state in self.initial_weighting.keys():
            if starting_state not in self._allowed_initial_states:
                raise ValueError('invalid state {0} in initial_weighting'.format(starting_state))
            if self.initial_weighting[starting_state] < 0:
                raise ValueError('weightings must be positive')
        return
            
    def _set_initial_condition(self):
        self._y0 = np.zeros(self._number_of_states)
        total_weights = np.sum(np.array(list(self.initial_weighting.values())))
        for key in self.initial_weighting.keys():
            idx = self._initial_state_mapping[key]
            weight = self.initial_weighting[key]/total_weights
            self._y0[idx] = weight*self.G
        return


class TimeResolvedModel(RateModel):

    def __init__(self):
        super().__init__()
        self.t_step = 0.0052391092278624
        self.t_end = 1e6
        self.num_points = 10000
        return
    
    def _calculate_time_axis(self):
        self.t = np.geomspace(self.t_step, self.t_end+self.t_step, self.num_points)-self.t_step
        self.t[0] = 0
        return
    
    def view_timepoints(self):
        self._calculate_time_axis()
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.semilogx(self.t, np.ones_like(self.t), 'bx')
        plt.show()
        print('\n')
        for t in self.t[0:5]:
            print(t)
        print('\n')
        for t in self.t[-5:]:
            print(t)
        return
    
    def _rate_equations(self, y, t):
        return np.ones(self._number_of_states+1)
        
    def _initialise_simulation(self):
        self._calculate_time_axis()
        self._check_initial_weighting()
        self._set_initial_condition()
        return
    
    def simulate(self):
        self._initialise_simulation()
        y = odeint(lambda y, t: self._rate_equations(y, t), self._y0, self.t)
        self._unpack_simulation(y)
        return


class PentaceneModel(TimeResolvedModel):

    def __init__(self):
        super().__init__()
        # metadata
        self.model_name = 'Pentacene Model'
        self._number_of_states = 3
        self.states = ['S1', 'TT', 'T1']
        self.rates = ['kSF', 'k_SF', 'kSEP', 'kTTA', 'kSNR', 'kTTNR', 'kTNR']
        # rates between excited states
        self.kSF = 20.0
        self.k_SF = 0.03
        self.kSEP = 0.067
        self.kTTA = 1e-18
        # rates of decay
        self.kSNR = 0.1
        self.kTTNR = 0.067
        self.kTNR = 1e-5
        # fraction of annihilations that form TT
        self.fTTA = 1e-2

    def _rate_equations(self, y, t):
        S1, TT, T1 = y
        dydt = np.zeros(self._number_of_states)
        # S1
        dydt[0] = -(self.kSNR+self.kSF)*S1 + self.k_SF*TT
        # TT
        dydt[1] = self.kSF*S1 - (self.k_SF+self.kSEP+self.kTTNR)*TT + self.fTTA*self.kTTA*T1*T1
        # T1
        dydt[2] = 2*self.kSEP*TT - 2*self.kTTA*T1*T1 - self.kTNR*T1
        #
        return dydt

    def _unpack_simulation(self, y):
        self.S1 = y[:, 0]
        self.TT = y[:, 1]
        self.T1 = y[:, 2]
        self._wrap_simulation_results()
        return
    
    def _wrap_simulation_results(self):
        self.simulation_results = dict(zip(self.states, [self.S1, self.TT, self.T1]))
        return
