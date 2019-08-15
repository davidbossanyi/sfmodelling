'''
Contains the class SpinHamiltonian. This class can be used to calculate the
overlaps of the triplet pair wavefunctions with the singlet triplet pair.
These are then the cslsq inputs for the rate models in models.py.

Version 1.0
David Bossanyi 12/08/2019
dgbossanyi1@sheffield.ac.uk
'''

import numpy as np


class SpinHamiltonian(object):
    '''
    ABOUT
    A class containing methods for calculating and using the total spin
    Hamiltonian for triplet pair states.
    
    The Hamiltonian is the sum of the Zeeman, D, E, and dipole-dipole terms.
    
    The class includes methods to compute the eigenstates and eigenvalues of
    the spin Hamiltonian, as well as the overlap of the triplet pair states
    with the bright singlet - the |Cs|^2 from Merrifield/Bardeen etc.
    
    The |Cs|^2 can be computed for a specific triplet pair orientation with
    respect to the B-field, or for random triplet pair orientation, in which
    case the program samples 10^4 possible orientations and takes the average
    
    Methods do exactly what they say on the tin.
    
    USAGE
    The D, E, and dipole-dipole interaction parameters can be adjusted as
    required using, for example:
    
    >>> SpinHamiltonian.D = 5.1e-6
    >>> SpinHamiltonian.E = 1.6e-7
    >>> SpinHamiltonian.X = 1.1e-9
    
    DETAILS
    uses constants:
        hbar = 1
    uses units:
        B:      tesla
        angles: radians
        energy: electronvolt 
    uses high-field basis states:
        (1,1) (1,0) (1,-1) (0,1) (0,0) (0,-1) (-1,1) (-1,0) (-1,-1)
    '''
    def __init__(self):
        self._set_constants()
        self._initialise_magnetic_parameters()
        self._set_singlet_state()
        
    def _set_constants(self):
        self.mu_B = 5.788e-5  # Bohr magneton in eV/T
        self.g = 2.002        # electron gyromagnetic ratio
        
    def _initialise_magnetic_parameters(self):
        self.D = -5.8e-6  # D parameter in eV from Bayliss PRL 2014 (TIPS-Tc)
        self.E = self.D/3   # E parameter in eV
        self.X = 1e-10    # intertriplet dipole-dipole interaction in eV
    
    def _set_singlet_state(self):
        self.S_state = (1/np.sqrt(3))*np.array([0, 0, -1, 0, 1, 0, -1, 0, 0])
        
    @staticmethod
    def _calculate_projections(theta_A, phi_A, theta_B, phi_B):
        hxA = np.sin(theta_A)*np.cos(phi_A)
        hyA = np.sin(theta_A)*np.sin(phi_A)
        hzA = np.cos(theta_A)
        hxB = np.sin(theta_B)*np.cos(phi_B)
        hyB = np.sin(theta_B)*np.sin(phi_B)
        hzB = np.cos(theta_B)
        return hxA, hyA, hzA, hxB, hyB, hzB
                
    def calculate_zeeman_hamiltonian(self, B, theta_A, phi_A, theta_B, phi_B):
        isqrt2 = 1/np.sqrt(2)
        hxA, hyA, hzA, hxB, hyB, hzB = self._calculate_projections(theta_A, phi_A, theta_B, phi_B)
        hxyAp = isqrt2*(hxA+1j*hyA)
        hxyAm = isqrt2*(hxA-1j*hyA)
        hxyBp = isqrt2*(hxB+1j*hyB)
        hxyBm = isqrt2*(hxB-1j*hyB) 
        matrix = np.array([[hzA+hzB, hxyBm, 0      , hxyAm, 0    , 0    , 0      , 0    , 0         ],
                           [hxyBp  , hzA  , hxyBm  , 0    , hxyAm, 0    , 0      , 0    , 0         ],
                           [0      , hxyBp, hzA-hzB, 0    , 0    , hxyAm, 0      , 0    , 0         ],
                           [hxyAp  , 0    , 0      , hzB  , hxyBm, 0    , hxyAm  , 0    , 0         ],
                           [0      , hxyAp, 0      , hxyBp, 0    , hxyBm, 0      , hxyAm, 0         ],
                           [0      , 0    , hxyAp  , 0    , hxyBp, -hzB , 0      , 0    , hxyAm     ],
                           [0      , 0    , 0      , hxyAp, 0    , 0    , hzB-hzA, hxyBm, 0         ],
                           [0      , 0    , 0      , 0    , hxyAp, 0    , hxyBp  , -hzA , hxyBm     ],
                           [0      , 0    , 0      , 0    , 0    , hxyAp, 0      , hxyBp, -(hzA+hzB)]])
        H_Z = self.g*self.mu_B*B*matrix
        return H_Z
        
    def calculate_D_hamiltonian(self):
        matrix = np.array([[2,  0, 0,  0,  0,  0, 0,  0, 0],
                           [0, -1, 0,  0,  0,  0, 0,  0, 0],
                           [0,  0, 2,  0,  0,  0, 0,  0, 0],
                           [0,  0, 0, -1,  0,  0, 0,  0, 0],
                           [0,  0, 0,  0, -4,  0, 0,  0, 0],
                           [0,  0, 0,  0,  0, -1, 0,  0, 0],
                           [0,  0, 0,  0,  0,  0, 2,  0, 0],
                           [0,  0, 0,  0,  0,  0, 0, -1, 0],
                           [0,  0, 0,  0,  0,  0, 0,  0, 2]])
        H_D = (1/3)*self.D*matrix
        return H_D
        
    def calculate_E_hamiltonian(self):
        matrix = np.array([[0, 0, 1, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1, 0],
                           [1, 0, 0, 0, 0, 0, 0, 0, 1],
                           [0, 0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0, 0, 0],
                           [1, 0, 0, 0, 0, 0, 0, 0, 1],
                           [0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 1, 0, 0]])
        H_E = self.E*matrix
        return H_E
        
    def calculate_dd_hamiltonian(self):
        matrix = np.array([[1, 0,  0, 0, 0, 0,  0, 0, 0],
                           [0, 0,  0, 1, 0, 0,  0, 0, 0],
                           [0, 0, -1, 0, 1, 0,  0, 0, 0],
                           [0, 1,  0, 0, 0, 0,  0, 0, 0],
                           [0, 0,  1, 0, 0, 0,  1, 0, 0],
                           [0, 0,  0, 0, 0, 0,  0, 1, 0],
                           [0, 0,  0, 0, 1, 0, -1, 0, 0],
                           [0, 0,  0, 0, 0, 1,  0, 0, 0],
                           [0, 0,  0, 0, 0, 0,  0, 0, 1]])
        H_dd = self.X*matrix
        return H_dd
        
    def calculate_hamiltonian(self, B, theta_A, phi_A, theta_B, phi_B):
        H_Z = self.calculate_zeeman_hamiltonian(B, theta_A, phi_A, theta_B, phi_B)
        H_D = self.calculate_D_hamiltonian()
        H_E = self.calculate_E_hamiltonian()
        H_dd = self.calculate_dd_hamiltonian()
        H = H_Z + H_D + H_E + H_dd
        return H
        
    def calculate_TTl_states(self, B, theta_A, phi_A, theta_B, phi_B):
        H = self.calculate_hamiltonian(B, theta_A, phi_A, theta_B, phi_B)
        TTl_eigenvalues, TTl_eigenstates = np.linalg.eig(H)
        return TTl_eigenvalues, TTl_eigenstates
        
    def calculate_overlap_specific_orientation(self, B, theta_A, phi_A, theta_B, phi_B):
        TTl_eigenvalues, TTl_eigenstates = self.calculate_TTl_states(B, theta_A, phi_A, theta_B, phi_B)
        overlap = np.matmul(self.S_state, TTl_eigenstates)
        Cslsq = np.abs(overlap)**2
        return Cslsq
        
    def calculate_overlap_random_orientation(self, B):
        theta_range = np.linspace(0, np.pi, 10)
        phi_range = np.linspace(0, 2*np.pi, 10)
        Cslsq = np.zeros(((len(theta_range)**2)*(len(phi_range)**2), 9))
        index = 0
        for theta_A in theta_range:
            for phi_A in phi_range:
                for theta_B in theta_range:
                    for phi_B in phi_range:
                        Cslsq[index, :] = self.calculate_overlap_specific_orientation(B, theta_A, phi_A, theta_B, phi_B)
                        index += 1
        Cslsq = Cslsq.mean(axis=0)
        return Cslsq
