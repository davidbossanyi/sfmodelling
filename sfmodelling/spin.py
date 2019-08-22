'''
Contains the class SpinHamiltonian. This class can be used to calculate the
overlaps of the triplet pair wavefunctions with the singlet triplet pair.
These are then the cslsq inputs for the rate models in models.py.

Version 2.0
David Bossanyi 12/08/2019
dgbossanyi1@sheffield.ac.uk
'''

import numpy as np


class SpinHamiltonian(object):
    '''
    ABOUT
    A class containing methods for calculating and using the total spin
    Hamiltonian for triplet pair states.
    
    The Hamiltonian is the sum of the Zeeman, Zero-Field and dipole-dipole 
    terms.
    
    The class includes methods to compute the eigenstates and eigenvalues of
    the spin Hamiltonian, as well as the overlap of the triplet pair states
    with the bright singlet - the |Cs|^2 from Merrifield/Bardeen etc.
    
    The |Cs|^2 can be computed for a specific triplet pair orientation with
    respect to the B-field, or for random triplet pair orientation, in which
    case the program samples 10^4 possible orientations and takes the average.
    Note that for simulating fluorescence, any averaging should be done AFTER
    computing the fluorescence for EACH possible set of |Cs| values.
    
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
    uses zero-field basis states:
        (x,x) (x,y) (x,z) (y,x) (y,y) (y,z) (z,x) (z,y) (z,z)
    '''
    def __init__(self):
        self._set_constants()
        self._initialise_magnetic_parameters()
        self._set_singlet_state()
        
    def _set_constants(self):
        self.mu_B = 5.788e-5  # Bohr magneton in eV/T
        self.g = 2.002        # electron gyromagnetic ratio
        
    def _initialise_magnetic_parameters(self):
        self.D = 6.45e-5  # D parameter in eV from Bayliss PRL 2014 (TIPS-Tc)
        self.E = -6.45e-6   # E parameter in eV
        self.X = self.D/100   # intertriplet dipole-dipole interaction in eV
        self.rAB = (0, 0, 1)  # unit vector from COM of A to COM of B in A coordinates
    
    def _set_singlet_state(self):
        self.S_state = (1/np.sqrt(3))*np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])
        
        
    @staticmethod    
    def _rotation_matrix(alpha, beta, gamma):
        '''
        computes the 3x3 rotation matrix using the Euler angles that would
        rotate molecule A onto molecule B according to the zx'z'' convention
        '''
        R = np.array([[ np.cos(alpha)*np.cos(gamma)-np.sin(alpha)*np.cos(beta)*np.sin(gamma),  np.sin(alpha)*np.cos(gamma)+np.cos(alpha)*np.cos(beta)*np.sin(gamma), np.sin(beta)*np.sin(gamma)],
                      [-np.cos(alpha)*np.sin(gamma)-np.sin(alpha)*np.cos(beta)*np.cos(gamma), -np.sin(alpha)*np.sin(gamma)+np.cos(alpha)*np.cos(beta)*np.cos(gamma), np.sin(beta)*np.cos(gamma)],
                      [ np.sin(alpha)*np.sin(beta)                                          , -np.cos(alpha)*np.sin(beta)                                          , np.cos(beta)              ]])
        return R
    
    def calculate_zerofield_hamiltonian_molecule_A(self):
        D3 = self.D/3
        E = self.E
        H_ZF_A = np.array([[D3-E, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, D3-E, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, D3-E, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, D3+E, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, D3+E, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, D3+E, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, -2*D3, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, -2*D3, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, -2*D3]])
        return H_ZF_A
    
    def calculate_zerofield_hamiltonian_single_molecule(self):
        D3 = self.D/3
        E = self.E
        H_ZF_SM = np.array([[D3-E, 0, 0],
                            [0, D3+E, 0],
                            [0, 0, -2*D3]])
        return H_ZF_SM
    
    def calculate_zerofield_hamiltonian_molecule_B(self, alpha, beta, gamma):
        R = self._rotation_matrix(alpha, beta, gamma)
        H_ZF_SM = self.calculate_zerofield_hamiltonian_single_molecule()
        H_ZF_SM_B = np.matmul(np.transpose(R), np.matmul(H_ZF_SM, R))
        H_ZF_B = np.zeros((9, 9))
        H_ZF_B[0:3, 0:3] = H_ZF_SM_B
        H_ZF_B[3:6, 3:6] = H_ZF_SM_B
        H_ZF_B[6:9, 6:9] = H_ZF_SM_B
        return H_ZF_B
    
    def calculate_zeeman_hamiltonian(self, B, theta, phi):
        Hx, Hy, Hz = self._calculate_projections(B, theta, phi)
        H_Z = np.array([[0, -Hz, Hy, -Hz, 0, 0, Hy, 0, 0],
                        [Hz, 0, -Hx, 0, -Hz, 0, 0, Hy, 0],
                        [-Hy, Hx, 0, 0, 0, -Hz, 0, 0, Hy],
                        [Hz, 0, 0, 0, -Hz, Hy, -Hx, 0, 0],
                        [0, Hz, 0, Hz, 0, -Hx, 0, -Hx, 0],
                        [0, 0, Hz, -Hy, Hx, 0, 0, 0, -Hx],
                        [-Hy, 0, 0, Hx, 0, 0, 0, -Hz, Hy],
                        [0, -Hy, 0, 0, Hx, 0, Hz, 0, -Hx],
                        [0, 0, -Hy, 0, 0, Hx, -Hy, Hx, 0]])
        return 1j*self.g*self.mu_B*H_Z
        
    def calculate_dipoledipole_hamiltonian(self):
        u, v, w = self.rAB
        H_dd = np.array([[0, 0, 0, 0, 1-3*w*w, 3*v*w, 0, 3*v*w, 1-3*v*v],
                         [0, 0, 0, -1+3*w*w, 0, -3*u*w, -3*v*w, 0, 3*u*v],
                         [0, 0, 0, -3*v*w, 3*u*w, 0, -1+3*v*v, -3*u*v, 0],
                         [0, -1+3*w*w, -3*v*w, 0, 0, 0, 0, -3*u*w, 3*u*v],
                         [1-3*w*w, 0, 3*u*w, 0, 0, 0, 3*u*w, 0, 1-3*u*u],
                         [3*v*w, -3*u*w, 0, 0, 0, 0, -3*u*v, -1+3*u*u, 0],
                         [0, -3*v*w, -1+3*v*v, 0, 3*u*w, -3*u*v, 0, 0, 0],
                         [3*v*w, 0, -3*u*v, -3*u*w, 0, -1+3*u*u, 0, 0, 0],
                         [1-3*v*v, 3*u*v, 0, 3*u*v, 1-3*u*u, 0, 0, 0, 0]])
        return -self.X*H_dd
        
    @staticmethod
    def _calculate_projections(B, theta, phi):
        Hx = B*np.sin(theta)*np.cos(phi)
        Hy = B*np.sin(theta)*np.sin(phi)
        Hz = B*np.cos(theta)
        return Hx, Hy, Hz
        
    def calculate_hamiltonian(self, B, theta, phi, alpha, beta, gamma):
        H_Z = self.calculate_zeeman_hamiltonian(B, theta, phi)
        H_ZF_A = self.calculate_zerofield_hamiltonian_molecule_A()
        H_ZF_B = self.calculate_zerofield_hamiltonian_molecule_B(alpha, beta, gamma)
        H_dd = self.calculate_dipoledipole_hamiltonian()
        H = H_Z + H_ZF_A + H_ZF_B + H_dd
        return H
        
    def calculate_TTl_states(self, B, theta, phi, alpha, beta, gamma):
        H = self.calculate_hamiltonian(B, theta, phi, alpha, beta, gamma)
        TTl_eigenvalues, TTl_eigenstates = np.linalg.eigh(H)
        return TTl_eigenvalues, TTl_eigenstates
        
    def calculate_overlap_specific_orientation(self, B, theta, phi, alpha, beta, gamma):
        TTl_eigenvalues, TTl_eigenstates = self.calculate_TTl_states(B, theta, phi, alpha, beta, gamma)
        overlap = np.matmul(self.S_state, TTl_eigenstates)
        Cslsq = np.abs(overlap)**2
        return Cslsq
    
    def calculate_overlap_semirandom_orientation(self, B, alpha, beta, gamma):
        theta_range = np.linspace(0, np.pi, 10)
        phi_range = np.linspace(0, 2*np.pi, 10)
        Cslsq = np.zeros((len(theta_range)*len(phi_range), 9))
        index = 0
        for theta in theta_range:
            for phi in phi_range:
                Cslsq[index, :] = self.calculate_overlap_specific_orientation(B, theta, phi, alpha, beta, gamma)
                index += 1
        Cslsq = Cslsq.mean(axis=0)
        return Cslsq
    
    def calculate_overlap_random_orientation(self, B, density=10):
        theta_range = np.linspace(0, np.pi, density)
        phi_range = np.linspace(0, 2*np.pi, density)
        Cslsq = np.zeros((len(theta_range)**2*len(phi_range)**3, 9))
        index = 0
        for theta in theta_range:
            for phi in phi_range:
                for alpha in phi_range:
                    for beta in theta_range:
                        for gamma in phi_range:
                            Cslsq[index, :] = self.calculate_overlap_specific_orientation(B, theta, phi, alpha, beta, gamma)
                            index += 1
        Cslsq = Cslsq.mean(axis=0)
        return Cslsq