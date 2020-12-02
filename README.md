# sfmodelling

This repository contains the code necessary to reproduce the kinetic modelling of diftes and pentacene PL dynamics in the article **Emissive spin-0 triplet-pairs are a direct product of triplet-triplet annihilation in pentacene single crystals and anthradithiophene films** by Bossanyi et al. You can read the paper [here](https://www.nature.com/articles/s41557-020-00593-y).

### Requirements
- python
- matplotlib
- numpy
- scipy
- pandas
- usefulfunctions (from [here](https://github.com/davidbossanyi/useful-functions))
- tripletpairs (from [here](https://github.com/davidbossanyi/triplet-pair-states))

### Notes
To reproduce the simulations found in the paper, simply download the repository, unzip, and run the two scripts included in the main directory.

I have since developed this code into a more complete package for kinetic modelling of singlet fission systems which can be found [here](https://github.com/davidbossanyi/triplet-pair-states). Note that the simulations for pentacene use some of the functionality of this package.
