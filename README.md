# Rift Inversion Strain Analysis Project

This repository is designed for development of quantitative strain analysis of rift inversion models in ASPECT. Please conduct all code development within this repository on a separate branch and use pull requests when code review is needed. 

Currently, it contains 5 files from Dylan for Peter:
1. `calculate_strain.py` This has code used for an initial attempt at strain calculation across a model
2. `initial_production_sides.py` This has code for defining a rift axis using particles and then tracking it during model inversion
3. `vtk_plot.py` This has support functions used in the other 2 scripts. In the course of this project, some of these should probably be modified and adapted for GDMATE.
4. `suture.py` This has the code that actually assigns the rift axis to particles (which is then used by `initial_production_sides.py`)
5. `particles.py` This contains code used by `suture.py` and likely replicates code from `vtk_plot.py`

These scripts will not work as written, in particular because they rely on a local copy of a set of 16 model files that are quite large (multiple TB). Dylan will set up a copy of this repository on the Tufts cluster in addition to one of these models so that there is data to work with. It may also be worth developing a small model that can be included with this repository (i.e., adding particles to the continental extension cookbook) to facilitate development.