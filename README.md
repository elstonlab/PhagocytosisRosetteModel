# Phagocytosis Rosette Model

Accompanies Herron JC, Hu S, Liu B, Watanabe T, Hahn KM, & Elston TC. <i>Spatial Models of Pattern Formation During Phagocytosis</i>.

__Includes code to:__
- Simulate the WPGAP model from Jacobs 2019 (https://doi.org/10.1371/journal.pone.0213188) in a Cartesian coordinate system, and observe changes in spot size due to diffusion. Also, perform parameter sweeps on individual parameter to examine the effect on patterning.
	- See 'WPGAP_Original_Cartesian_Spot_Size.ipynb', 'WPGAP_Carteisan_Parameter_Sweeps.ipynb', & 'WPGAP_Cartesian_Grid_TwoParamSweep.ipynb'.
- Simulate the WPGAP model in a polar coordinate system, and observe the effect of a ring impacting individual parameters on rosette patterning.
	- See 'WPGAP_FP_TwoStepRing.ipynb'.
- Simulate a simple reaction-diffusion equation based on a diffusing species being activated locally over a disk. Also, demonstrate how a logistic function can reasonably fit these distributions.
	- See 'Simple_Disk_Activation_RxnDiff.ipynb'.
- Perform parameterization (evolutionary algorithm followed by a Markov chain Monte Carlo) to determine how modulating parameters based on the logistic function (see above) can generate rosette patterns. Perform post-parameterization parmeter sweeps as a rough sensitivity analysis.
	- See 'WPGAP_FP_EA.py', 'WPGAP_FP_Check_EA_Results.ipynb', 'WPGAP_FP_MCMC.py', 'WPGAP_FP_Check_MCMC_Results.ipynb', 'WPGAP_FP_PostMCMC_ParamSweep.py', & 'WPGAP_Cartesian_Grid_TwoParamSweep.ipynb'.
- Simulate specific models to a steady state.
	- See 'WPGAP_FP_SimulateToSS.ipynb'.
- Demonstrate how this model could be coupled when discretely modeling individual modulators as the simple reaction-diffusion equation described above (rather than using the logistic function approximation).
	- See 'WPGAP_Coupled_Model.ipynb'.
- Perform additional simulations on various antibody disk and hole sizes.
	- See 'WPGAP_FP_DiskSimulations.py' & 'WPGAP_FP_HoleSimulations.py'.
- Count the number of podosomes experimentally to compare to simulated results. Note, this requires additional code available from: https://github.com/elstonlab/PodosomeImageAnalysis.
	- See 'Podosome_Counting.ipynb'.


__Installation:__
Note: This code was created using Python v 3.7.11 and Jupyter Notebook 6.1.4
1. Requires Python (https://www.python.org/downloads/) and Jupyter Notebook (https://jupyter.org/install).
2. Requires installation of the 'unusual' Python package Dedalus (https://pypi.org/project/dedalus/, v2.2006, __required__ for numerically solving reaction-diffusion equations).
3. Install time should be insignificant, especially if Python and Jupyter are pre-installed. 

__Additional Notes:__
- All analyses conducted on a Macintosh computer (2.3 GHz Quad-Core Intel Core i7) and should only require a maximum of a few minutes per step. 





