# Complex-System-Simulation-Project

## Using cellular automata to model vegetation patterns in drylands
Team 15
Students: Emma van der Spel, Justus de Bruijn Kops, Lila Wagenaar

### Project overview 
Many arid and semi-arid regions exhibit large-scale vegetation patterns spanning multiple spatial scales. These patters suggest that ecosystems arise through self-organization rather than fine-tuned environmental conditions. 

In this project, we implement and study a two-dimensional cellular automaton (CA) model based on Scanlon et al. (2007). The model uses probabilistic update rules combining local vegetation feedback and a global rainfall-dependent target, and seems to reproduces power-law cluster size distribution. 

Our goal is to reproduce the results of Scanlon et al. (2007), as well as exploring the parameter space (rainfall, neighborhood radius, local/global weighting), study sensitivity and robustness of power-law scaling, and analyze percolation behavior.

### Research questions
- What ingredients in the CA model give rise to power-law clustering? 
- How do local and global contributions affect large-scale structure? 
- How sensitive is the power-law behavior to parameter changes?

### Hypotheses (maybe to update a little bit?)
- A combination of short and long-range interaction rules is necessary to generate structured vegetation patterns.
- There are possibly several types of rules/methods in CA-based models that can give rise to certain behavior, i.e. some states (power-law clustering etc.) can be achieved in several different models.
- To be able to reproduce the variety and statistics of real-life patterns, there is some probabilistic component necessary in the CA model. 

### Model description 
In our model, each cell represents vegetation presence/absence. At every time step, cells update probabilitically based on: 
- A distance-weighted local vegetation density
- A global rainfall-dependent target fraction

The model is implemented in ``` src/CA_model.py```  with analysis tools in ``` src/analysis.py```  and data management utilities in ``` src/utils.py``` 


### Analysis & Experiments
The project includes: 
- Cluster size distribution analysis, including truncated power-law fitting
- Percolation probability analysis, measuring connectivity across the grid
- Parameter sweeps over rainall(``` true_frac``` ), neighborhood size ("``` k, M``` ), and local/global weight (``` phi``` )
- Visualization of equilibrium states for different parameter settings 
These analysis are primarily conducted in Jupyter notebookds in the ``` notebooks/```  directory, using saved simulation data to avoir recomputation.

### Testing & Validation
To ensure correctness and robustness of the implementation, we included multiple assert statements and internal checks, in the core CA logic. 

### References 
[1] Scanlon TM, Caylor KK, Levin SA, Rodriguez-Iturbe I. Positive feedbacks promote power-law clustering of Kalahari vegetation. Nature. 2007.
[2] Pascual M, Guichard F. Criticality and disturbance in spatial ecological systems. Trends Ecol. Evol. 2005.


### Notes 
The simulations were computationally expensive; saved ```.npy``` files allow analysis without rerunning model. 
For some models, faster simulations are also available, with different parameters, but less precision.
