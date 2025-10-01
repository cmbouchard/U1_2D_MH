30 Sep 2025
Chris Bouchard

Generates pure U(1) gauge field configurations via Markov chain Monte Carlo in two Euclidean spacetime dimensions.
Uses Metropolis-Hastings for updates.

Files included and their role:
* generate.py: user interface, set parameters here; run this to generate configurations
* lattice_collection.py: some functions defined here
* gauge_latticeqcd.py: most of work is done here, where Metropolis-Hastings updates are carried out
* action_v_cfg.py: calcualates action of existing configurations
* plaq_v_cfg.py: calculates the average plaquette of existing configurations
* topoQ_v_cfg.py: calcualates topological charge of existing configurations
* autocorr.py: calculates autocorrelation function for specified observable (action, plaquette, or topological charge)
