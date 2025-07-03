# Load Modulation Simulator

This code is used in the paper:

Agwan, U., Bobick, S., Rangan, S., Poolla, K., & Spanos, C. (2025). The potential for building emissions reduction through sporadic and targeted load modulation. In Workshop on Computational Optimization of Buildings at the 42nd International Conference on Machine Learning (ICML) (Vol. 267). PMLR.

## Data

This code is designed to interface with marginal emissions intensities (MOERs) from [Wattime](https://watttime.org/) and synthetic load profiles from NREL's [ResStock](https://resstock.nrel.gov/) and [ComStock](https://comstock.nrel.gov/). 

### Wattime MOER Data

The [Wattime API](https://docs.watttime.org/) provides access to real-time, forecasted, and historical Marginal Operating Emissions Rate (MOER) in units of kg CO$_2$ / MWh for electric grids around the world. Users without a subscription can access MOERs for the CAISO_NORTH region, which is analyzed in the numerical study in the paper.

### NREL Load Profiles

NREL's [ResStock](https://resstock.nrel.gov/) and [ComStock](https://comstock.nrel.gov/) datasets model residential and commercial loads, respectively. NREL utilized super-computing to run over 20 million simulations using a statistical model of building stock characteristics for the entire United States in 15-minute intervals, differentiated by end-use. For our study, we queried load profiles from Alameda County, California, USA from a simulation of 2018 using actual meteorological data. Users can view and access the data using NREL's [data viewer](https://resstock.nrel.gov/datasets) its [API](https://github.com/NREL/buildstock-query).

## Using this Repo
To use this code, after querying data from the sources above and adding into the `data` directory, run the entire simulation with
```
python main.py
```
This will populate a .csv in the `out` directory which details the optimal load modulation actions for each day of the year and end-use.

To change the assumptions of the load flexibility of different end uses, edit `data/commercial_assumptions.json` and/or `data/commercial_assumptions.json`. Each end use corresponds to a 4-tuple, which contains the load flexibility parameter $\gamma$ for 15-minute, 1-hour, 2-hour, and 4-hour load modulation lengths $\Delta$, respectively.




