# Master_Thesis
Master Thesis code and results. The PV data used is not publicly available, therefore these results are not directly reproducible.

## Input data preparation
Contains the scripts used to transform PV power data as well as KNMI meteorological data into Neural Network input. Specifically files containing data from 2015 to 2017 with 1 minute and 15 minute temporal resolutions, for both temporal resolutions with and without DWT as features implemented. Some simple input data visualizations are included.

## Results
Contains the results of the MSc Thesis research. Scripts and results are seperated by temporal horizon, Model archetype and DWT implementation. The gridsearch results can be found in the .csv files.

### Ramp score classification
The ramp score classification notebook file contains an unfinished example of how the ramp error could be used to classify rather than forecast PV system power output 32 minutes into the future. Although unfinished and likely flawed, the results are promosing!
