# 2024ESN-TNN
Supplemental Codes for "High Resolution Urban Air Quality Monitoring from Citizen Science Data with Echo-State Transformer Networks" by Matthew Bonas and Stefano Castruccio

## Data
Folder containing simulated data `L96SimData.RData` with 40 variables (locations) and 1000 time points as well as the PurpleAir sensor data `APFour.RData`. These data are to be used in conjunction with the associated R and Python scripts to produce forecasts with the DESN and TNN models referenced in the manuscript. 


<p align = "center">
  <img src="https://github.com/user-attachments/assets/ec8b4ec5-7728-46f9-a3e1-04a8edcf3583" alt="F1-SFPurpleAirDemo" width="600"/>
  <br>
</p>

## Code
R and Python scripts to produce forecasts on the simulated data for the DESN and TNN models. User should run the R script to produce forecasts for the DESN and to generate the data to to be used as input for the TNN. We also provide an R script to generated the calibrated uncertainty from the forecasts of the ESN-TNN method. This script is written for use with the PurpleAir data but can easily be modified for the simulated data example. Finally, we provide codes to interpolate the forecasts from the ESN-TNN method to the census tracts in the city in order to calculate exposure estimates.
