# DetectSparsity
This repository contains the replication codes for 
"Detecting the Sparsity Levels of Economic Time Series: On the Impact of Noise" by
[Adämmer and Schüssler (2022)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4019646). The code is mainly written in Julia but we also use R for some routines and visualizations.  We have also rewritten the Matlab code by [Giannone et al. (2021) ](https://www.econometricsociety.org/publications/econometrica/2021/09/01/economic-predictions-big-data-illusion-sparsity) in Julia.

# Codes
The code is - especially for the macroeconomic data - computationally expensive and it heavily relies on parallelization. 
The code is thread safe and fully reproducible. Lowering the iterations for Bayesian FSS vastly improves computation time. 
The simulations have been conducted on a large cluster and lowering the number of iterations will also improve the computation time. <br />

The Julia packages will be automatically installed when instantiating the environment. The procedure is explained in the README files. 
For R you have to install the following packages:  <br />
R packages |
:--------|
`BeSS`   |
`xtable` | 
`ggplot2` | 
`dplyr` |
`tidyr` |
`stringr`|
`scales` | 

## Case Studies 
'Readme_CaseStudies.txt' explains how to initialize the environment in Julia and how to use the codes to replicate the results of the paper. 

## Simulations
'Readme_Simulation.txt' explains how to initialize the environment in Julia and how to use the codes to replicate the results of the paper. 

## Figures
The folder "Figures" contains all empirical results and R codes to replicate the Figures of the paper.  
'Readme_Figures.txt' explains how to use the R codes.

# Data
All information and data sources can be found in 'Data_Sources.txt'. The data sets are contained in the folder "Data".  <br />
