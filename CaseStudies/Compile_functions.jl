"""
This script runs all main functions once for compilation
"""

  @info string("Compiling functions...")

# Only few observations
  dataset   = 0 
 
# Load and prepare data
  include("LoadandPrepare.jl")


#-------------------------------------------------------------------------------#	
# ------------------ Forecast combinations (time series CV) ------------------- #
#-------------------------------------------------------------------------------#	

# Shrinkage intensity for univariate forecasts
  const δ = 0.5

# Run script
  include("01_FComb_Fcasts.jl")


#-------------------------------------------------------------------------------#	
# -----------    BSS using Bess package (Wen et al., 2020)    ----------------- #
#-------------------------------------------------------------------------------#	

# Run script
  include("02_BSS_Fcasts.jl")

#-------------------------------------------------------------------------------#	
#---------------------------   GLP forecasts    --------------------------------#
#-------------------------------------------------------------------------------#

# Set parameters for Bayesian model (N = burnin) 
  N = 1     
  M = 2 + N     
  abeta	= bbeta = Abeta = Bbeta = 1.0

# Run script 
  include("03_BFSS_Fcasts.jl")


#-------------------------------------------------------------------------------#	
#--------------------   Enet, Lasso, Ridge    (Time Series CV)  ----------------#
#-------------------------------------------------------------------------------#	

# Vector with alphas (Enet, Lasso, Ridge)
  αvec  = [0.5; 1.0; 0.0]

# Run script
  include("04_Glmnet_Fcasts_tscv.jl")


#-------------------------------------------------------------------------------#	
#--------------------            Relaxed Lasso                  ----------------#
#-------------------------------------------------------------------------------#	
# Only put one value for α, the code is not vectorized
  α = 1.0                         # Lasso
  ζ = [0.0, 0.25, 0.5, 0.75, 1.0] # Levels of relaxation

  include("05_Glmnet_relaxed.jl")  

  @info string("Compilation completed")


