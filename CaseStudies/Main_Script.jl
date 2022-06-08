# Activate environment
  @__DIR__
  using Pkg	
  Pkg.activate(".")

# Load packages
  using DataFrames
  using DataFramesMeta
  using Chain
  using LinearAlgebra
  using Distributions
  using StatsBase
  using CSV
  using ThreadsX
  using Dates
  using BenchmarkTools
  using GLM
  using Combinatorics
  using Lasso
  using GLMNet
  using HypothesisTests
  using Random
  using MLDataUtils
  using RCall
  using Distributed

# Set data path 
  data_path = ""

# Set path to save results (only necessary when 'save_results = true')
  save_path    = "" 
  save_results = false  # Save results?

# Set number of cores for OpenBlas
  BLAS.set_num_threads(1)

# Set up workers for parallelization (run line 38 only once!)
if nprocs() == 1
  addprocs(10)
  @everywhere begin
    using Pkg; Pkg.activate(".")  
    using Random
    using StatsBase
    using LinearAlgebra
    using Distributions
    using GLMNet
    using RCall
    using Lasso
    BLAS.set_num_threads(1)
    include("Functions.jl")	
    include("GLP_SpikeSlab.jl")
  end
end

# Load script with functions  
  include("GLP_SpikeSlab.jl")
  include("Functions.jl")
  
# Choose dataset
  dataset = 1 # 1: y = US excess stock returns,         x = Goyal Welch predictors  
              # 2: y = US excess stock returns,         x = Goyal Welch and Pyun predictor
              # 3: y = US industrial production growth, x = macroeconomic variables
              # 4: y = US industrial production growth, x = macroeconomic variables (including 4 lags of INDPRO)
              # 5: y = US industrial production growth, x = Only 4 lags of INDPRO

  covid_out = false # End in '2019-12-01'? Only for macroeconomic data (datasets 3 - 5)
              
# Load data
  include("LoadandPrepare.jl")

# Date vectors with start dates
# First tuple entry: dataset. 
# Second tuple entries: starting days
  sdates_val = [(0, "2015-09-01"),
                (1, "1954-11-01", "1993-11-01"),
                (2, "1993-11-01"),
                (3, "1964-11-01"),
                (4, "1964-11-01"),
                (5, "1964-11-01")]

  sdates_noval = [(0, "2020-09-01"),
                  (1, "1959-11-01", "1998-11-01"),
                  (2, "1998-11-01"),
                  (3, "1969-11-01"),
                  (4, "1969-11-01"),
                  (5, "1969-11-01")]

 # Number of observations for time series cross validation            
   τ0 = 60            

# Pre-compile all functions for later speedup (run only once)
  include("Compile_functions.jl")

#-------------------------------------------------------------------------------#


#-------------------------------------------------------------------------------#	
# ------------------ Forecast combinations (time series CV) ------------------- #
#-------------------------------------------------------------------------------#	

# Shrinkage intensity for univariate forecasts
  const δ = 0.5
	
# Run script
  @time include("01_FComb_Fcasts.jl")

#-------------------------------------------------------------------------------#	
# -----------    BSS using Bess package (Wen et al., 2020)    ----------------- #
#-------------------------------------------------------------------------------#	

# Run script
  @time include("02_BSS_Fcasts.jl")

#-------------------------------------------------------------------------------#	
#---------------------------   GLP forecasts    --------------------------------#
#-------------------------------------------------------------------------------#

# Set parameters for Bayesian model (N = burnin) 
  N = Int64(1e3)	       
  M = Int64(10e3) + N     
  abeta	= bbeta = Abeta = Bbeta = 1.0

# Run script 
  @time include("03_BFSS_Fcasts.jl")

#-------------------------------------------------------------------------------#	
#--------------------   Enet, Lasso, Ridge    (Time Series CV)  ----------------#
#-------------------------------------------------------------------------------#	

# Vector with alphas (Enet, Lasso, Ridge)
  αvec  = [0.5; 1.0; 0.0]

# Run script
  @time include("04_Glmnet_Fcasts_tscv.jl")

#-------------------------------------------------------------------------------#	
#--------------------            Relaxed Lasso                  ----------------#
#-------------------------------------------------------------------------------#	
# Only put one value for α, the code is not vectorized
  α = 1.0                         # Lasso
  ζ = [0.0, 0.25, 0.5, 0.75, 1.0] # Levels of relaxation

  @time include("05_Glmnet_relaxed.jl")
	  
 
# Close all workers
  rmprocs(workers())


#-------------------------------------------------------------------------------#
#  The results of the following methods are not shown in the paper 
#-------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------#	
#--------------------            Adaptive Lasso                  ----------------#
#-------------------------------------------------------------------------------#	
#  @time include("06_Adaptive_Lasso.jl")

# #-------------------------------------------------------------------------------#
# # ----------  BSS (with time series CV) Only use for the financial data  --- #
# #-------------------------------------------------------------------------------#
# Only run this code for the financial datasets! (dataset = 1 or 2)
 #  @time include("07_BSS_tscv.jl")


# #-------------------------------------------------------------------------------#
# # ------------      Enet, Lasso, Ridge (Cross sectional CV)         ------------#	
# #-------------------------------------------------------------------------------#

# Vector with alphas (Ridge, Enet, Lasso)
#   αvec  = [0.5 1.0 0]

# # Run script
#   @time include("08_Glmnet_Fcasts_cscv.jl")





