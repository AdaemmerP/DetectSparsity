"""
This script loads the necessary packages, sets up the workers and prepares 
the data as well as pre-allocates for Figure 5
"""

# Activate environment
	using Pkg	
	Pkg.activate(".")

# Load packages
	using DataFrames
	using DataFramesMeta
	using LinearAlgebra
	using Distributions
	using CSV
	using Dates
	using Chain
	using StatsBase
	using Combinatorics
	using GLM
	using ThreadsX
	using Random
	using GLMNet
	using MLDataUtils
	using RCall
	using Distributed
  using Lasso
  using Plots
 
 # Set up workers for parallelization
  BLAS.set_num_threads(1)
	addprocs(ncores)
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
		include("GLP_SpikeSlab.jl")
		include("Functions.jl")
	end

	
#---------------------------------------------------------------------------------------------------#	
#------------------------------------- Load data sets ----------------------------------------------#
#----------------------------------------------------------------------------------------------------#

if dataset == 0
    # Types for GW and Pyun data 
      typesPyun = Any[Float64 for  i = 1:16] 

    # Load GW and Pyun datac
      Xall =	CSV.read(string(data_path, "GWP_Data.csv"), 
                      types = typesPyun, DataFrame) 	

    # Drop missing observation					 
      Xall = @chain Xall begin
                  dropmissing(_)	
                  @transform!(:date = string.(:date))
                  @transform!(:date = map(x -> Date(Dates.Year(x[1:4]), Dates.Month(x[5:6])), :date))			
            end		

 elseif dataset == 1

      coltypes    = Any[Float64 for i= 1:117] 
      coltypes[1] = String
      Xall        = CSV.read(string(data_path, "MacroData_INDPRO_prepared.csv"),
                              types = coltypes, DataFrame) 
      Xall.date   = Date.(Xall.date, "yyyy-mm-dd")        
    
    # Compute correlation Matrix untile 12-1969 
      corx       = cor(Matrix(Xall[1:findfirst(i -> i == Date("1969-12-01"), Xall.date), Not(:date)]))
      corx       = UpperTriangular(corx)
      removecols = unique(getindex.(findall(i -> (abs(i) >= 0.95 && i !==1.0), corx), 2)) .+ 1
      Xall       = Xall[:, Not(removecols)]

    # Remove 'Covid sample'    
      Xall     = filter(row -> row.date <= Date("2019-12-01"), Xall)

 end

#-----------------------------------------------------------------------------------------------------#
					
#-----------------------------------------------------------------------------------------------------#	
#------------------------------------- Estimate moments ----------------------------------------------#
#-----------------------------------------------------------------------------------------------------#

# Goyal Welch and Pyun data	
	x_mat  = Xall[1:(end - 1), 4:end] |> Matrix{Float64} 
	x_mat  = standardize(ZScoreTransform, x_mat, dims = 1) 
	y_GWP  = Xall[1:(end - 1), 3] 
	
# Set parameters	  
	μx    = repeat([0.0], size(x_mat, 2))
  βx    = GLM.coef(lm(hcat(repeat([1], size(x_mat, 1)), x_mat), y_GWP))[2:end] 
	ϕx    = map(1:size(x_mat, 2)) do ii 
            x_t  = x_mat[2:end, ii] 
            x_t1 = hcat(repeat([1], size(x_mat, 1) - 1), Float64.(lag(x_mat[:, ii])[2:end]))
            GLM.coef(lm(x_t1, x_t))[2]         
          end
# Covariance matrix of residuals for predictors
  Σ_x = cov((x_mat - ϕx'.*lag(x_mat))[2:end, :]) 

# Estimate degrees of freedom 
  @rput y_GWP      
  R"""
  library(MASS)
  results = fitdistr(y_GWP, "t", start = list(m = mean(y_GWP), s = sd(y_GWP), df = 3), 
                     lower = c(-1, 0.0001, 3))
  nu_est  = results[[1]][3]
  """
  ν = @rget nu_est

# Simulation parameters
  q0  = Int64(140)  
  τ0  = Int64(60)
  n   = q0 + τ0 + 1                  		 
  if dataset == 0
    ω  	= [10.0; 8.0; 5.0] 
  elseif dataset == 1
    ω  	= [2.0; 1.5; 1.0] 
  end
  α        = [0.5;  1.0]		        
  ζ        = [0.0, 0.25, 0.5, 0.75, 1.0]          

  if dataset == 0 
	  nz_β     = [3; 6; 13]  
  elseif dataset == 1 
    nz_β     = [5; 50; 100]
  end

# Set parameters for GLP code
  abeta = bbeta = Abeta = Bbeta = 1.0

# Model and iterator combinations for forecast combinations
  nr_preds          = 1:size(x_mat, 2)
  moos_total        = τ0 + 1           
  sample_train      = @views map(x -> 1:(q0 + x), 0:(moos_total - 1)) 
  models_univ_time  = collect(Iterators.product(Tuple(x for x in sample_train), nr_preds)) # Iterators for univariate forecasts

  if dataset == 0
	  model_comb   = collect(combinations(1:(last(nr_preds) + 1))) # ('+1' because intercept only will be added)
  elseif dataset == 1
    model_comb   = reduce(vcat, map(ii -> collect(combinations(1:(size(x_mat, 2) + 1), ii)), (1:3)))
    model_comb   = vcat(model_comb, [collect(2:(size(x_mat, 2) + 1))])
  end
  
# Iterators for forecast combinations  
  comb_t = (collect(Iterators.product(1:moos_total, model_comb)))
	
# Pre-allocate for forecast combinations
  fccomb_flex_nr              = DataFrame(zeros(N, length(ω)), :auto);
  rename!(fccomb_flex_nr, ["nb_5"; "nb_50"; "nb_100"])
  fccomb_flex_nr[!, :Method] .= "FC-Flex"

# Pre-allocate for glmnets  
  enet_nr               = DataFrame(zeros(N, length(ω)), :auto);
  rename!(enet_nr, ["nb_5"; "nb_50"; "nb_100"])
  enet_nr[!, :Method]  .= "Elastic Net" 

  lasso_nr              = DataFrame(zeros(N, length(ω)), :auto);
  rename!(lasso_nr, ["nb_5"; "nb_50"; "nb_100"])
  lasso_nr[!, :Method] .= "Lasso"

  lasso_relax_nr    = DataFrame(zeros(N, length(ω)), :auto);
  rename!(lasso_relax_nr, ["nb_5"; "nb_50"; "nb_100"])
  lasso_relax_nr[!, :Method] .= "Relaxed Lasso"
# lasso_adapt_nr   = zeros(length(nz_β), length(ω));

# Pre-allocate for Bayesian FSS
  glp_nr  = DataFrame(zeros(N, length(ω)), :auto);
  rename!(glp_nr, ["nb_5"; "nb_50"; "nb_100"])
  glp_nr[!, :Method] .= "BFSS"

# Pre-allocate for BKM
  bss_nr   =  DataFrame(zeros(N, length(ω)), :auto);
  rename!(bss_nr, ["nb_5"; "nb_50"; "nb_100"])
  bss_nr[!, :Method] .= "BSS"

    

