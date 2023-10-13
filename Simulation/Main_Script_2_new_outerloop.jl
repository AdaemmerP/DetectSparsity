"""
# This is the main script for the histogram simulations
# """

# Set path to data
  include("data_path.jl")

# Simulation parameter
  ncores   = 10          # Number of cores (for workers) 	
  N    	   = 100 # Int64(1e3)  # Number of Monte Carlo iterations 
  dataset  = 1           # 0 = Financial data, 1 = Macroeconomic data (no lags)
  err_type = 1           # 0 = normal errors,  1 = t-distributed errors 
  diag_cov = 0           # Use diagonal covariance matrix?
  q0       = Int64(140)  # Training length 
  τ0       = Int64(60)   # Length for cross validation

# Set parameters for GLP code (N_glp = burnin sample)
   N_glp 	= Int64(1e1)	
   M_glp	= Int64(10e1) + N_glp

# Run script to load packages and prepare data (run only once (!))
  include("PrepareData_2.jl")

# Include scripts with functions
  include("GLP_SpikeSlab.jl")
  include("Functions.jl")

# Run functions once for compilation
  include("Compile_functions.jl")
  include("test.jl")

#-----------------------------------------------------------------------------------------------------#	
#---------------------------------------- Begin Simulation -------------------------------------------#
#-----------------------------------------------------------------------------------------------------#

# Runf function once to precompile
N = 20
ii = 1
N_glp_pre = Int64(1e1)	
M_glp_pre	= Int64(1e1) + N_glp
include("PrepareData_2.jl")

include("test.jl")
@everywhere include("test.jl")
@everywhere begin
    βx = $βx
    ω  = $ω 
    Σ_x = $Σ_x
    ϕx  = $ϕx
    μx = $μx
    ν  = $ν
    α  = $α
    nz_β = $nz_β
    n = $n
    ii = $ii
    enet_nr = $enet_nr
    lasso_nr = $lasso_nr
    lasso_relax_nr = $lasso_relax_nr
    glp_nr = $glp_nr
    bss_nr = $bss_nr
    err_type = $err_type
end

using SharedArrays  
using ProgressMeter

enet_nr = SharedArray{Float64}(Matrix(enet_nr[:, 1:end-1])) 
lasso_nr = SharedArray{Float64}(Matrix(lasso_nr[:, 1:end-1])) 
lasso_relax_nr = SharedArray{Float64}(Matrix(lasso_relax_nr[:, 1:end-1])) 
glp_nr = SharedArray{Float64}(Matrix(glp_nr[:, 1:end-1])) 
bss_nr = SharedArray{Float64}(Matrix(bss_nr[:, 1:end-1])) 

# Make chunks for tasks
N_chunks = Iterators.partition(1:N, div(N, nworkers())) |> collect

@showprogress pmap(N_chunks) do N_chunk
  simul_hist!(βx, 
  ω, 
  Σ_x, 
  ϕx, 
  μx, 
  nz_β, 
  M_glp_pre, 
  N_glp_pre, 
  abeta, 
  bbeta, 
  Abeta, 
  Bbeta,
  n, 
  ii, 
  N_chunk,
  enet_nr,
  lasso_nr,
  lasso_relax_nr,
  glp_nr,
  bss_nr)
end

# Update variables
N = 100 
N_glp = Int64(1e3)	
M_glp	= Int64(1e4) + N_glp  
include("PrepareData_2.jl")  
@everywhere include("test.jl") 
# Make chunks for tasks
N_chunks = Iterators.partition(1:N, div(N, nworkers())) |> collect
enet_nr = SharedArray{Float64}(Matrix(enet_nr[:, 1:end-1])) 
lasso_nr = SharedArray{Float64}(Matrix(lasso_nr[:, 1:end-1])) 
lasso_relax_nr = SharedArray{Float64}(Matrix(lasso_relax_nr[:, 1:end-1])) 
glp_nr = SharedArray{Float64}(Matrix(glp_nr[:, 1:end-1])) 
bss_nr = SharedArray{Float64}(Matrix(bss_nr[:, 1:end-1])) 

for ii = 1:length(nz_β)

@showprogress pmap(N_chunks) do N_chunk
  simul_hist!(βx, 
      ω, 
      Σ_x, 
      ϕx, 
      μx, 
      nz_β, 
      M_glp, 
      N_glp, 
      abeta, 
      bbeta, 
      Abeta, 
      Bbeta,
      n, 
      ii, 
      N_chunk,
      enet_nr,
      lasso_nr,
      lasso_relax_nr,
      glp_nr,
      bss_nr)
end
 end

results_all = [#fccomb_flex_nr; 
               bss_nr; 
               lasso_relax_nr;
               lasso_nr; 
               enet_nr; 
               glp_nr; ]


@rput results_all

# Save DataFrame as data.frame for ggplot
R"""
  save(results_all, file = "Sim_Hist_Macro_n400.RData")
"""

# Close workers
  rmprocs(workers())
