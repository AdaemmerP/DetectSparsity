"""
# This is the main script for the histogram simulations
# """

# Set path to data
  include("data_path.jl")

# Simulation parameter
  ncores   = 12          # Number of cores (for workers) 	
  N    	   = Int64(1e2)  # Number of Monte Carlo iterations 
  dataset  = 1           # 0 = Financial data, 1 = Macroeconomic data (no lags)
  err_type = 0           # 0 = normal errors,  1 = t-distributed errors 
  diag_cov = 0           # Use diagonal covariance matrix?
  q0       = Int64(140)  # Training length 
  τ0       = Int64(60)   # Length for cross validation

# Set parameters for GLP code (N_glp = burnin sample)
   N_glp 	= Int64(1e3)	
   M_glp	= Int64(10e2) + N_glp

# Run script to load packages and prepare data (run only once (!))
  include("PrepareData_2.jl")

# Include scripts with functions
  include("GLP_SpikeSlab.jl")
  include("Functions.jl")

# Run functions once for compilation
  include("Compile_functions.jl")

#-----------------------------------------------------------------------------------------------------#	
#---------------------------------------- Begin Simulation -------------------------------------------#
#-----------------------------------------------------------------------------------------------------#

for ii = 1:length(nz_β)

#------------------------------------- Simulate data -----------------------------------------------#
  # Simulate data
    Random.seed!(6) # 6: Equals seed combination for finance; 4: Equals seed combination for macro
    β_active    = map(kk -> sample(1:length(βx), nz_β[ii]; replace = false), 1:N) # Draw active predictors	  
    xysim_data  = map(kk -> data_simul(n, βx , Σ_x, μx, ω[ii], β_active[kk], ϕx, 
                                       err_type, ν, (kk + ii)), 1:N)                          																 

  # Merge simulated data and active predictors                                          
    xydata_βact     = [xysim_data β_active]       
                                          
#------------------------------------- Forecast combinations ---------------------------------------#																							
   results_fc_all  = pmap(eachrow(xydata_βact)) do kk 

                          fcomb_simul(kk[1], 
                                      sample_train, 
                                      model_comb, 
                                      models_univ_time, 
                                      comb_t, kk[2])

                      end
                               
  # Convert output to Matrix	 
    fccomb_flex_nr[:, ii]  = @views getindex.(results_fc_all, 3)

#------------------------------------- Glmnet  ----------------------------------------------#	
# Make all combinations of α and N
  αN_comb = collect(Iterators.product(1:N, α));	

# Results for weak predictors
  results_glmnet = pmap(αN_comb) do kk

                  glmnet_simul_cvts(xysim_data[kk[1]][:, 1], 
                                    xysim_data[kk[1]][:, 2:end], 
                                    q0, 
                                    τ0, 
                                    kk[2],
                                    β_active[kk[1]])                                  
                                                              
              end        

# Extratc number of included predictors 
  enet_nr[:, ii]   = @views getindex.(results_glmnet[:, 1], 3)
  lasso_nr[:, ii]	 = @views getindex.(results_glmnet[:, 2], 3)
 
#------------------------------------- Relax Lasso  ----------------------------------------------#	 

# Results for relaxed Lasso                                                      
  results_relax = pmap(eachrow(xydata_βact)) do kk

                glmnet_relaxed_cvts(kk[1][:, 1], 
                                    kk[1][:, 2:end], 
                                    q0, τ0, 1.0, ζ,
                                    kk[2])

                  end
 
  lasso_relax_nr[:, ii]  = @views getindex.(results_relax, 3)


#--------------------------------------------- GLP -----------------------------------------------#
# GLP results
  results_glp_all  = pmap(eachrow(xydata_βact)) do kk 

                      GLP_oos(kk[1][1:(end - 1), 2:end], kk[1][end, 2:end], 
                              kk[1][1:(end - 1), 1], kk[1][end, 1],
                              M_glp, N_glp, abeta, bbeta, Abeta, Bbeta, 
                              Int64(floor(abs(sum(kk[1])))),
                              kk[2])

                      end
# Save results
  glp_nr[:, ii]     = @views round.(getindex.(results_glp_all, 3), digits = 0)

#-------------------------------------------- BSS  ---------------------------------------------#
  # Train and test observations
    train_seq = @views collect(1:(size(xysim_data[1], 1) - 1))
    obs_test  = @views size(xysim_data[1], 1)[1]

  # Compute models
    bkm_results_all  = pmap(kk -> bkm_predict(kk[1], train_seq, obs_test, kk[2]), eachrow(xydata_βact)) 

  # Save results for weak predictors: (i) AIC, (ii) BIC and (iii) EBIC	   
    bss_nr[:, ii]   = getindex.(bkm_results_all, 2)

  # Show progress
    @info string("ω = ", ω[ii], "; ", "nz_β = ", nz_β[ii])

end

results_all = [fccomb_flex_nr; 
               bss_nr; 
               lasso_relax_nr;
               lasso_nr; 
               enet_nr; 
               glp_nr; ]


@rput results_all

# Save DataFrame as data.frame for ggplot
R"""
  save(results_all, file = "Sim_Hist_Financial_n200.RData")
"""

# Close workers
  rmprocs(workers())

 #  