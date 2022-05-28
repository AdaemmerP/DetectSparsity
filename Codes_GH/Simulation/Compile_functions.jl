
"""
This script runs all main functions once for compilation
but with a fixed number of N (here N = ncores)
"""

# Info message
  @info string("Compiling functions...")

# Generate data	 
  β_actpre    = map(i -> sample(1:length(βx), 3; replace = false), 1:ncores) # Draw active predictors	  
         
	xysim_data_prec  = map(i -> data_simul(n, βx, Σ_x, μx, ω[1], β_actpre[1], ϕx, 
																		 err_type, ν, i), 1:ncores)

# Forecast combinations
  ThreadsX.map(i -> fcomb_simul(xysim_data_prec[i], sample_train, model_comb, models_univ_time, comb_t,
                       β_actpre[i]), 1:ncores)  

# Glmnet
  ThreadsX.map(i -> glmnet_simul_cvts(xysim_data_prec[i][:, 1], 
                                      xysim_data_prec[i][:, 2:end], 
                                      q0, τ0, 1.0,
                                      β_actpre[i]), 1:ncores);

# Relaxed Lasso               
  pmap(ii -> glmnet_relaxed_cvts(xysim_data_prec[ii][:, 1], 
                                 xysim_data_prec[ii][:, 2:end], 
                                 q0, τ0, 1.0, ζ,
                                 β_actpre[ii]), 1:ncores)               

# GLP
	pmap(ii -> GLP_oos(xysim_data_prec[ii][1:(end - 1), 2:end], xysim_data_prec[ii][end, 2:end], 
                     xysim_data_prec[ii][1:(end - 1), 1], xysim_data_prec[ii][end, 1],
									   2, 1, abeta, bbeta, Abeta, Bbeta, 
									   Int64(floor(abs(sum(xysim_data_prec[ii])))),
                     β_actpre[ii]), 
									   1:ncores)

# BKM
  pmap(ii -> bkm_predict(xysim_data_prec[ii], collect(1:(n - 1)), n, β_actpre[ii]), 1:ncores) 
  
# Adaptive Lasso 
  pmap(ii -> 
    adaptive_lasso_cvts(xysim_data_prec[ii][:, 1], 
                        xysim_data_prec[ii][:, 2:end], 
                        q0, τ0,
                        β_actpre[ii]), 1:ncores)

		
# Info message
  @info string("Compilation completed")					
