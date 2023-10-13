
function simul_hist!(βx, 
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
  
  #------------------------------------- Simulate data -----------------------------------------------#
    # Simulate data
    #Random.seed!(6) # 6: Equals seed combination for finance; 4: Equals seed combination for macro
    β_active    = map(kk -> sample(1:length(βx), nz_β[ii]; replace = false), N_chunk) # Draw active predictors	  
    xysim_data  = map(kk -> data_simul(n, βx , Σ_x, μx, ω[ii], β_active[kk], ϕx, 
                                       err_type, ν, (kk + ii)), 1:length(β_active))                          																 
  
  # Merge simulated data and active predictors                                          
    xydata_βact     = [xysim_data β_active]   
           
  #------------------------------------- Glmnet  ----------------------------------------------#	
  # Make all combinations of α and N
  αN_comb = collect(Iterators.product(1:length(N_chunk), α));	
  
  # Results for weak predictors
  results_glmnet = map(αN_comb) do kk
  
             @views glmnet_simul_cvts(xysim_data[kk[1]][:, 1], 
                                    xysim_data[kk[1]][:, 2:end], 
                                    q0, 
                                    τ0, 
                                    kk[2],
                                    β_active[kk[1]])                                  
                                                              
              end        
  
  # Extratc number of included predictors 
  enet_nr[N_chunk, ii]   = @views getindex.(results_glmnet[:, 1], 3)
  lasso_nr[N_chunk, ii]	 = @views getindex.(results_glmnet[:, 2], 3)
  
  # Show progress
  #@info string("enet und lasso done.")
  
  #------------------------------------- Relax Lasso  ----------------------------------------------#	 
  
  # Results for relaxed Lasso                                                      
  results_relax = map(eachrow(xydata_βact)) do kk
  
          @views glmnet_relaxed_cvts(kk[1][:, 1], 
                                    kk[1][:, 2:end], 
                                    q0, τ0, 1.0, ζ,
                                    kk[2])
  
                  end
  
  lasso_relax_nr[N_chunk, ii]  = @views getindex.(results_relax, 3)
  
  # Show progress
   #@info string("relaxed lasso done.")
    
  #--------------------------------------------- GLP -----------------------------------------------#
  # GLP results
  results_glp_all  = map(eachrow(xydata_βact)) do kk 
  
                @views GLP_oos(kk[1][1:(end - 1), 2:end], kk[1][end, 2:end], 
                              kk[1][1:(end - 1), 1], kk[1][end, 1],
                              M_glp, N_glp, abeta, bbeta, Abeta, Bbeta, 
                              Int64(floor(abs(sum(kk[1])))),
                              kk[2])
  
                      end
  # Save results
  glp_nr[N_chunk, ii]     = @views round.(getindex.(results_glp_all, 3), digits = 0)
  
  # Show progress
  #@info string("GLP done.")
  
  #-------------------------------------------- BSS  ---------------------------------------------#
  # Train and test observations
    train_seq = @views collect(1:(size(xysim_data[1], 1) - 1))
    obs_test  = @views size(xysim_data[1], 1)[1]
  
  # Compute models
    bkm_results_all  = map(kk -> bkm_predict(kk[1], train_seq, obs_test, kk[2]), eachrow(xydata_βact)) 
  
  # Save results for weak predictors: (i) AIC, (ii) BIC and (iii) EBIC	   
    bss_nr[N_chunk, ii]   = getindex.(bkm_results_all, 2)
  
  # Show progress
    #@info string("ω = ", ω[ii], "; ", "nz_β = ", nz_β[ii])
  
  end
  
  