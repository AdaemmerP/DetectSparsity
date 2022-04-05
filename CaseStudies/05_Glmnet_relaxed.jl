# Set start dates
  if dataset == 0

    startdate   = ["2015-10-01"]  

  elseif dataset == 1   

    startdate  = ["1954-11-01", "1993-11-01"] 

  elseif dataset == 2 

    startdate  = ["1993-11-01"] 

  elseif dataset == 3 || dataset == 4 || dataset == 5

    startdate  = ["1964-11-01"] # , "1993-11-01"]

  end


 # Loop for all start dates  
  for jj = 1:length(startdate)

  # Set parameters for model
    q0	= (findfirst(isequal.(Date(startdate[jj]), Xall.date))) |> Int64	 
    τ0  = 60
    
  # Compute all models with time series cross validation	
    results_all = glmnet_relaxed_cvts(x, y_lead, q0, τ0, α, ζ)
    fc_relax    = @views results_all[:, 1]

  # Get number of predictors != 0	
    nr_relax   = @views results_all[:, 2] 
      
  # Get level of relaxation 
    ζ_val      = @views results_all[:, 3] 

  # Compute MSEs of models
    mse_model  = mean((fc_relax - y_lead[(q0 + τ0 + 1):(end - 1)]).^2)

  # MSE for historical mean	
    moos_total  = size(x, 1) - q0 - τ0 - 1
    hmean_iter  = map(x -> 1:(q0  + τ0 + x), 0:(moos_total - 1)) 
    hmean       = map(i -> mean(y_lead[i]), hmean_iter)               
    mse_hmean   = mean((hmean .- y_lead[(q0 + τ0 + 1):(end - 1)]).^2)

  # CW test
    CW  = round(CW_test(y_lead[(q0 + τ0 + 1):(end - 1)], hmean, fc_relax)[2], digits = 4)	
      
  # Compute shrunk results
    fc_shrunk  = (fc_relax  .+ hmean)./2
    
  # Compute MSEs of shrunk models
    mse_shrunk = mean((fc_shrunk - y_lead[(q0 + τ0 + 1):(end - 1)]).^2)

  # CW test
    CW_shrunk  = round(CW_test(y_lead[(q0 + τ0 + 1):(end - 1)], hmean, fc_shrunk)[2], digits = 4)	
    
  # R²_oos and p-value by Clark and West
    if dataset != 0    
      @info repeat("-", 60)		
      @info string("Method: Relaxed Lasso")
      @info string("Start: ", Xall[q0 + τ0 + 2, 1])
      @info string("Roo² (Unshrunk): ", round((1 .- mse_model./mse_hmean), digits = 4)) 
      @info string("CW   (Unshrunk): ", [CW])
      @info string("Roo² (Shrunk):   ", round((1 .- mse_shrunk./mse_hmean), digits = 4)) 
      @info string("CW   (Shrunk):   ", [CW_shrunk])
      @info repeat("-", 60)	
    end

  # Save results 
    if save_results == true
        df_lasso_rel = DataFrame(date            = Xall.date[(q0 + τ0 + 2):(end)], 
                                  y_true          = y_lead[(q0 + τ0 + 1):(end - 1)], 
                                  fcast_lassorel  = fc_relax, 
                                  hmean           = hmean, 
                                  nr_lasso_rel    = nr_relax)
        df_lasso_rel = hcat(df_lasso_rel, DataFrame(results_all[:, 4:end], :auto))         
                      rename!(df_lasso_rel, [names(df_lasso_rel)[1:5]; names(Xall)[4:end]])   

        lasso_name_rel  = string(save_path, "lasso_rel_", Xall.date[q0 + τ0 + 2], ".csv")	
            
        CSV.write(lasso_name_rel, df_lasso_rel)
    end
   
  end