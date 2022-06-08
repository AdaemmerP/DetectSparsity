 # Extract start dates 
 startdate   = sdates_val[findfirst(first.(sdates_val) .== dataset)][2:end]

# Loop to run all samples
  for jj = 1:length(startdate)

    # Set parameters for model
      q0	= (findfirst(isequal.(Date(startdate[jj]), Xall.date))) |> Int64	 

    # Compute all models with time series cross validation	
      results_all = map(i -> glmnet_forecasts_cvts(x, y_lead, q0, τ0, i), αvec)
      fc_enet     = @views results_all[1][:, 1]
      fc_lasso    = @views results_all[2][:, 1]
      fc_ridge    = @views results_all[3][:, 1]
      fc_all      = hcat(fc_enet, fc_lasso, fc_ridge)

    # Get number of predictors != 0	
      nr_enet     = @views results_all[1][:, 2] 
      nr_lasso    = @views results_all[2][:, 2] 
      nr_ridge    = @views results_all[3][:, 2] 

    # Compute MSEs of models
      mse_models  = map(i -> mean((i .- y_lead[(q0 + τ0 + 1):(end - 1)]).^2), eachcol(fc_all))

    # MSE for historical mean	
      moos_total  = size(x, 1) - q0 - τ0 - 1
      hmean_iter  = map(x -> 1:(q0  + τ0 + x), 0:(moos_total - 1)) 
      hmean       = map(i -> mean(y_lead[i]), hmean_iter)               
      mse_hmean   = mean((hmean .- y_lead[(q0 + τ0 + 1):(end - 1)]).^2)

    # CW test
      CW_enet  = round(CW_test(y_lead[(q0 + τ0 + 1):(end - 1)], hmean, fc_all[:, 1])[2], digits = 4)	
      CW_lasso = round(CW_test(y_lead[(q0 + τ0 + 1):(end - 1)], hmean, fc_all[:, 2])[2], digits = 4)
      CW_ridge = round(CW_test(y_lead[(q0 + τ0 + 1):(end - 1)], hmean, fc_all[:, 3])[2], digits = 4)

    # Compute shrunk results
      fc_enet_shrunk   = (fc_enet  .+ hmean)./2
      fc_lasso_shrunk  = (fc_lasso .+ hmean)./2
      fc_ridge_shrunk  = (fc_ridge .+ hmean)./2
      fc_all_shrunk    = hcat(fc_enet_shrunk, fc_lasso_shrunk, fc_ridge_shrunk)

    # Compute MSEs of shrunk models
      mse_models_shrunk  = map(i -> mean((i .- y_lead[(q0 + τ0 + 1):(end - 1)]).^2), eachcol(fc_all_shrunk))

    # CW test
      CW_enet_shrunk   = round(CW_test(y_lead[(q0 + τ0 + 1):(end - 1)], hmean, fc_all_shrunk[:, 1])[2], digits = 4)	
      CW_lasso_shrunk  = round(CW_test(y_lead[(q0 + τ0 + 1):(end - 1)], hmean, fc_all_shrunk[:, 2])[2], digits = 4)
      CW_ridge_shrunk  = round(CW_test(y_lead[(q0 + τ0 + 1):(end - 1)], hmean, fc_all_shrunk[:, 3])[2], digits = 4)

    # R²_oos and p-value by Clark and West
      if dataset != 0    
        @info repeat("-", 60)		
        @info string("Methods: E-Net, Lasso, Ridge")
        @info string("Start: ", Xall[q0 + τ0 + 2, 1])
        @info string("Roo² (Unshrunk):  ", round.((1 .- mse_models./mse_hmean), digits = 4)) 
        @info string("CW:  (Unshrunk):  ", [CW_enet, CW_lasso, CW_ridge])
        @info string("Roo² (Shrunk):    ", round.((1 .- mse_models_shrunk./mse_hmean), digits = 4)) 
        @info string("CW   (Shrunk):    ", [CW_enet_shrunk, CW_lasso_shrunk, CW_ridge_shrunk])
        @info repeat("-", 60)	
      end	

    # Save results 
    if save_results == true
        df_enet = DataFrame(date    	  = Xall.date[(q0 + τ0 + 2):(end)], 
                            y_true 		  = y_lead[(q0 + τ0 + 1):(end - 1)], 
                            fcast_enet  = fc_enet, 
                            hmean   		= hmean, 
                            nr_enet 		= nr_enet)
        df_enet = hcat(df_enet, DataFrame(results_all[1][:, 3:end], :auto))         
                  rename!(df_enet, [names(df_enet)[1:5]; names(Xall)[4:end]])           

        df_lasso = DataFrame(date 		   = Xall.date[(q0 + τ0 + 2):(end)], 
                            y_true       = y_lead[(q0 + τ0 + 1):(end - 1)], 
                            fcast_lasso  = fc_lasso, 
                            hmean        = hmean, 
                            nr_lasso     = nr_lasso)
        df_lasso = hcat(df_lasso, DataFrame(results_all[2][:, 3:end], :auto))         
        rename!(df_lasso, [names(df_lasso)[1:5]; names(Xall)[4:end]])   

        df_ridge = DataFrame(date 		   = Xall.date[(q0 + τ0 + 2):(end)], 
                            y_true      = y_lead[(q0 + τ0 + 1):(end - 1)], 
                            fcast_ridge = fc_ridge, 
                            hmean       = hmean, 
                            nr_ridge    = nr_ridge)
        df_ridge     = hcat(df_ridge, DataFrame(ones(size(df_ridge, 1), size(x, 2)), :auto))
        rename!(df_ridge, [names(df_ridge)[1:5]; names(Xall)[4:end]])  
        
        enet_name 	  = string(save_path, "enet_",    
                                Xall.date[q0 + τ0 + 2], ".csv")	
        lasso_name    = string(save_path, "lasso_", 
                                Xall.date[q0 + τ0 + 2], ".csv")	
        ridge_name    = string(save_path, "ridge_",       
                                Xall.date[q0 + τ0 + 2], ".csv")	

        CSV.write(enet_name,  df_enet)
        CSV.write(lasso_name, df_lasso)
        CSV.write(ridge_name, df_ridge)
    end
  end