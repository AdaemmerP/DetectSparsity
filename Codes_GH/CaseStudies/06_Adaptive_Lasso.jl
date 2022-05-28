# Set start dates
if dataset == 0

  startdate   = ["2015-09-01"]  

elseif dataset == 1   

  startdate  = ["1954-11-01", "1993-11-01"] 

elseif dataset == 2 

  startdate  = ["1993-11-01"] 

elseif dataset == 3 || dataset == 4 || dataset == 5

  startdate  = ["1964-11-01"] 

end


# Loop for all start dates  
for jj = 1:length(startdate)

  # Set parameters for model
    q0	= (findfirst(isequal.(Date(startdate[jj]), Xall.date))) |> Int64	 
    τ0  = 60

  # Compute all models with time series cross validation	
    results_all = adaptive_lasso_cvts(x, y_lead, q0, τ0)
    fc_lasso    = @views results_all[:, 1]

  # Get number of predictors != 0	
    nr_lasso    = results_all[:, 2] 

  # Compute MSEs of models
    mse_lasso  = mean((fc_lasso - y_lead[(q0 + τ0 + 1):(end - 1)]).^2)

  # MSE for historical mean	
    moos_total  = size(x, 1) - q0 - τ0 - 1
    hmean_iter  = map(x -> 1:(q0  + τ0 + x), 0:(moos_total - 1)) 
    hmean       = map(i -> mean(y_lead[i]), hmean_iter)               
    mse_hmean   = mean((hmean .- y_lead[(q0 + τ0 + 1):(end - 1)]).^2)

  # CW test
    CW_lasso = round(CW_test(y_lead[(q0 + τ0 + 1):(end - 1)], hmean, fc_lasso)[2], digits = 4)
 
  # Compute shrunk results
    fc_lasso_shrunk  = (fc_lasso .+ hmean)./2

  # Compute MSEs of shrunk models
    mse_lasso_shrunk  = mean((fc_lasso_shrunk - y_lead[(q0 + τ0 + 1):(end - 1)]).^2)

  # CW test
    CW_lasso_shrunk  = round(CW_test(y_lead[(q0 + τ0 + 1):(end - 1)], hmean, fc_lasso_shrunk)[2], digits = 4)
  
  # R²_oos and p-value by Clark and West
    if dataset != 0    
      @info repeat("-", 60)		
      @info string("Method: Adaptive Lasso")
      @info string("Start: ", Xall[q0 + τ0 + 2, 1])
      @info string("Roo² (Unshrunk):  ", round((1 - mse_lasso/mse_hmean), digits = 4)) 
      @info string("CW:  (Unshrunk):  ", CW_lasso)
      @info string("Roo² (Shrunk):    ", round((1 - mse_lasso_shrunk/mse_hmean), digits = 4)) 
      @info string("CW   (Shrunk):    ", CW_lasso_shrunk)
      @info repeat("-", 60)	
    end	

  # Save results 
  if save_results == true
      df_lasso = DataFrame(date 		   = Xall.date[(q0 + τ0 + 2):(end)], 
                          y_true       = y_lead[(q0 + τ0 + 1):(end - 1)], 
                          fcast_lasso  = fc_lasso, 
                          hmean        = hmean, 
                          nr_lasso     = nr_lasso)
      df_lasso = hcat(df_lasso, DataFrame(results_all[:, 3:end], :auto))         
      rename!(df_lasso, [names(df_lasso)[1:5]; names(Xall)[4:end]])   

      lasso_name    = string(save_path, "adaptive_lasso_", 
                              Xall.date[q0 + τ0 + 2], ".csv")	

      CSV.write(lasso_name, df_lasso)
    end
end