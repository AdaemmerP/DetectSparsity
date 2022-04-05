
# Set start dates
  if dataset == 1   
         
    startdate     = ["1959-11-01"; "1998-11-01"] 

  elseif dataset == 2  

    startdate     = ["1998-11-01"]  

  elseif dataset == 3  

    startdate     = ["1969-11-01"] 

  end

# Loop to get forecasts for oos samples  
for jj = 1:length(startdate)

  # Find training observations  
    q0   		       = (findfirst(isequal.(Date(startdate[jj]), Xall.date))) |> Int64
    moos_total  	 = size(x, 1) - q0 - 1

  # Compute models	
    fcasts_glmnet   = pmap(i -> glmnet_forecasts_cv(x, y_lead, q0, moos_total, i), αvec)
    fcasts_enet     = @views fcasts_glmnet[1][:, 1]
    fcasts_lasso		= @views fcasts_glmnet[2][:, 1]
    fcasts_ridge 		= @views fcasts_glmnet[3][:, 1]
    fcast_all       = hcat(fcasts_enet, fcasts_lasso, fcasts_ridge)

  # Get number of predictors != 0	
    nr_enet         = @views fcasts_glmnet[1][:, 2] 
    nr_lasso        = @views fcasts_glmnet[2][:, 2] 
    nr_ridge        = @views fcasts_glmnet[3][:, 2] 

  # Compute MSEs 
    mse_models  = map(i -> mean((i .- y_lead[(q0 + 1):(end - 1)]).^2), eachcol(fcast_all))

  # MSE for historical mean	
    hmean_iter  = map(x -> 1:(q0 + x), 0:(moos_total - 1)) 
    hmean       = map(i -> mean(y_lead[i]), hmean_iter)               
    mse_hmean   = mean((hmean .- y_lead[(q0 + 1):(end - 1)]).^2)

  # CW test
    CW1 = round(CW_test(y_lead[(q0 + 1):(end - 1)], hmean, fcast_all[:, 1])[2], digits = 4)	
    CW2 = round(CW_test(y_lead[(q0 + 1):(end - 1)], hmean, fcast_all[:, 2])[2], digits = 4)	
    CW3 = round(CW_test(y_lead[(q0 + 1):(end - 1)], hmean, fcast_all[:, 3])[2], digits = 4)	

  # Rsquared and p-value by Clark and West
    @info repeat("-", 60)		
    @info string("Start: ", Xall[q0 + 2, 1])
    @info string("Roo²:  ", round.((1 .- mse_models./mse_hmean), digits = 4)) 
    @info string("CW:    ", [CW1, CW2, CW3])
    @info repeat("-", 60)			
end	