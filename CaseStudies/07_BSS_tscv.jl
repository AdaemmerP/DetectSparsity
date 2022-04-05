
  if dataset       == 0    
      
      startdate   = ["2015-09-01"]   

    elseif dataset == 1  

      startdate   = ["1954-11-01", "1993-11-01"] 

    elseif dataset == 2  

      startdate   = ["1993-11-01"]   

  end

# Loop to run all samples
  for jj = 1:length(startdate)

    # Set initial parameters and make iterators
      q0 	= (findfirst(isequal.(Date(startdate[jj]), Xall.date))) |> Int64
      τ0 	= 60
      moos_total   = size(x, 1) - q0 - 1
      sample_train = map(i -> 1:(q0 + i), 0:(moos_total - 1)) 
      
    # Compute historical mean	
      hist_mean   = @views map(i -> mean(y_lead[i]), sample_train)

    # Model and iterator combinations for individual forecasts
      models       = collect(combinations(1:size(x, 2))) 
      models_time  = collect(Iterators.product(Tuple(x for x in sample_train), models))

    # Compute all individual forecasts for unshrinked (1) and shrinked (2) univariate foreacsts	
      fcast_comb  = ThreadsX.map(i -> ols_forecasts(x[i[1], i[2]], 
                                                    y_lead[i[1]], 
                                                    x[last(i[1]) + 1, i[2]]), models_time)

    # Compute squared errors 
      ϵ_sq 					= @views (fcast_comb .- y_lead[last.(sample_train) .+ 1]).^2

    # Compute MSE recursively	for model selection
      recurs_mean	  = cumsum(ϵ_sq, dims = 1)./collect(1:size(ϵ_sq, 1))
      recurs_mean   = recurs_mean[τ0:end, :]
          
    # Get column-index of that model with lowest MSE until period t
      index_pick  = map(i -> i[2], argmin(recurs_mean, dims = 2))

    # Pick (out-of-sample) models
      index_oos		= hcat(collect(1:(length(index_pick) - 1)) .+ τ0, vec(index_pick[1:(end - 1)]))
    
    # Compute MSE for historical mean	
      mse_hmean = mean(ϵ_sq[(τ0 + 1):end, 1]) 
      MSE_hm 		= mean(((hist_mean .- y_lead[last.(sample_train) .+ 1]).^2)[(τ0 + 1):end])
      
    # Compute R_OOS: (i) unshrunken combination, (ii) shrunken combinations, (iii) RSZ
      R_OO2	 = 1 - mean(map(i -> ϵ_sq[i[1], i[2]], eachrow(index_oos)))/mse_hmean

    # Compute p-value by Clark and West (2007)	for unshrunken switching model
      e1 = hist_mean[(τ0 + 1):end];
      e2 = map(i -> fcast_comb[i[1], i[2]], eachrow(index_oos));
    
    # Print results
      @info repeat("-", 60)	
      @info string("Start: ", Xall[q0 + τ0 + 2, 1])
      @info string("Roo²:  ", round(R_OO2,   digits = 4))
      @info string("CW:    ", round(CW_test(y_lead[(q0 + τ0 + 1):(end - 1)], e1, e2)[2],   digits = 4))
      @info string("#Avg:  ", round(mean(size.(models[index_oos[:, 2]], 1)), digits = 4))
      @info repeat("-", 60)		
      
    # Save results 
        # Substract one from model index because "1" = intercept only model (historical mean)
          paramsin_1 = map(i -> i, model_comb[index_oos[:, 2]]) 
          numparams  = collect(1:size(x, 2)) 

        # Get included predictors  
          paramsin_1 = (map(i -> numparams .== i',  paramsin_1))
          paramsin_1 = reduce(vcat, map(i -> sum(i, dims = 2)', paramsin_1))

       if save_results == true
          df_bss_tscv = DataFrame(date           = Xall.date[(q0 + τ0 + 2):(end)], 
                                  y_true         = y_lead[(q0 + τ0 + 1):(end - 1)], 
                                  fcast_bss_tscv = e2, 
                                  hmean          = e1, 
                                  nr_bss_tscv    = length.(model_comb[index_oos[:, 2]]))

          df_bss_tscv = hcat(df_bss_tscv, DataFrame(paramsin_1, :auto))
          rename!(df_bss_tscv, [names(df_bss_tscv)[1:5]; names(Xall)[4:end]])  
        
                                
          bss_tscv_name = string(save_path, "bss_tscv_",  Xall.date[q0 + τ0 + 2], ".csv")	
        
          CSV.write(bss_tscv_name, df_bss_tscv)
       end

  end