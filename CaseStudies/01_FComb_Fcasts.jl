
# Set start dates (0 = pre-compilation only)
  if dataset       == 0    
  
      startdate   = ["2015-09-01"]   

    elseif dataset == 1  
  
      startdate   = ["1954-11-01", "1993-11-01"]    

    elseif dataset == 2

      startdate   = ["1993-11-01"]      

    elseif  dataset == 3 || dataset == 4 || dataset == 5

      startdate   = ["1964-11-01"] 

  end

  # Get combinations of univariate forecasts (+ 1 because historical mean will be added=
    size_x2 = (size(x, 2)) + 1
    if last(size_x2) > 20
        model_comb   = reduce(vcat, map(ii -> collect(combinations(1:size_x2, ii)), (1:3)))
        model_comb   = vcat(model_comb, [collect(2:(size_x2))])
    else
        model_comb   = collect(combinations(1:size_x2)) 
    end  

  # Convert to Int8
    model_comb = @views convert(Vector{Vector{Int8}}, model_comb)

#  Loop to run all samples
  for jj = 1:length(startdate)

    # Set initial parameters and make iterators
      q0  = (findfirst(isequal.(Date(startdate[jj]), Xall.date))) |> Int64
      τ0  = 60
      moos_total = size(x, 1) - q0 - 1 # Total number of OOS forecasts (!= final forecast number)
      sample_train = map(i -> 1:(q0 + i), 0:(moos_total - 1)) 
    
    # Compute historical mean	x
      hist_mean    = @views map(i -> mean(y_lead[i]), sample_train)

    # Model and iterator combinations for individual forecasts
      models       = 1:size(x, 2) 
      models_time  = collect(Iterators.product(Tuple(x for x in sample_train), models))

    # Compute univariate forecasts
      fcast_univ  = @views map(i -> ols_forecasts(x[i[1], i[2]], y_lead[i[1]], 
                                                  x[last(i[1]) + 1, i[2]]), 
                                                  models_time)

    # Add historical mean (intercept only regression) 
      fcast_univ  = hcat(hist_mean, fcast_univ) 	
     
    # Compute tuples of time and combination possibilities  
      comb_t = collect(Iterators.product(map(i -> [i], Int16.(1:size(fcast_univ, 1))), model_comb))
    
    # Compute squared errors of individually unshrunk models 
      ϵ_sq_all     = @views comb_sqf(comb_t, fcast_univ, y_lead[last.(sample_train) .+ 1], hist_mean)
      ϵ_sq         = @views ϵ_sq_all[1] 
      ϵ_sq_shrunk  = @views ϵ_sq_all[2] 
   
    # Compute MSE recursively	
      recurs_mse = @views cumsum(ϵ_sq, dims = 1)./collect(1:size(ϵ_sq, 1))
   
    # Get MSE after initialization  
      recurs_mse = @views recurs_mse[τ0:end, :]
     
    # Get column-index of that model with lowest MSE until period t
      index_pick = getindex.(argmin(recurs_mse, dims = 2), 2)
   
    # Pick (out-of-sample) models (start picking model at τ0 + 1)
      index_oos = hcat(collect(1:(length(index_pick) - 1)) .+ τ0, vec(index_pick[1:(end - 1)]))
      index_ew  = findall(map(i -> i == collect(2:(size(x, 2) + 1)), model_comb))[1]

    # Compute MSE for historical mean	
      MSE_hm = mean(ϵ_sq[(τ0 + 1):end, 1])   

    # Compute R_OOS: (i) unshrunken combination, (ii) shrunken combinations, (iii) RSZ
      R_OOS_flex	    	= @views 1 - mean(map(i -> ϵ_sq[i[1], i[2]], eachrow(index_oos)))/MSE_hm
      R_OOS_flex_shrunk = @views 1 - mean(map(i -> ϵ_sq_shrunk[i[1], i[2]], eachrow(index_oos)))/MSE_hm

      R_OOS_ew         = @views 1 - mean(ϵ_sq[index_oos[:, 1],        index_ew])/MSE_hm
      R_OOS_ew_shrunk  = @views 1 - mean(ϵ_sq_shrunk[index_oos[:, 1], index_ew])/MSE_hm

    # Get historical mean for Clark and West test
      e11 = hist_mean[(τ0 + 1):end];

    # Get final forecasts of combinations (for CW test)
      e12        = @views map(i -> mean(fcast_univ[i[1], model_comb[i[2]]]), eachrow(index_oos))     
      e12_shrunk = @views (e12 + e11)./2
  
    # Compute equal weights
      e22        = @views map(i -> mean(fcast_univ[i[1], model_comb[index_ew]]), eachrow(index_oos))    
      e22_shrunk = @views (e22 + e11)./2
        
    # Print results
      if dataset != 0    
        @info repeat("-", 60)	
        @info string("Method: Forecast combinations")
        @info string("Start: ", Xall[q0 + τ0 + 2, 1])
        @info string("Roo² (Unshrunk): ",  [round(R_OOS_flex,  digits = 4) round(R_OOS_ew, digits = 4)] ) 
        @info string("CW   (Unshrunk): ",  [round(CW_test(y_lead[(q0 + τ0 + 1):(end - 1)], e11, e12)[2],   digits = 4)                                     
                                          round(CW_test(y_lead[(q0 + τ0 + 1):(end - 1)], e11, e22)[2],   digits = 4) ])
        @info string("Roo² (Shrunk):   ",  [round(R_OOS_flex_shrunk,  digits = 4) round(R_OOS_ew_shrunk, digits = 4)] ) 
        @info string("CW   (Shrunk):   ",  [round(CW_test(y_lead[(q0 + τ0 + 1):(end - 1)], e11, e12_shrunk)[2],   digits = 4) 
                                          round(CW_test(y_lead[(q0 + τ0 + 1):(end - 1)], e11, e22_shrunk)[2],   digits = 4) ])
        @info repeat("-", 60)	
      end
      
    # Substract one from model index because "1" = intercept only model (historical mean)
      paramsin_1 = map(i -> i .- 1, model_comb[index_oos[:, 2]]) 
      numparams  = collect(1:size(x, 2)) 

    # Get included predictors  
      paramsin_1 = (map(i -> numparams .== i',  paramsin_1))
      paramsin_1 = reduce(vcat, map(i -> sum(i, dims = 2)', paramsin_1))
   
    # Save results 
    if save_results == true
        df_fcomby = DataFrame(date        = Xall.date[(q0 + τ0 + 2):(end)], 
                              y_true      = y_lead[(q0 + τ0 + 1):(end - 1)], 
                              fcast_comby = e12, 
                              hmean       = hist_mean[(τ0 + 1):end], 
                              nr_comby    = length.(model_comb[index_oos[:, 2]]))

        df_fcomby = hcat(df_fcomby, DataFrame(paramsin_1, :auto))
        rename!(df_fcomby, [names(df_fcomby)[1:5]; names(Xall)[4:end]])  

        df_ew   = DataFrame(date       = Xall.date[(q0 + τ0 + 2):(end)], 
                            y_true     = y_lead[(q0 + τ0 + 1):(end - 1)], 
                            fcast_ew   = e22, 
                            hmean      = hist_mean[(τ0 + 1):end], 
                            nr_ew      = length(model_comb[index_ew]))

        df_ew = hcat(df_ew, DataFrame(ones(size(df_ew, 1), size(x, 2)), :auto))
        rename!(df_ew, [names(df_ew)[1:5]; names(Xall)[4:end]])  

                              
        fcomby_name = string(save_path, "fcomby_",  Xall.date[q0 + τ0 + 2], ".csv")	
        ew_name     = string(save_path, "fcombew_", Xall.date[q0 + τ0 + 2], ".csv")	

        CSV.write(fcomby_name, df_fcomby)
        CSV.write(ew_name, df_ew)
    end
  end



