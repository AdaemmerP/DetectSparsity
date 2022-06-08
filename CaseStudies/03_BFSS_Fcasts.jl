# Extract start dates 
  startdate   = sdates_noval[findfirst(first.(sdates_noval) .== dataset)][2:end]

# Loop to run all samples
  for jj = 1:length(startdate)

    # Initial months for training
      q0 = findfirst(isequal.(Date(startdate[jj]), Xall.date))

    # Number of total OOS forecasts	
      moos_total = size(x, 1) - q0 - 1

    # Iterators for training and testing
      sample_train     = map(x -> 1:(q0 + x), 0:(moos_total - 1)) 
      obs_test         = last.(sample_train) .+ 1
      traintest_tuples = map(ii -> (sample_train[ii], obs_test[ii]), 1:length(obs_test))

    # OOS function for GLP code
      glp_results = pmap(traintest_tuples) do ii

                GLP_oos(x[ii[1], :], x[ii[2], :], y_lead[ii[1]], 
                          M, N, abeta, bbeta, Abeta, Bbeta)                     

                  end

      glp_results = reduce(vcat, glp_results)			

    # Compute historical mean	and SSRs
      y_test    = map(ii -> y_lead[(ii[2])],  		 traintest_tuples)
      hmean     = map(ii -> mean(y_lead[ii[1]]),   traintest_tuples)
      ssr_hm    = mean((y_test .- hmean).^2)
      ssr_fc    = mean((y_test .- glp_results[:, 1]).^2)
     
    # Compute MSE of shrunk forecasts  
      fc_shrunk     = (glp_results[:, 1] + hmean)./2
      ssr_fc_shrunk = mean((y_test .- fc_shrunk).^2)

    # OOS R^2	
      R_OOS        = 1 - ssr_fc/ssr_hm
      R_OOS_shrunk = 1 - ssr_fc_shrunk/ssr_hm

    # Print results
      if dataset != 0            
        @info repeat("-", 60)	
        @info string("Method: Bayesian FSS")
        @info string("Start: ", Xall[q0 + 2, 1])
        @info string("Roo² (Unshrunk): ", [round(R_OOS,   digits = 4)])
        @info string("CW:  (Unshrunk): ", [round(CW_test(y_lead[(q0 + 1):(end - 1)], hmean, glp_results[:, 1])[2], digits = 4)])
        @info string("Roo² (Shrunk):   ", [round(R_OOS_shrunk,   digits = 4)])
        @info string("CW   (Shrunk):   ", [round(CW_test(y_lead[(q0 + 1):(end - 1)], hmean, fc_shrunk)[2], digits = 4)])    
        @info repeat("-", 60)	
      end		

   # Save results 
   if save_results == true
        df_GLP  = DataFrame(date       = Xall.date[(q0 + 2):(end)], 
                            y_true     = y_lead[(q0 + 1):(end - 1)], 
                            fcast_glp  = glp_results[:, 1], 
                            hmean      = hmean, 
                            nr_glp     = glp_results[:, 2])
        df_GLP    = hcat(df_GLP, DataFrame(glp_results[:, 3:end], :auto))   
        rename!(df_GLP, [names(df_GLP)[1:5]; names(Xall)[4:end]])       

        
        save_name = string(save_path, "GLP_", Xall.date[q0 + 2], ".csv")	
        CSV.write(save_name,  df_GLP)
    end
  end