# Set start dates
  if  dataset == 0  

      startdate     = ["2020-09-01"]

    elseif dataset == 1   

      startdate     = ["1959-11-01", "1998-11-01"] 
      
    elseif dataset == 2  

      startdate     = ["1998-11-01"] 

    elseif  dataset == 3 || dataset == 4  || dataset == 5

      startdate     = ["1969-11-01"] 

  end

 for jj = 1:length(startdate) 

  # Start training
    q0 	= (findfirst(isequal.(Date(startdate[jj]), Xall.date))) |> Int64

  # Number of total OOS forecasts	
    moos_total    = size(x, 1) - q0 - 1

  # Samples for training and observation for test prediction
    sample_train  = map(x  -> 1:(q0  +  x), 0:(moos_total - 1)) 
    obs_test      = last.(sample_train) .+ 1
    train_test    = map(i -> (sample_train[i], obs_test[i]), 1:length(obs_test))
  
  # Compute results  
    results_bkm = pmap(train_test) do ii

                  x_train = x[ii[1], :]
                  y_train = y_lead[ii[1]]
                  x_test  = reshape(x[ii[2], :], 1, :)
                  bkm_sc(x_train, y_train, x_test) 

                end

  # Historical mean               
    hmean 		= map(i -> mean(y_lead[i]), sample_train)
    mse_hmean = mean((y_lead[obs_test] - hmean).^2)
 
  # MSE of (unshrunken) models
    #fc_aic       = @views getindex.(results_bkm, 1)           
    #fc_bic       = @views getindex.(results_bkm, 2)      
    fc_ebic      = @views getindex.(results_bkm, 3)      
    #mse_bss_aic  = mean((y_lead[obs_test] - fc_aic).^2)
    #mse_bss_bic  = mean((y_lead[obs_test] - fc_bic).^2)
    mse_bss_ebic = mean((y_lead[obs_test] - fc_ebic).^2)

  # MSE of (shrunken) models
    #fc_aic_shrunk       = (@views getindex.(results_bkm, 1) .+ hmean)./2          
    #fc_bic_shrunk       = (@views getindex.(results_bkm, 2) .+ hmean)./2           
    fc_ebic_shrunk      = (@views getindex.(results_bkm, 3) .+ hmean)./2           
    #mse_bss_aic_shrunk  = mean((y_lead[obs_test] - fc_aic_shrunk).^2)
    #mse_bss_bic_shrunk  = mean((y_lead[obs_test] - fc_bic_shrunk).^2)
    mse_bss_ebic_shrunk = mean((y_lead[obs_test] - fc_ebic_shrunk).^2)
    
    
  # Show all results
    if dataset != 0    
      @info repeat("-", 60)		
      @info string("Method: BSS")
      @info string("Start: ", Xall[q0 + 2, 1])
      # Results unshrunk
      #@info string("Roo² (AIC):  ", round(1 - mse_bss_aic/mse_hmean,  digits = 4))
      #@info string("Roo² (BIC):  ", round(1 - mse_bss_bic/mse_hmean,  digits = 4))
      @info string("Roo² (Unshrunk): ", round(1 - mse_bss_ebic/mse_hmean, digits = 4))
      #@info string("CW (AIC):    ", round(CW_test(y_lead[obs_test], hmean, fc_aic)[2],  digits = 4))
      #@info string("CW (BIC):    ", round(CW_test(y_lead[obs_test], hmean, fc_bic)[2],  digits = 4))
      @info string("CW   (Unshrunk): ", round(CW_test(y_lead[obs_test], hmean, fc_ebic)[2], digits = 4))
      # Results shrunk
      #@info string("Roo² (AIC-shrunk):  ", round(1 - mse_bss_aic_shrunk/mse_hmean,  digits = 4))
      #@info string("Roo² (BIC-shrunk):  ", round(1 - mse_bss_bic_shrunk/mse_hmean,  digits = 4))
      @info string("Roo² (Shrunk):   ", round(1 - mse_bss_ebic_shrunk/mse_hmean, digits = 4))
      #@info string("CW (AIC-shrunk):    ", round(CW_test(y_lead[obs_test],  hmean, fc_aic_shrunk)[2],  digits = 4))
      #@info string("CW (BIC-shrunk):    ", round(CW_test(y_lead[obs_test],  hmean, fc_bic_shrunk)[2],  digits = 4))
      @info string("CW   (Shrunk):   ", round(CW_test(y_lead[obs_test],  hmean, fc_ebic_shrunk)[2], digits = 4))
      @info repeat("-", 60)	
    end

  # Save results (with EBIC)
    if save_results == true
        df_bss = DataFrame(date    		 = Xall.date[(q0 + 2):end], 
                          y_true  		 = y_lead[obs_test], 
                          fcast_bss    = fc_ebic, 
                          hmean   		 = hmean,                   
                          nr_bss_ebic = getindex.(results_bkm, 6))
        df_preds = DataFrame(reduce(vcat, getindex.(results_bkm, [7:size(results_bkm[1], 2)])'), :auto)
        df_bss   = hcat(df_bss, df_preds)
                  rename!(df_bss, [names(df_bss)[1:5]; names(Xall)[4:end]]) 
        
        bss_name = string(save_path, "bss_",       
                          Xall.date[q0 + 2], ".csv")	

        CSV.write(bss_name, df_bss)
    end
    
  end

