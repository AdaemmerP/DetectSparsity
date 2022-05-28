# -------------------------------------------------------------------------		
#           Function for OOS forecasts with GLP approach
# -------------------------------------------------------------------------	
  function GLP_oos(xtrain::Matrix{Float64}, 
                   xtest::Vector{Float64}, 
                   ytrain::Vector{Float64},  
                   M::Int64, N::Int64, 
                   abeta::Float64, bbeta::Float64, 
                   Abeta::Float64, Bbeta::Float64) 

      # Set seed for reproducibility 
        seedit = Int(floor(abs(sum(xtrain))))
        Random.seed!(seedit)

      # Make vector for ones				
        u::Array{Float64} = repeat([1.0], size(xtrain, 1), 1)

      # Compute mean and standard deviation and standardize predictors 
        μ_x    = reshape(mean(xtrain, dims = 1), :, 1)
        σ_x    = reshape(std(xtrain,  dims = 1), :, 1)
        xtrain = standardize(ZScoreTransform, xtrain, dims = 1) 

      # Compute GLP model											
        results_temp = GLP_SpikeSlab(ytrain, xtrain, u, abeta, bbeta, Abeta, Bbeta, M)

      # Compute slope coefficients as mean of all draws			      										
        βhat = @views mean(results_temp[1][:, (N + 1):end], dims = 2)

      # Get intercept
        β0   = @views mean(results_temp[3][(N + 1):end]) 

      # Standardise test observations for prediction	
        xtest = (xtest - μ_x)./σ_x

      # OOS prediction
        yhat = dot([1; xtest], [β0; βhat]) 	
    
      # Return (i) forecast (ii) average number of included predictors and (iii) beta coefficients
        z_avg_col::Float64  = mean(map(i -> sum(i), eachcol(results_temp[2][:, (N + 1):end]))) 	 
        z_avg_row  = map(i -> mean(i .!= 0), eachrow(results_temp[2][:, (N + 1):end]))

      return [yhat z_avg_col z_avg_row'] 

  end
# -------------------------------------------------------------------------	


# -------------------------------------------------------------------------	
#           	    Function for univariate OLS-OOS forecasts
# -------------------------------------------------------------------------	
	function ols_forecasts(xtrain, ytrain, xtest) 
							
			# Add constant to regression model								
				xtrain = hcat(ones(size(xtrain, 1)), xtrain)
 
				βhat 	 = inv(xtrain'xtrain)xtrain'ytrain     				  
				yhat   = ([1.0 xtest']*βhat)[1]    
  
		return yhat

	end
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------	
#                         BKM for AIC, BIC and EBIC
# -------------------------------------------------------------------------	

# Load necessary R-package
  R"""
  library(BeSS)
  """

function bkm_sc(x_train::Array{Float64, 2},
                y_train::Array{Float64, 1},
                x_test::Array{Float64, 2})

# Put Julia data to R	
  @rput x_train y_train x_test

# Use BeSS package with R
  R"""
    # Train model
      bess_results = bess(x_train, y_train,
                          method = "sequential", epsilon = 0, family = "gaussian")
      
    # Count number of nonzero coefficients (excluding intercept)	
      nonzero_coef_aic  = coef(bess_results, sparse = F)[, which.min(bess_results$AIC)]	 != 0
      nonzero_coef_bic  = coef(bess_results, sparse = F)[, which.min(bess_results$BIC)]	 != 0
      nonzero_coef_ebic = coef(bess_results, sparse = F)[, which.min(bess_results$EBIC)] != 0

      sum_nz_aic  = sum(nonzero_coef_aic[2:length(nonzero_coef_aic)])
      sum_nz_bic  = sum(nonzero_coef_bic[2:length(nonzero_coef_bic)])
      sum_nz_ebic = sum(nonzero_coef_ebic[2:length(nonzero_coef_ebic)])

    # Forecasts
      fc_bkm_aic   = predict(bess_results, x_test, type = "AIC")
      fc_bkm_bic   = predict(bess_results, x_test, type = "BIC")
      fc_bkm_ebic  = predict(bess_results, x_test, type = "EBIC")
      
  """

    fc_bkm_aic::Float64        = @rget fc_bkm_aic
    fc_bkm_bic::Float64        = @rget fc_bkm_bic 
    fc_bkm_ebic::Float64       = @rget fc_bkm_ebic 
    sum_nz_aic::Float64        = @rget sum_nz_aic 
    sum_nz_bic::Float64        = @rget sum_nz_bic 
    sum_nz_ebic::Float64       = @rget sum_nz_ebic 
    nonzero_coef_ebic::Vector{Float64} = @rget nonzero_coef_ebic

  # Return values
    return hcat(fc_bkm_aic, 
                fc_bkm_bic, 
                fc_bkm_ebic, 
                sum_nz_aic, 
                sum_nz_bic, 
                sum_nz_ebic, 
                nonzero_coef_ebic[2:end]') # Remove intercept from parameters

  end


# -------------------------------------------------------------------------	
#           Function for Glmnet OOS forecasts with time series cv
# -------------------------------------------------------------------------	
function glmnet_forecasts_cvts(xtrain::Matrix{Float64}, 
                               y_lead::Vector{Float64}, 
                               q0::Int64, 
                               τ0::Int64,
                               α::Float64)

    # Compute number of total OOS forecasts	
      moos_total   = size(xtrain, 1) - q0 - τ0 - 1	

    # Pre-allocate vector for final forecasts	
      glmnet_fcast = zeros(moos_total, 1)
      nrs_preds    = zeros(moos_total, 1)
      params_in    = zeros(moos_total, size(xtrain, 2))

    # Iterator for training samples to choose λ	and iterators for cross validation
      sample_train = map(i -> 1:(q0 +  i), 0:(moos_total - 1)) 	
      sample_test  = map(i -> (last(i) + 1):(last(i) + τ0), sample_train) 

    # Loop for computing forecasts	
      Threads.@threads for ii = 1:length(sample_train)	

        # Get initial sequence of lambda values
          λ_init = @views glmnet(xtrain[sample_train[ii], :], 
                                 y_lead[sample_train[ii]], 
                                 alpha        = α, 
                                 standardize  = true, 
                                 intercept    = true).lambda
      

        # Create Matrix for cross validation forecasts and get sequence for cv sample										 
          cv_fcast  = zeros(length(λ_init), τ0)											 
          obs_test  = collect(sample_test[ii])		

        # Loop for cross validation		
        for jj = 1:τ0

            # Training sample sequence and test observation
              train_seq = 1:(obs_test[jj] - 1)
              xtest     = reshape(xtrain[obs_test[jj], :], 1, :)

            # Compute path with initialized λ values	
              fit_glmnet = @views glmnet(xtrain[train_seq, :], 
                                         y_lead[train_seq], 
                                         alpha       = α, 
                                         standardize = true, 
                                         intercept   = true,
                                         lambda      = λ_init)
                        
            # Make forecast											 
              predict_cv =  GLMNet.predict(fit_glmnet, xtest) 					 

            # Save forecast	
              cv_fcast[:, jj] = predict_cv

        end

      # Compute squared error	for each λ_init 
        ϵ_sq = (cv_fcast' .- y_lead[sample_test[ii]]).^2

      # Compute MSE recursively	
        mse_λ  = mean(ϵ_sq, dims = 1)
        λopt	 = argmin(mse_λ)[2]

      # Train final model with chosen λ	
        fit_glmnet = @views glmnet(xtrain[1:(last(obs_test)), :], 
                                   y_lead[1:(last(obs_test))], 
                                   alpha       = α, 
                                   standardize = true, 
                                   intercept   = true,
                                   lambda      = λ_init) 

      # Make final forecast	
        predict_oos = GLMNet.predict(fit_glmnet, reshape(xtrain[(last(obs_test) + 1), :], 1, :))[1, λopt] 
        # Lasso.predict(fit_glmnet, reshape(xtrain[(last(obs_test) + 1), :], 1, :))[1, λopt]  # 

      # Save final forecast	
        glmnet_fcast[ii] = predict_oos
        nrs_preds[ii] 	 = sum(fit_glmnet.betas[:, λopt] .!= 0.0)  # sum(fit_glmnet.coefs[:, λopt] .!= 0.0) 
        params_in[ii, :] = Float64.(fit_glmnet.betas[:, λopt] .!= 0.0)'

       # @info ii

    end

    # Return vector with forecasts and (number of) nonzero predictors
    return hcat(glmnet_fcast, nrs_preds, params_in) # 

end

# -------------------------------------------------------------------------	
#                       Function for Adaptive Lasso
# -------------------------------------------------------------------------	
function adaptive_lasso_cvts(xtrain::Matrix{Float64}, 
                             y_lead::Vector{Float64}, 
                             q0::Int64, 
                             τ0::Int64)

# Compute number of total OOS forecasts	
  moos_total   = size(xtrain, 1) - q0 - τ0 - 1	

# Pre-allocate vector for final forecasts	
  glmnet_fcast = zeros(moos_total, 1)
  nrs_preds    = zeros(moos_total, 1)
  params_in    = zeros(moos_total, size(xtrain, 2))

# Iterator for training and CV samples
  sample_train = map(i -> 1:(q0 +  i), 0:(moos_total - 1)) 	
  sample_test  = map(i -> (last(i) + 1):(last(i) + τ0), sample_train) 

# Loop for computing forecasts	
  Threads.@threads for ii = 1:length(sample_train)	

   # Compute Ridge model for weights 
    fit_glmnet = @views glmnet(xtrain[sample_train[ii], :], 
                               y_lead[sample_train[ii]], 
                               alpha        = 0, 
                               standardize  = true, 
                               intercept    = true)
    
     	# Get lambda sequence, sample size,  etc.
        n   	     = length(sample_train[ii]) 
    		λ_seq      = fit_glmnet.lambda				
    		bic_vec    = zeros(length(λ_seq), 1)	
    		Xmat       = hcat(ones(size(xtrain[sample_train[ii], :], 1)), xtrain[sample_train[ii], :])
        ridge_pred = GLMNet.predict(fit_glmnet, xtrain[sample_train[ii], :]) 

    	# Loop to compute BIC 
      # For computation see e.g.: https://hastie.su.domains/TALKS/enet_talk.pdf 

    	 for jj = 1:length(λ_seq)				

          Σϵ²   = sum((y_lead[sample_train[ii]] - ridge_pred[:, jj]).^2)
          Id    = λ_seq[jj]*I(size(xtrain, 2) + 1) # Plus one because of constant
          H     = Xmat*inv(Xmat'*Xmat + Id)*Xmat'
          df    = tr(H)
      
        # Compute BIC
          bic_vec[jj] = n*log(Σϵ²) + log(n)*df    
       end

        coef_ridge =  fit_glmnet.betas[:, argmin(bic_vec)[1]]
 
# Get initial sequence of lambda values (using weights for adaptive lasso)
  λ_init = @views glmnet(xtrain[sample_train[ii], :], 
                         y_lead[sample_train[ii]], 
                         alpha          = 1, 
                         standardize    = true, 
                         penalty_factor = 1 ./ abs.(coef_ridge),
                         intercept      = true).lambda

# Create Matrix for cross validation forecasts and get sequence for cv sample										 
  cv_fcast  = zeros(length(λ_init), τ0)											 
  obs_test  = collect(sample_test[ii])		

# Loop for cross validation		
  for jj = 1:τ0

# Training sample sequence and test observation
  train_seq = 1:(obs_test[jj] - 1)
  xtest     = reshape(xtrain[obs_test[jj], :], 1, :)

# Compute path with initialized λ values	
  fit_glmnet = @views glmnet(xtrain[train_seq, :], 
                             y_lead[train_seq], 
                             alpha          = 1, 
                             standardize    = true, 
                             intercept      = true,
                             penalty_factor = 1 ./ abs.(coef_ridge),
                             lambda         = λ_init)

# Make forecast											 
  predict_cv =  GLMNet.predict(fit_glmnet, xtest) 					 

# Save forecast	
  cv_fcast[:, jj] = predict_cv

end

# Compute squared error	for each λ_init 
  ϵ_sq = (cv_fcast' .- y_lead[sample_test[ii]]).^2

# Compute MSE recursively	
  mse_λ  = mean(ϵ_sq, dims = 1)
  λopt	 = argmin(mse_λ)[2]

# Train final model with chosen λ	
  fit_glmnet = @views glmnet(xtrain[1:(last(obs_test)), :], 
                             y_lead[1:(last(obs_test))], 
                             alpha          = 1, 
                             standardize    = true, 
                             intercept      = true,
                             penalty_factor = 1 ./ abs.(coef_ridge),
                             lambda         = λ_init) 

# Make finale forecast	
  predict_oos = GLMNet.predict(fit_glmnet, reshape(xtrain[(last(obs_test) + 1), :], 1, :))[1, λopt] 

# Save final forecast	
  glmnet_fcast[ii] = predict_oos
  nrs_preds[ii] 	 = sum(fit_glmnet.betas[:, λopt] .!= 0.0)  # sum(fit_glmnet.coefs[:, λopt] .!= 0.0) 
  params_in[ii, :] = Float64.(fit_glmnet.betas[:, λopt] .!= 0.0)'

end

# Return vector with forecasts and (number of) nonzero predictors
  return hcat(glmnet_fcast, nrs_preds, params_in)  

end



#-------------------------------------------------------------------------	
# 	              	Function for relaxed Lasso 
# -------------------------------------------------------------------------	
# Info: This function uses the pure Julia implementation of the coordinate 
#       descent algorithm since the estimated parameters can be more easily 
#       modified than in 'GLMNet.jl'. Yet the function uses GLMNet.jl to 
#       compute the initial values of λ

  function glmnet_relaxed_cvts(xtrain, y_lead, q0, τ0, α, ζ)

      # Number of total OOS forecasts	
        moos_total   = size(xtrain, 1) - q0 - τ0 - 1	

      # Pre-allocate vector for final forecasts	
        fcasts       = zeros(moos_total, 1)
        nrs_preds    = zeros(moos_total, 1)
        ζpick        = zeros(moos_total, 1)
        βpick        = zeros(size(xtrain, 2), moos_total)

      # Iterator for training samples to choose λ and ζ	
        sample_train = map(i -> 1:(q0 +  i), 0:(moos_total - 1)) 	
        sample_test  = map(i -> (last(i) + 1):(last(i) + τ0), sample_train) 

      # Loop for computing forecasts	 
        Threads.@threads for ii = 1:length(sample_train)	
      
        # Get initial sequence of lambda values with original Fortran code
          λ_init = @views  glmnet(xtrain[sample_train[ii], :], y_lead[sample_train[ii]], 
                                  alpha        = α, 
                                  standardize  = true, 
                                  intercept    = true).lambda 
    
         # Vectors to store validation results					 
           cv_fcast  = zeros(length(λ_init), τ0, length(ζ))											 
           obs_test  = collect(sample_test[ii])		


        # ---------------------- Begin validation loop ---------------------------- # 
        # Loop for cross validation		
          for jj = 1:τ0

            # Training sample sequence and test observation
               train_seq = 1:(obs_test[jj] - 1)
               xtest     = reshape(xtrain[obs_test[jj], :], 1, :)

             # Compute initial path  
               fitLStart = Lasso.fit(LassoPath, 
                                     xtrain[train_seq, :], y_lead[train_seq], 
                                     standardize = true,                                
                                     intercept   = true,
                                     λ = λ_init; 
                                     α = α) 

              # Get active OLS sets 
                ASets = map(i -> i .!= 0.0, eachcol(fitLStart.coefs))  

              # Pre-allocate Matrix for OLS coefficients
                LS_β  = zeros(size(xtrain, 2), length(ASets)) 
                LS_β0 = zeros(length(ASets))

                # Loop to compute OLS models      
                  for kk = 1:length(ASets)

                    if sum(ASets[kk]) == 0

                    # Intercept only 
                      LS_β0[kk] = mean(y_lead[train_seq])

                    else

                    # Make OLS predictions (X is not standardized because 'coefs' in Lasso.jl are also not standardized)
                      xglm                = @views hcat(ones(length(train_seq)), xtrain[train_seq, ASets[kk]])
                      βcoefs              = GLM.coef(lm(xglm, y_lead[train_seq]))
                      LS_β0[kk]           = @views βcoefs[1]
                      LS_β[ASets[kk], kk] = @views βcoefs[2:end]        

                    end

                  end

                # Get validation data  
                  xtest_tmp  = xtrain[obs_test[jj], :] 
                  ytest_temp = y_lead[obs_test[jj]]

                # Make validation forecasts for relaxed Lasso    
                  for ll = 1:length(ζ)

                      fittemp             = deepcopy(fitLStart)
                      fittemp.b0          = ζ[ll].*fittemp.b0    + (1 - ζ[ll]).*LS_β0
                      fittemp.coefs       = ζ[ll].*fittemp.coefs + (1 - ζ[ll]).*LS_β            
                      cv_fcast[:, jj, ll] = (Lasso.predict(fittemp, xtest_tmp') .- ytest_temp).^2

                  end
            end
        # ----------------------------------------------------------------------- #   

        # Compute MSEs  
          mse_allpr = @views map(i -> mean(cv_fcast[:, :, i], dims = 2), 1:length(ζ))
       
        # Find optimal lambda and ζ
        # Check whether results are equal
          if all(minimum.(mse_allpr) .== minimum.(mse_allpr)[1]) == true
            ζopt = length(ζ)   
            λopt = argmin(mse_allpr[1])[1]
          else
            ζopt = argmin(minimum.(mse_allpr))    
            λopt = argmin(mse_allpr[ζopt])[1]
          end

        # Train final model with chosen λ and ζ 
          xtrainf = xtrain[1:(last(obs_test)), :]
          ytrainf = y_lead[1:(last(obs_test))]     

        # Fit model for final forecast  
          fit_path = Lasso.fit(LassoPath,
                               xtrainf, ytrainf,                         
                               standardize = true, 
                               intercept   = true,                          
                               λ           = λ_init;
                               α           = α) 

        # Check whether coefficients have to be relaxed 
          if ζ[ζopt] == 1.0

           # Make final forecast with pure Lasso (no relaxation)	
             ffinal = Lasso.predict(fit_path, reshape(xtrain[(last(obs_test) + 1), :], 1, :))[1, λopt] 

          else
                
              # Compute relaxed coefficients     
                LS_β    = zeros(size(xtrainf, 2))      
                Afinal  = Vector(fit_path.coefs[:, λopt] .!= 0)

                  if sum(Afinal) == 0 
                     
                    # Intercept only 
                      LSb0 = mean(y_lead[1:(last(obs_test))])

                  else
                      xglm          = @views hcat(ones(size(xtrainf, 1), 1), xtrainf[:, Afinal]) 
                      βcoefs        = GLM.coef(lm(xglm, y_lead[1:(last(obs_test))]))
                      LSb0          = βcoefs[1] 
                      LS_β[Afinal]  = βcoefs[2:end] 
                  end

              # Relax coefficients      
                fit_path.coefs .= ζ[ζopt].*fit_path.coefs .+ (1 -  ζ[ζopt]).*LS_β
                fit_path.b0    .= ζ[ζopt].*fit_path.b0    .+ (1 -  ζ[ζopt]).*LSb0
                
              # Make final forecast with optimal λ 	
                ffinal = Lasso.predict(fit_path, reshape(xtrain[(last(obs_test) + 1), :], 1, :))[1, λopt] 
          end            

        # Save final forecast	
          fcasts[ii]     = ffinal
          nrs_preds[ii]  = @views sum(fit_path.coefs[:, λopt] .!= 0.0) 
          ζpick[ii]      = @views ζ[ζopt]
          βpick[:, ii]   = @views Vector{Float64}(fit_path.coefs[:, λopt] .!= 0.0) 

      end

    # Return vector with forecasts and (number of) nonzero predictors
      return [fcasts nrs_preds ζpick βpick']

end


#-------------------------------------------------------------------------	
#       Function for Glmnet OOS forecasts with cross sectional cv
#-------------------------------------------------------------------------	

  function glmnet_forecasts_cv(xtrain::Matrix{Float64}, 
                               y_lead::Vector{Float64}, 
                               q0::Int64,
                               moos_total::Int64,
                               α::Float64) 


  # Set seed for reproducibility 
    Random.seed!(convert(Int64, α*10 + 10))

  # Pre-allocate vectors for forecasts	
    glmnet_fcast = zeros(moos_total, 1)
    nrs_preds    = zeros(moos_total, 1)

  # Iterator for training samples
    sample_train = map(i -> 1:(q0 +  i), 0:(moos_total - 1)) 	

    for ii = 1:length(sample_train)	
                    
      fit_glmnet_cv = glmnetcv(xtrain[sample_train[ii], :], 
                               y_lead[sample_train[ii]], 
                               alpha       = α,
                               standardize = true, 
                               intercept   = true,
                               nfolds      = 10)

      # Make OOS prediction for evaluation			
      xtest			 = reshape(xtrain[last(sample_train[ii]) + 1, :], 1, :)								 
      fcast_temp = GLMNet.predict(fit_glmnet_cv, xtest)											 

      # Make final forecast	
      glmnet_fcast[ii] = fcast_temp[1]
      nrs_preds[ii] 	 = sum(GLMNet.coef(fit_glmnet_cv) .!= 0.0) 

    end

  # Return vector with forecasts
    return [glmnet_fcast nrs_preds]

  end


#-------------------------------------------------------------------------	
#                    Function for Clark and West test (2007)
#-------------------------------------------------------------------------	

  function CW_test(actual, forecast_1, forecast_2)

      e_1   				= actual - forecast_1
      e_2    				= actual - forecast_2
      f_hat  				= e_1.^2 - e_2.^2 + ((forecast_1 - forecast_2).^2)
      Y_f    				= f_hat
      X_f    				= ones(size(f_hat, 1), 1)
      beta_f 				= (inv(X_f'*X_f))*(X_f'*Y_f) 
      e_f           = Y_f - X_f*beta_f
      sig2_e        = (e_f'*e_f)/(size(Y_f,1) - 1)
      cov_beta_f    = sig2_e*inv(X_f'*X_f)
      MSPE_adjusted = beta_f/sqrt(cov_beta_f)
      p_value       = 1 - cdf(Normal(), MSPE_adjusted[1])

      return [MSPE_adjusted p_value]

	end
	

#-------------------------------------------------------------------------	
# 	       Function to compute means of univariate combinations
#-------------------------------------------------------------------------		

  function meanscf(comb_t, fcasts_mat)::Matrix{Float32}

    vec_avrgs    = vec(zeros(size(comb_t)))
    (dima, dimb) = size(comb_t)
    comb_t       = vec(comb_t)
    index        = 1

   for jj in comb_t
      #  
      @inbounds vec_avrgs[index] = @views mean(fcasts_mat[jj[1], jj[2]])
      index += 1

    end

    return Matrix{Float32}(reshape(vec_avrgs, dima, :))

  end

#-------------------------------------------------------------------------		
#   Function to compute squared forecast error of univariate combinations
#-------------------------------------------------------------------------	
 function comb_sqf(comb_t, fcasts_mat, y_true, hist_mean)

    # Convert to Float32 to use less memory 
      fcasts_mat    = Float32.(fcasts_mat)
      y_true        = Float32.(y_true)
      hist_mean     = Float32.(hist_mean)

    # Pre-allocate 
      sq_err        = vec(Float32.(zeros(size(comb_t))))
      sq_err_shrunk = vec(Float32.(zeros(size(comb_t))))
      (dima, dimb)  = size(comb_t)
      comb_t        = vec(comb_t)
      index         = 1

   for jj in comb_t

    # Compute forecast
      fcast_temp        = @views mean(fcasts_mat[jj[1], jj[2]])
      fcast_shrunk_temp = @views (fcast_temp + hist_mean[jj[1][1]])/2

    # Compute and save squared forecast error 
      sq_err[index]        = @views (fcast_temp        - y_true[jj[1][1]])^2
      sq_err_shrunk[index] = @views (fcast_shrunk_temp - y_true[jj[1][1]])^2
      index += 1

    end
    
    return  push!(Matrix{Float32}[], reshape(sq_err, dima, :), reshape(sq_err_shrunk, dima, :))
  
  end







