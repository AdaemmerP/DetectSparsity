# ----------------------------------------------------------------	
#               Function to simulate data
# ----------------------------------------------------------------	
function data_simul(n::Int64,
                    β::Vector{Float64},
                    Σ::Matrix{Float64},
                    μ_x::Vector{Float64},
                    ω::Float64,			
                    β_active::Vector{Int64},					
                    ϕ::Vector{Float64},
                    err_type::Int64,
                    ν::Float64,
                    seed_it::Int64)

  # Set seed for reproducibility 
    Random.seed!(seed_it)

  # Draw autoregressive x variables 
    dx     			 = MvNormal(μ_x, Σ)	
    x_temp 			 = zeros(length(μ_x), n) 
    x_temp[:, 1] = rand(dx, 1)

  # Draw values	
    for ii = 2:n
      x_temp[:, ii] = (ϕ.*x_temp[:, (ii - 1)])' .+ rand(dx, 1)'			
    end

  # Transpose
    x_temp = x_temp'

  # Draw nonzero β    
    β_temp            = zeros(length(β))
    β_temp[β_active] .= 1
    β_temp            = β_temp.*β
    
  # β_temp .= 0  # For simulating with zero signal

  # Draw noise (normal or t-distributed)	
    if err_type == 0	
      sd_x   = sqrt(β_temp'*Σ*β_temp) 
      dη     = Normal(0, ω*sd_x) 
      η_temp = rand(dη, n)
    
    elseif err_type == 1	
      dη     = TDist(ν)
      η_temp = rand(dη, n)*sqrt(ω^2*β_temp'*Σ*β_temp*((ν-2)/ν)) 
      
    end

  # Simulate y variable	
    y_temp = x_temp*β_temp + η_temp		

    return [y_temp x_temp ]  

 end

# ----------------------------------------------------------------	
#         Function for univariate OLS forecasts
# ----------------------------------------------------------------	
 function ols_forecasts(xtrain::Array{Float64}, 
                        ytrain::Array{Float64}, 
                        xtest::Union{Float64, Array{Float64}})

  if mean(xtrain) != 1.0								

        # Add constant to regression model								
        xtrain = hcat(ones(size(xtrain, 1)), xtrain)

        # Compute coefficients 	
        βhat 	 = GLM.lm(xtrain, ytrain) 						
        yhat   = GLM.predict(βhat, [1.0 xtest'])[1] 

    else

        βhat 	 = xtrain\ytrain         
        yhat   = (βhat*xtest)[1] 

    end

    return yhat

 end
# -----------------------------------------------------------------------------

# ------------------------------------------------------------------------	
# 	Function to compute squared forecast error of univariate combinations
# -----------------------------------------------------------------------	
  function comb_sqf(comb_t, fcasts_mat, y_lead, hist_mean)

    # Convert to Float32 for memory 
      fcasts_mat    = Float32.(fcasts_mat)
      y_lead        = Float32.(y_lead)
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
      sq_err[index]        = @views (fcast_temp        - y_lead[jj[1][1]])^2
      sq_err_shrunk[index] = @views (fcast_shrunk_temp - y_lead[jj[1][1]])^2
      index += 1

    end
    
    return  push!(Matrix{Float32}[], reshape(sq_err, dima, :), reshape(sq_err_shrunk, dima, :))

  end

# ----------------------------------------------------------------	
#       Function for combination method (time series cv)
# ----------------------------------------------------------------	
function fcomb_simul(xy_mat, sample_train,
                     model_comb, models_univ_time, comb_t,
                     β_active)::Vector{Float64} 
										
    # Build matrices
      y_temp = @views xy_mat[:, 1]
      x_temp = @views xy_mat[:, 2:end]

    # Compute historical mean	(same as regression with constant only)
      hist_mean = @views map(i -> mean(y_temp[i]), sample_train)
      
    # Compute all individual forecasts	
      fcast_univ::Matrix{Float64} = map(i -> ols_forecasts(x_temp[i[1], i[2]], 
                                        y_temp[i[1]], 
                                        x_temp[last(i[1]) + 1, i[2]]), models_univ_time)

    # Add historical mean and shrink individual models towards the historical mean (model 2) 																													 
      fcast_univ = hcat(hist_mean, fcast_univ)        	
    
      # Compute squared errors of individually unshrunk models 
        ϵ_sq_all     = @views comb_sqf(comb_t, fcast_univ, y_temp[last.(sample_train) .+ 1], hist_mean)
        ϵ_sq         = @views ϵ_sq_all[1] 
        ϵ_sq_shrunk  = @views ϵ_sq_all[2] 

    # Pick model with lowest MSE until t-1
      index_pick::Int64  = argmin(mean(ϵ_sq[1:(end - 1), :], dims = 1))[2]
      ew_index 					 = Int64(findall(i -> i == 2:(size(x_temp, 2) + 1), model_comb)[1])

    # Get chosen predictors and compute true positives  
      pred_pick = model_comb[index_pick] .- 1 # '-1' because intercept is first predictor
      truep     = length(intersect(pred_pick, β_active))/length(β_active)

    # Return squared errors and length of chosen parameters
      return 	[ϵ_sq[end, index_pick]; 
               ϵ_sq[end, ew_index];
               length(model_comb[index_pick]); 
               ϵ_sq_shrunk[end, index_pick];           
               ϵ_sq_shrunk[end, ew_index];
               truep] 
	end

# -------------------------------------------------------------------------	
#             Glmnet forecasts with time series cross validation 
# -------------------------------------------------------------------------	
function glmnet_simul_cvts(y_temp::Vector{Float64}, 
                           x_temp::Matrix{Float64}, 
                           q0::Int64, 
                           τ0::Int64,
                           α::Float64,
                           β_active)::Vector{Float64} 

  # Iterator for training samples to choose λ	and iterators for oos cv
    sample_train      = map(i -> 1:(q0 +  i), 0:(τ0 - 1)) 	
    sample_test       = map(i -> (last(i) + 1), sample_train)
    s_train_test      = hcat(sample_train, sample_test)

  # Get initial sequence of lambda values
    λ_init = @views glmnet(x_temp[1:q0, :], 
                           y_temp[1:q0], 
                           alpha        = α, 
                           standardize  = true, 
                           intercept    = true).lambda

  # Create Matrix for cross validation forecasts and get sequence for cv sample										 
    cv_fcast  = zeros(length(λ_init), τ0)											 

      # Loop for cross validation		
        for jj = 1:size(s_train_test, 1)

            obs_train::UnitRange{Int64} = @views s_train_test[jj, 1]
            obs_test::Int64						  = @views s_train_test[jj, 2]
            xtest::Matrix{Float64}      = reshape(x_temp[obs_test, :], 1, :)

            # Compute path with initialized λ values	
            fit_glmnet = @views glmnet(x_temp[obs_train, :], 
                                       y_temp[obs_train], 
                                       alpha       = α, 
                                       standardize = true, 
                                       intercept   = true,
                                       lambda      = λ_init)
                                    
            # Make forecast											 
            predict_cv::Matrix{Float64} = GLMNet.predict(fit_glmnet, xtest)											 

            # Save forecast	
            cv_fcast[:, jj] = predict_cv

        end

  # Compute squared error	for each λ_init 
    ϵ_sq = @views (cv_fcast' .- y_temp[s_train_test[:, 2]]).^2

  # Compute MSE 	
    recurs_mean = mean(ϵ_sq, dims = 1)
    λopt        = argmin(recurs_mean)[2]

  # Train final model with chosen λ	
    fit_glmnet = @views glmnet(x_temp[1:(end - 1), :], 
                               y_temp[1:(end - 1)], 
                               alpha       = α, 
                               standardize = true, 
                               intercept   = true,
                               lambda      = λ_init) 

  # Make final oos prediction	
    fc_glm        = @views GLMNet.predict(fit_glmnet, reshape(x_temp[end, :], 1, :))[1, λopt]
    fc_glm_shrunk = @views (fc_glm + mean(y_temp[1:(end - 1)]))/2

  # Get chosen predictors and compute true positives
    pred_pick = collect(1:size(x_temp, 2))[(fit_glmnet.betas[:, λopt] .!= 0.0)]
    truep     = length(intersect(pred_pick, β_active))/length(β_active)
    
  # Return values
    return [(fc_glm - y_temp[end])^2; (fc_glm_shrunk - y_temp[end])^2; Float64(length(pred_pick)); truep] 
 
end

# -------------------------------------------------------------------------	
#                       Function for Adaptive Lasso
# -------------------------------------------------------------------------	
function adaptive_lasso_cvts(y_temp::Vector{Float64}, 
                             x_temp::Matrix{Float64}, 
                             q0::Int64, 
                             τ0::Int64,
                             β_active)

  # Iterator for training and test samples 
    sample_train      = map(i -> 1:(q0 +  i), 0:(τ0 - 1)) 	
    sample_test       = map(i -> (last(i) + 1), sample_train)

  # Compute Ridge model for weights 
    fit_glmnet = @views glmnet(x_temp[1:q0, :], 
                                y_temp[1:q0], 
                                alpha        = 0, 
                                standardize  = true, 
                                intercept    = true)

  # Get lambda sequence, sample size,  etc.
    n   	     = q0 
    λ_seq      = fit_glmnet.lambda				
    bic_vec    = zeros(length(λ_seq), 1)	
    Xmat       = hcat(ones(size(x_temp[1:q0, :], 1)), x_temp[1:q0, :])
    ridge_pred = GLMNet.predict(fit_glmnet, x_temp[1:q0, :]) 

  # Loop to compute BIC 
  # For computation see: https://hastie.su.domains/TALKS/enet_talk.pdf and
  # https://rdrr.io/cran/lmridge/man/infocr.html
    for jj = 1:length(λ_seq)				

      Σϵ²   = sum((y_temp[1:q0] - ridge_pred[:, jj]).^2)
      Id    = λ_seq[jj]*I(size(x_temp, 2) + 1) # Plus one because of constant
      H     = Xmat*inv(Xmat'*Xmat + Id)*Xmat'
      df    = tr(H)

    # Compute BIC
      bic_vec[jj] = n*log(Σϵ²) + log(n)*df    
    end

    coef_ridge =  fit_glmnet.betas[:, argmin(bic_vec)[1]]

  # Get initial sequence of lambda values (using weights for adaptive lasso)
    λ_init = @views glmnet(x_temp[1:q0, :], 
                           y_temp[1:q0], 
                           alpha          = 1, 
                           standardize    = true, 
                           penalty_factor = 1 ./ abs.(coef_ridge),
                           intercept      = true).lambda

  # Create Matrix for cross validation forecasts and get sequence for cv sample										 
    cv_fcast  = zeros(length(λ_init), τ0)											 
    
      # Loop for cross validation				
        for jj = 1:τ0

          # Training sample sequence and test observation
            train_seq = 1:(sample_test[jj] - 1)
            xtest     = reshape(x_temp[sample_test[jj], :], 1, :)

          # Compute path with initialized λ values	
            fit_glmnet = @views glmnet(x_temp[train_seq, :], 
                                      y_temp[train_seq], 
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
    ϵ_sq = (cv_fcast' .- y_temp[sample_test]).^2

  # Compute MSE 	
    mse_λ  = mean(ϵ_sq, dims = 1)
    λopt	 = argmin(mse_λ)[2]

  # Train final model with chosen λ	
    fit_glmnet = @views glmnet(x_temp[1:(end - 1), :], 
                               y_temp[1:(end - 1)], 
                               alpha          = 1, 
                               standardize    = true, 
                               intercept      = true,
                               penalty_factor = 1 ./ abs.(coef_ridge),
                               lambda         = λ_init) 

  # Make final forecast	
    fc_adlasso        = @views GLMNet.predict(fit_glmnet, reshape(x_temp[end, :], 1, :))[1, λopt]
    fc_adlasso_shrunk = @views (fc_adlasso + mean(y_temp[1:(end - 1)]))/2
  
  # Get chosen predictors and compute true positives
    pred_pick = collect(1:size(x_temp, 2))[(fit_glmnet.betas[:, λopt] .!= 0.0)]
    truep     = length(intersect(pred_pick, β_active))/length(β_active)

  # Return values
    return [(fc_adlasso - y_temp[end])^2; (fc_adlasso_shrunk - y_temp[end])^2; Float64(length(pred_pick)); truep] 

end


#-------------------------------------------------------------------------	
# 		          Function for relaxed Lasso
# -------------------------------------------------------------------------	
# Info: This function uses the pure Julia implementation of the coordinate 
#       descent algorithm because the estimated parameters can be modified 
#       much more easily than in 'GLMNet.jl'. Yet the function uses GLMNet.jl to wrap the original
#       Fortran code to compute the initial values of λ

  function glmnet_relaxed_cvts(y_temp, x_temp,  
                               q0, τ0, α, ζ,
                               β_active)

      # Iterator for training samples and validation sample
        sample_train = map(i -> 1:(q0 +  i), 0:(τ0 - 1)) 
        sample_test  = map(i -> (last(i) + 1), sample_train)

      # Get initial sequence of lambda values with original Fortran code
        λ_init    = @views glmnet(x_temp[sample_train[1], :], 
                                  y_temp[sample_train[1]], 
                                  alpha        = α, 
                                  standardize  = true, 
                                  intercept    = true).lambda 
        
       # Vectors to store validation results					 
         cv_fcast  = zeros(length(λ_init), τ0, length(ζ))											           

        # ---------------------- Begin validation loop ---------------------------- # 
        # Loop for cross validation		
          for jj = 1:length(sample_train)

              # Training sample sequence and test observation
                train_seq = @views sample_train[jj]
                xtest     = reshape(x_temp[sample_test[jj], :], 1, :)

              # Compute initial path  
                fitLStart  = Lasso.fit(LassoPath, 
                                       x_temp[train_seq, :], 
                                       y_temp[train_seq], 
                                       standardize = true,                                
                                       intercept   = true,
                                       λ = λ_init; 
                                       α = α) 

              # Get active OLS sets 
                ASets = map(i -> i .!= 0.0, eachcol(fitLStart.coefs))  

              # Pre-allocate Matrix for OLS coefficients
                LS_β  = zeros(size(x_temp, 2), length(ASets)) 
                LS_β0 = zeros(length(ASets))

                # Loop to compute OLS models  
                # Account for the intercepts 
                  for kk = 1:length(ASets)

                    if sum(ASets[kk]) == 0

                    # Intercept only 
                      LS_β0[kk] = mean(y_temp[train_seq])

                    else

                    # Make OLS predictions (Coefficients in Lasso.jl are not standardized)
                      xglm                = @views hcat(ones(length(train_seq)), x_temp[train_seq, ASets[kk]])
                      βcoefs              = GLM.coef(lm(xglm, y_temp[train_seq]))
                      LS_β0[kk]           = @views βcoefs[1]
                      LS_β[ASets[kk], kk] = @views βcoefs[2:end] 
                                
                    end

                  end

                # Get validation oata  
                  xtest_tmp  = x_temp[sample_test[jj], :] 
                  ytest_temp = y_temp[sample_test[jj]]

                # Make validation forecasts for relaxed Lasso    
                  for ll = 1:length(ζ)

                      fittemp             = deepcopy(fitLStart)
                      fittemp.b0          = ζ[ll].*fittemp.b0    + (1 - ζ[ll]).*LS_β0
                      fittemp.coefs       = ζ[ll].*fittemp.coefs + (1 - ζ[ll]).*LS_β            
                      cv_fcast[:, jj, ll] = (Lasso.predict(fittemp, xtest_tmp') .- ytest_temp).^2

                  end
                 
            end

        # ---------------------- End validation loop ---------------------------- #     

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

        # Train final model with chosen λ	and ζ 
          xtrainf = x_temp[1:last(sample_test), :]
          ytrainf = y_temp[1:last(sample_test)]     

        # Compute final model   
          fit_path = Lasso.fit(LassoPath,
                              xtrainf, ytrainf,                         
                              standardize = true, 
                              intercept   = true,                          
                              λ           = λ_init;
                              α           = α) 

        # Check whether coefficients have to be relaxed 
          if ζ[ζopt] == 1.0 # no relaxation
            
            # Final forecast with unrelaxed coefficients
             fcast = Lasso.predict(fit_path, reshape(x_temp[(last(sample_test) + 1), :], 1, :))[1, λopt] 

          else 
                # Pre-allocate for relaxed coefficients     
                  LS_β    = zeros(size(xtrainf, 2))      
                  Afinal  = Vector(fit_path.coefs[:, λopt] .!= 0)

                  # Check whether any parameter will be added  
                  if sum(Afinal) == 0

                    # Constant only model	
                      LSb0        = mean(ytrainf)
                   
                  else

                    # Compute OLS coefficients
                      xglm          = @views hcat(ones(size(xtrainf, 1), 1), xtrainf[:, Afinal]) 
                      βcoefs        = GLM.coef(lm(xglm, y_temp[1:(last(sample_test))]))
                      LSb0          = βcoefs[1] 
                      LS_β[Afinal]  = βcoefs[2:end] 

                   end

              # Final forecast      
                fit_path.coefs .= ζ[ζopt].*fit_path.coefs .+ (1 -  ζ[ζopt]).*LS_β
                fit_path.b0    .= ζ[ζopt].*fit_path.b0 .+ (1 -  ζ[ζopt]).*LSb0

            # Final forecast with relaxed coefficients
              fcast = Lasso.predict(fit_path, reshape(x_temp[(last(sample_test) + 1), :], 1, :))[1, λopt] 

            end
         
        # Save final forecast	         
          fcast_shrunk = (fcast + mean(ytrainf))/2         
          ζpick        = @views ζ[ζopt]
    
       # Get chosen predictors and compute true positives
         pred_pick = collect(1:size(x_temp, 2))[(fit_path.coefs[:, λopt] .!= 0.0)]
         truep     = length(intersect(pred_pick, β_active))/length(β_active)
           
    # Return vector with forecasts and (number of) nonzero predictors
      return [(fcast - y_temp[end])^2; (fcast_shrunk - y_temp[end])^2; Float64(length(pred_pick)); ζpick; truep]

end



# ----------------------------------------------------------------	
#             Function for OOS forecasts with GLP approach
# ----------------------------------------------------------------	
 function	GLP_oos(xtrain::Matrix{Float64}, xtest::Vector{Float64}, 
                  ytrain::Vector{Float64}, ytest::Float64, 
                  M::Int64, N::Int64, 
                  abeta::Float64, bbeta::Float64, 
                  Abeta::Float64, Bbeta::Float64,
                  iter::Int64,
                  β_active::Vector{Int64}) 

  # Set seed for reproducibility 
    Random.seed!(iter)

  # Make vector for ones				
    u::Array{Float64} = repeat([1.0], size(xtrain, 1), 1)

  # Standardize (x) observations and save mean and standard deviation
    μ_x    = reshape(mean(xtrain, dims = 1), :, 1)
    σ_x    = reshape(std(xtrain,  dims = 1), :, 1)
    xtrain = standardize(ZScoreTransform, xtrain, dims = 1) 

  # Compute GLP model											
    results_temp = GLP_SpikeSlab(ytrain, xtrain, u, abeta, bbeta, Abeta, Bbeta, M)

  # Compute slope coefficients as mean of all draws			      										
    βhat::Matrix{Float64} = @views mean(results_temp[1][:, (N + 1):end], dims = 2) 

  # Unstandardize intercept	
    β0::Float64 = @views mean(results_temp[3][(N + 1):end]) 

  # Standardise test observations for prediction	
    xtest = (xtest .- μ_x)./σ_x

  # OOS prediction
    yhat::Float64 = dot([1; xtest], [β0; βhat]) 
    yhat_shrunk   = (yhat + mean(ytrain))/2

  # Average inclusion of preditors per Gibbs iteration
    z_run::Float64 = mean(map(i -> sum(i), eachcol(results_temp[2][:, (N + 1):end])))

  # Get active predictors and compute true positive rate
    pred_pick::Vector{Vector{Int64}} = map(i -> collect(1:length(βhat))[Bool.(i)], eachcol(results_temp[2][:, (N + 1):end]))  
    truep::Float64 = mean(length.(map(i -> intersect(i, β_active), pred_pick))./length(β_active)) # 

  # Compute average q  
  #  mean_q::Float64 = mean(results_temp[4][(N + 1):end])*length(βhat)
    
  return [(yhat - ytest)^2; (yhat_shrunk - ytest)^2; z_run; truep] 

end
# -------------------------------------------------------------------------	


# -------------------------------------------------------------------------	
# 		              Forecast function for BSS
# -------------------------------------------------------------------------	
# Load necessary BeSS-package (R)
	R"""
	library(BeSS)
	library(parallel)
	"""

 # Start function 
  function bkm_predict(xy_data::Matrix{Float64}, 
                      train_seq::Vector{Int64},
                      obs_test::Int64,
                      β_active)    

    @rput xy_data train_seq obs_test 

    R"""
      # Train models (supress output)
        bess_results <- bess(xy_data[train_seq, 2:dim(xy_data)[2]], 
                              xy_data[train_seq, 1],
                              method  = "sequential",
                              epsilon = 0, 
                              family  = "gaussian")		

        fc_bkm_ebic <- predict(bess_results, xy_data[obs_test, 2:dim(xy_data)[2], drop = F], type = "EBIC") 
      
      # Count number predictors	
        pred_pick   <- (coef(bess_results, sparse = F)[, which.min(bess_results$EBIC)] != 0)
        pred_pick   <- pred_pick[2:length(pred_pick)] # Start at 2 to exclude the intercept
        nz_bkm_ebic <- sum(pred_pick) 
        pred_pick   <- (1:(dim(xy_data)[2] - 1))[pred_pick] # '-1' because first column is 'y'

    """

      # Get results       
        fc_bkm_ebic = @rget fc_bkm_ebic
        nz_bkm_ebic::Float64 = @rget nz_bkm_ebic
        pred_pick   = @rget pred_pick     

      # Shrink forecasts towards historical mean    
        fc_bkm_ebic_shrunk = (fc_bkm_ebic + mean(xy_data[train_seq, 1]))/2  

      # Compute squared errors 			
        sqerr_ebic::Float64 = (xy_data[end, 1] - fc_bkm_ebic).^2
        sqerr_ebic_shrunk::Float64 = (xy_data[end, 1] - fc_bkm_ebic_shrunk).^2

      # Compute true positives
        truep  = length(intersect(pred_pick, β_active))/length(β_active)

      # Return results  
      return [sqerr_ebic; nz_bkm_ebic; sqerr_ebic_shrunk; truep]	


  end
