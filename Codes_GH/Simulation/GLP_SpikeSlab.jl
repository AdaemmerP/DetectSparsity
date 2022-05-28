function GLP_SpikeSlab(y::Array{Float64}, 
                       x::Matrix{Float64}, 
                       u::Array{Float64}, 
                       abeta::Float64, 
                       bbeta::Float64, 
                       Abeta::Float64, 
                       Bbeta::Float64, 
                       M::Int64)         

#-----------------------------------------------------------------------------------------------------------------------#
# Code by Philipp Adämmer & Rainer Schüssler
#
# Based on Matlab code by Domenico Giannone; Michele Lenza & Giorgio Primiceri:  
#  "Economic Predictions with Big Data: the Illusion of Sparsity;"
#   https://www.econometricsociety.org/publications/econometrica/2021/09/01/economic-predictions-big-data-illusion-sparsity
#   
# Inputs:
#   y: T x 1 vector of observations of the response variable
#   x: T x k (standardized )matrix of observations of the predictors 
#   u: T x 1 vector of ones for constant
#   abeta and bbeta: parameters of beta prior on q 
#   Abeta and Bbeta: parameters of beta prior on R2 
#   M:               total number of draws;
#
# Output:
#   store_B:   posterior draws of regression coefficients
#   store_z:   draws of 0-1 variables indicating if a predictor is included
#   store_phi: draws of regression coefficients of predictors always included
#   store_q:   draws of probability of inclusion
#   store_R2:  draws of R squared
#   store_gam: draws of prior standard deviation conditional on inclusion
#   store_s2:  draws of residual variance
#
# Reference:
#   Giannone, Domenico, Michele Lenza & Giorgio E. Primiceri [2021] 
#   "Economic Predictions with Big Data: the Illusion of Sparsity;"
#   Econometrica; forthcoming.
#-----------------------------------------------------------------------------------------------------------------------#
        
# Pre-assign
  l = Int(1) 

# Determine sizes
  sizeX           = size(x)
  T::Int64        = (sizeX[1])
  k::Int64        = (sizeX[2])
  varx::Float64   = var(x[:, 1], corrected = false)  

# Edges of grids for q & R2
  edgesq  = vcat(collect(0:.001:.1), collect(.11:.01:.9), collect(.901:.001:1))
  edgesR2 = vcat(collect(0:.001:.1), collect(.11:.01:.9), collect(.901:.001:1))

# storage & preliminary calculations
  store_z     =   zeros(k, M)
  store_phi   =   zeros(l, M)
  store_B     =   zeros(k, M)
  store_s2    =   zeros(M)
  store_gam   =   zeros(M)
  store_q     =   zeros(M)
  store_R2    =   zeros(M)

  areaq      =   (edgesq[2:end]   - edgesq[1:end - 1])
  areaR2     =   (edgesR2[2:end]  - edgesR2[1:end - 1])
  areaqR2    =   (repeat(areaq, 1, length(areaR2)).*repeat(areaR2', length(areaq), 1))
  intq       =   (edgesq[1:(end - 1)] + areaq/2)
  intR2      =   (edgesR2[1:end - 1]  + areaR2/2)
  INTQ       =   (repeat(intq', length(intR2), 1))
  INTR2      =   (repeat(intR2, 1, length(intq)))

# OLS results		
  xx::Matrix{Float64} = (x'*x); 
  xy::Vector{Float64} = (x'*y); 
  yx::Matrix{Float64} = (y'x);
  yy::Float64         = (y'*y)

# Intercept 		 
  if  l == 1 
      xu           = x'*u; 
      yu::Float64  = dot(y, u); 
      uu::Float64  = dot(u, u); 
      invuu        = Float64(1/uu[1]); 
      cholinvuu 	 = Float64(sqrt(invuu)) 
      ux           = u'*x; 
      invuuuy      = invuu*u'*y; 
      invuuux      = invuu*u'*x
  elseif l == 0 
      xu = zeros(k, 1); yu = zeros(1, 1); uu = zeros(1, 1); ux = zeros(1, k); phi = 0; 
  end

# Descritize support to draw from R² and q
  QR2         = INTQ.*(ones(Float64,  size(INTR2, 1), size(INTR2, 1)) - INTR2)./INTR2
  prodINT     = zeros(size(INTQ, 1),  size(INTQ,  1), k + 1)
  onesINTQ    = ones(size(INTQ, 1),   size(INTQ,  1))
  onesINTR2   = ones(size(INTQ, 1),   size(INTQ,  1))

  for ttt = 0:k
      prodINT[:, :, ttt + 1] = INTQ.^(ttt + ttt/2 + abeta - 1).*(onesINTQ - INTQ).^(k - ttt + bbeta - 1).*
                              INTR2.^(Abeta - 1 - ttt/2).*(onesINTR2 - INTR2).^(ttt/2 + Bbeta - 1).*areaqR2
  end

# Pre-define vectors etc. to store data
  z  = BitVector(zeros(size(x, 2)))
  z0 = BitVector(zeros(size(x, 2)))
  z1 = BitVector(zeros(size(x, 2)))
  b  = zeros(Float64, size(x, 2), 1)        

# starting values (based on Lasso under the assumption that the x's explain 50% of var(resy)) (see original code by GLP)
  if l > 0; phi0  = invuuuy; resy = y - u*phi0; store_phi[:,1] = phi0; else; resy = y; end
  b0::Matrix{Float64}   = glmnet(x, y, lambda = [sqrt(8*k*varx*var(resy, corrected = false)/2)/(2*length(resy))], 
                                 standardize = false, intercept = true).betas	|> Matrix		

  store_B[:, 1]       = b0; 
  b::Matrix{Float64}  = (reshape(b0, :, 1))
  s2::Float64         = (sum((resy - x*b).^2)/T)
  store_s2[1]         = s2
  z[:]                = b.!= 0
  store_z[:, 1]       = z
  tau                 = sum(z)

#--------------------------------------------------------------#		
#-----------------  Gibbs Sampler  ----------------------------#
#--------------------------------------------------------------#		

for i = 2:M
                
  # Block I: draw R2 & q  
    pr_qR2       =   @views (exp.(-(0.5*k*varx*dot(b, b)/s2).*QR2).*prodINT[:, :, tau + 1])    
    cdf_qR2      =   @views cumsum(pr_qR2[:]./sum(pr_qR2))
    aux          =   sum(cdf_qR2 .< rand()) + 1
    q            =   INTQ[aux]
    store_q[i]   =   q
    R2           =   INTR2[aux]
    store_R2[i]  =   R2
    gam          =   sqrt((1/(k*varx*q))*R2/(1 - R2))
    store_gam[i] =   gam
  
    #  Block II: draw phi
    if l > 0
        phihat          = @views invuuuy - invuuux*store_B[:, i - 1]
        phi             = (randn()*sqrt(s2)*cholinvuu)' .+ phihat 
        store_phi[:, i] = phi 																		 
    end
  
  # Block III: draw z
    for j = 1:k
        z0[:]   =   z[:]; 			
        z0[j]   =   0
        logicz0 =   z0 .!= 0 				
        z1[:]   =   z[:]; 
        z1[j]   =   1
        logicz1 =   z1 .!= 0
        tau0    =   sum(z0)
        tau1    =   sum(z1)
        W0      =   @views xx[logicz0, logicz0] + I(tau0)/gam^2
        W1      =   @views xx[logicz1, logicz1] + I(tau1)/gam^2
        bhat0   =   @views W0\(xy[logicz0]      - xu[logicz0, :]*phi)
        bhat1   =   @views W1\(xy[logicz1]      - xu[logicz1, :]*phi)
        
      # Note: GLP use '2*sum(log(diag(chol(W0))))' instead of '.5*log(det(W0))'. 
        log_pr_z0   =  @views (tau0*log(q) + (k - tau0)*log(1 - q) - tau0*log(gam) - .5*log(det(W0)) .-
                                .5*T*log.(yy .- 2*yu*phi .+ phi'*uu*phi .- yx[logicz0]'*bhat0 .+ phi'*ux[ :, logicz0]*bhat0))[1]
        log_pr_z1   =  @views (tau1*log(q) + (k - tau1)*log(1 - q) - tau1*log(gam) - .5*log(det(W1)) .-
                              .5*T*log.(yy .- 2*yu*phi .+ phi'*uu*phi .- yx[logicz1]'*bhat1 .+ phi'*ux[:, logicz1]*bhat1))[1]
        
        z[j]        =   (rand() <= (1/(exp(log_pr_z0 - log_pr_z1) + 1)))

    end
  
# Store z	
  tau           =   sum(z);    
    
# Block III: draw s2 and β
  if tau == 0

      niu           = @views (yy .- 2*yu*phi .+ phi'*uu*phi)[1]
      s2            = (1/rand(Gamma(T/2, 2/niu)))
      store_s2[i]   = s2
      b             = zeros(Float64, k, 1)
      store_B[:, i] = b

  else
      W::Matrix{Float64} =   xx[z, z] + I(tau)/gam^2   
      bhat               =   W\(xy[z] - xu[z, :]*phi)
      store_z[:, i]      =   z
    
      niu          = @views (yy .- 2*yu*phi .+ phi'*uu*phi .- bhat'*W*bhat)[1]
      s2           = (1/rand(Gamma(T/2, 2/niu)))
      store_s2[i]  = s2
                  
    #	When tau = 1 
      if tau == 1 
          b          = (randn(1, tau)*cholesky(s2*I(tau)/W).U)' + bhat
      else
          b          = (randn(1, tau)*cholesky(Hermitian(s2*I(tau)/W)).U)' + bhat
      end
      
      store_B[z, i] = b;    
    
end

end

# Return values
  [store_B, store_z, store_phi, store_q, store_R2, store_gam, store_s2] 

end
