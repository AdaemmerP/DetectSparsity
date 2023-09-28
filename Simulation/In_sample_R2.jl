# Compute in-sample SNR

# Simulation parameter
ncores = 3           # Number of cores (for workers) 	
N = Int64(1e2)  # Number of Monte Carlo iterations 
dataset = 1           # 0 = Financial data, 1 = Macroeconomic data (no lags), 2 = Macroeconomic data (including 4 lags)
err_type = 1           # 0 = normal errors,  1 = t-distributed errors 
diag_cov = false       # Use diagonal covariance matrix?
q0 = Int64(140)
τ0 = Int64(60)

# Set parameters for GLP code (N_glp = burnin sample)
N_glp = Int64(1e3)
M_glp = Int64(10e3) + N_glp

# Run script to load packages and prepare data (run only once (!))
include("PrepareData.jl")

# Set parameters	  
#μx    = repeat([0.0], size(x_mat, 2))
#βx    = GLM.coef(lm(hcat(repeat([1], size(x_mat, 1)), x_mat), y_GWP))[2:end] 

σ_y = lm(hcat(repeat([1], size(x_mat, 1)), x_mat), y_GWP) |> residuals |> var
r_2 = lm(hcat(repeat([1], size(x_mat, 1)), x_mat), y_GWP) |> r2
σ_x = βx' * cov(x_mat) * βx

snr = σ_x / σ_y

snr / (1 + snr) # equals R² : https://statproofbook.github.io/P/snr-rsq.html
r_2 / (1 - r_2) # equals snr : https://statproofbook.github.io/P/snr-rsq.html

ω = sqrt(1 / snr)

# Function to compute in-sample R²
function compute_r2(preds, x_mat, y_GWP)
    x_mat = x_mat[:, preds]
    r_2 = lm(hcat(repeat([1], size(x_mat, 1)), x_mat), y_GWP) |> r2
    snr = r_2 / (1 - r_2)
    ω = sqrt(1 / snr)
    return ω
end

# For macroeconmic data
preds = map(x -> sample(1:size(x_mat, 2), 100, replace=false), 1:10^4)
map(x -> compute_r2(x, x_mat, y_GWP), preds) |> mean


# For financial data
Combinatorics.combinations(1:size(x_mat, 2), 2) |> collect

fit_glmnet = glmnet(x_mat,
y_GWP,
    alpha=0,
    standardize=true,
    intercept=true)

sum(fit_glmnet.betas .== 0, dims = 1)    
test = fit_glmnet.betas
isapprox(fit_glmnet.betas, 0) 

cv = glmnetcv(x_mat,
y_GWP) 
argmin(cv.meanloss)
coef(cv) 
cv.path.betas[:, argmin(cv.meanloss)]