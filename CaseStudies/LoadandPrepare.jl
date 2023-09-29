# Load data  
  if dataset == 0 # Only uses a subset for pre-compiling the function

    # For GW and Pyun data
      coltypes = Any[Float64 for i= 1:15] 
      Xall     =  CSV.read(string(data_path, "GW_Data.csv"), types = coltypes, DataFrame) 

    # Drop missings, transform dates and eq_prem and get subset									 
      Xall = @chain Xall begin
              @transform!(:date = string.(:date))
              @transform!(:date = map(x -> Date(Dates.Year(x[1:4]), Dates.Month(x[5:6])), :date))
              @subset(:date .>=  Date("2010-01-01"))
              end		
    
  elseif dataset == 1

    # For GW and Pyun data
      coltypes = Any[Float64 for i= 1:15] 
      Xall     =  CSV.read(string(data_path, "GW_Data.csv"),
                            types = coltypes, DataFrame) 

    # Drop missings, transform dates and eq_prem and get subset									 
      Xall = @chain Xall begin
              @transform!(:date = string.(:date))
              @transform!(:date = map(x -> Date(Dates.Year(x[1:4]), Dates.Month(x[5:6])), :date))
             end		
      
  elseif dataset == 2

    # For GW and Pyun data
      coltypes = Any[Float64 for i= 1:16] 
      Xall     =  CSV.read(string(data_path, "GWP_Data.csv"), types = coltypes, DataFrame) 

    # Transform dates 								 
      Xall = @chain Xall begin
                @transform!(:date = string.(:date))
                @transform!(:date = map(x -> Date(Dates.Year(x[1:4]), Dates.Month(x[5:6])), :date))             
            end		  

  elseif dataset == 3

      coltypes    = Any[Float64 for i= 1:117] 
      coltypes[1] = String
      Xall        = CSV.read(string(data_path, "MacroData_INDPRO_prepared.csv"),
                              types = coltypes, DataFrame) 
      Xall.date   = Date.(Xall.date, "yyyy-mm-dd")        
    
    # Compute correlation Matrix until 12-1969 
      corx       = cor(Matrix(Xall[1:findfirst(i -> i == Date("1969-11-01"), Xall.date),  Not(:date)] )) 
      corx       = abs.(UpperTriangular(corx) - I(size(corx, 2)))
      removecols = unique(getindex.(findall(i -> i .>= 0.95, corx), 2))  .+ 1 # Add one because :date is fist column in DataFrame 
      Xall       = Xall[:, Not(removecols)]

    # Remove 'Covid sample' 
      if covid_out == true
         Xall       = filter(row -> row.date <= Date("2019-12-01"), Xall)
      end

   # Add four lags of INDPRO   
   elseif dataset == 4

      coltypes    = Any[Float64 for i= 1:117] 
      coltypes[1] = String
      Xall        = CSV.read(string(data_path, "MacroData_INDPRO_prepared.csv"),
                              types = coltypes, DataFrame) 
      Xall.date   = Date.(Xall.date, "yyyy-mm-dd")   
      
      lags_indpro_df  = DataFrame(map(i -> lag(Xall[:, :INDPRO], i), 0:3),  :auto)
      rename!(lags_indpro_df,   map(i -> string("INDPRO_lag_", "$i"), 1:4))
    
     # Make DataFrame with lags
      Xall = hcat(Xall[:, 1:3], lags_indpro_df, Xall[:, 4:end])
      Xall = Xall[5:end, :] # Remove empty (lag) months
    
    # Compute correlation Matrix until 12-1969 
      corx       = cor(Matrix(Xall[1:findfirst(i -> i == Date("1969-11-01"), Xall.date), Not(:date)]))
      corx       = abs.(UpperTriangular(corx) - I(size(corx, 2)))
      removecols = unique(getindex.(findall(i -> i .>= 0.95, corx), 2))  .+ 1 # Add one because :date is first column in DataFrame 
      Xall       = Xall[:, Not(removecols[2:end])] # Start at second position because INDPRO and INPRO_lag are identical

    # Remove 'Covid sample' 
      if covid_out == true
         Xall       = filter(row -> row.date <= Date("2019-12-01"), Xall)
      end
  
    # Only use four lags of INDPRO   
    elseif dataset == 5

      coltypes    = Any[Float64 for i= 1:117] 
      coltypes[1] = String
      Xall        = CSV.read(string(data_path, "MacroData_INDPRO_prepared.csv"),
                              types = coltypes, DataFrame) 
      Xall.date   = Date.(Xall.date, "yyyy-mm-dd")   
      
      lags_indpro_df  = DataFrame(map(i -> lag(Xall[:, :INDPRO], i), 0:3),  :auto)
      rename!(lags_indpro_df,   map(i -> string("INDPRO_lag_", "$i"), 1:4))
      Xall = hcat(Xall[:, 1:3], lags_indpro_df, Xall[:, 4:end])
      Xall = Xall[:, r"date|INDPRO"]
      Xall = Xall[5:end, :] # Remove missing (lag) months     
    
    # Remove 'Covid sample' 
      if covid_out == true
         Xall       = filter(row -> row.date <= Date("2019-12-01"), Xall)
      end
      
  end    

# Exclude date from data matrix
  exclude_cols = [:date]

# Exclude variables and convert DataFrame to Matrix
  Xmat    = Xall[:, Not(exclude_cols)] |> Matrix{Float64} 
  y_lead  = Xmat[:, 2]
  x       = Xmat[:, 3:end]
