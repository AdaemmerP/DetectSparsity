
#=
The following function is used to prepare the FRED-MD macroeconomic data set. 
The code is based on the matlab code by Michael W. McCracken that can be found here:
https://research.stlouisfed.org/econ/mccracken/fred-databases/   

=#

# Transformation function
  function transform_f(data)
    
    tcode = data[1]
    data  = data[2:end]
        
    # 1. No transformation
    if tcode == 1.0
      
      output = data
      
      # 2. First differences
    elseif tcode == 2.0
      
      output =  data - lag(data, 1)
      
      # 3. Second differences
    elseif tcode == 3.0
      
      output  =  (data - lag(data, 1)) - (lag(data, 1) - lag(data, 2))
      
      # 4. Natural Log
    elseif tcode == 4.0
      
      output = log.(data)
      
      # 5. First differences of natural log
    elseif tcode == 5.0
      
      output = log.(data) - lag(log.(data), 1)
      
      # 6. Second differences of natural log
    elseif tcode == 6.0
      
      output = (log.(data) - lag(log.(data), 1)) - (lag(log.(data), 1) - lag(log.(data), 2))
      
      # 7. First differences of percent change
    elseif tcode == 7.0 
      
      output = (data./(lag(data, 1)) .- 1)  -  (lag(data, 1)./(lag(data, 2)) .- 1)
   
    end  
        
    return vcat(tcode, output)
 
end

# Load original macroeconomic data set
  coltypes    = Any[Float64 for i= 1:129] 
  coltypes[1] = String
  Xall        = CSV.read(string(data_path, "2021-02.csv"), 
                         types = coltypes, DataFrame) 

# Remove missing observations                            
  Xall      = Xall[:, map(ii -> sum(ismissing.(ii)) == 0, eachcol(Xall[:, 1:end]))]
  Xall      = hcat(DataFrame(date = Xall.date), mapcols(transform_f, Xall[:, 2:end])) 
  Xall      = dropmissing(Xall)[2:end, :]
  Xall.date = Date.(Xall.date, "mm/dd/yyyy")

# Compute lead of INDPRO
  Xall[!, :INDPRO_lead] = lead(Xall[:, :INDPRO])
 
# Change order of DataFrame
  Xall = hcat(Xall.date, hcat(Xall[:, r"INDPRO"], Xall[:, Not(r"INDPRO")]))[:, Not(:date)]
  rename!(Xall, ["date"; names(Xall)[2:end]])

# Save DataFrame   
#  CSV.write(string(data_path, "MacroData_INDPRO_prepared.csv"), Xall)
