You need Julia >=  1.7.0
You also need R >= 4.0.4. 

Save the folder 'CaseStudies' on your computer and open the folder within Julia. 
Activate the environment and install the necessary packages by typing the following in the REPL:

	1. ]
	2. activate .
	3. instantiate
	4. Strg C

Further information on activating an environment can be found here:
https://pkgdocs.julialang.org/v1/environments/

Only "Main_Script.jl" is necessary to replicate the results. Before running it, 
set the following values:

Line 28: Path to the folder 'Data'
Line 31: Path to save the results on your computer (csv-files)
Line 32: Set to 'false' if the results shall not be saved
Line 38: Set the number of workers for parallelization
Be careful to activate the workers only once! 
To close the workers write "rmprocs(workers())" (see also line 122 of "Main_Script.jl")
Line 61: Choose the dataset. Note that the Bayesian algorithm takes the longest time to run. To increase speed,
lower the values in lines 94 and 95. If lowered, however, the results are not identical to those of the paper 
Line 65: Decide whether to in- or exclude the Covid period. Only relevant for the macroeconomic data set. 

Run 'include("Main_Script.jl")' in the REPL and wait. 
Running the code takes some time, especially for the macroeconomic data.  
