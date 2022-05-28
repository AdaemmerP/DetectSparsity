You need at least Julia 1.7.0
You also need R >= 4.0.4. 

Save the folder 'CaseStudies' on your computer and open the folder within Julia. 
Activate the environment and install the necessary packages by typing the following in the REPL:

	1. ]
	2. activate .
	3. instantiate
	4. Strg C

Further information on activating an environment can be found here:
https://pkgdocs.julialang.org/v1/environments/

Only "Main_Script.jl" is relevant for running the simulations. 
Set the following values in the script:

Line 6:  Path to to the folder 'Data'
Line 9:  Number of workers
Line 10: Number of Monte Carlo iterations 
Line 11: Choose dataset (0 = financial, 1 = macroeconomic data)
Line 12: Error type (paper only reports results for t-distributed errors)

To increase the speed, lower the number of iterations for Bayesian FSS (lines 16 & 17)
Run 'include("Main_Script.jl")' in the REPL and wait. 

Close workers with rmprocs(workers())

"Main_Script_2.jl" is for replicating the results of Figure 5


