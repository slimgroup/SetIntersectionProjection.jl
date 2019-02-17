# SetIntersectionProjection
Julia software for computing projections onto intersections of constraint sets.

[Documentation](https://petersbas.github.io/SetIntersectionProjectionDocs/)


##Installation for Julia 1.1:

 - install [Julia Operators LIbrary (JOLI)](https://github.com/slimgroup/JOLI.jl)
 - then add SetIntersectionProjection:
 
 ```
 add https://github.com/slimgroup/SetIntersectionProjection.jl.git
 ``` 

 - SetIntersectionProjection also depends on the packages: 
 	- Parameters
	- Interpolations
	- DistributedArrays
	- SortingAlgorithms
	- FFTW
	
- The examples also use the packages:
	- [MAT](https://github.com/JuliaIO/MAT.jl), for Julia 1.1 use: 
		- pkg> rm MAT
		- add https://github.com/halleysfifthinc/MAT.jl#v0.7-update
	- [PyPlot](https://github.com/JuliaPy/PyPlot.jl)
 
## Status:

###  Feb 17 2019

 - the V06 branch works with Julia 0.6
 - one of the examples requires data that is not available online yet
 - master branch works with Julia 1.1
 - tests do not work at the moment, still being ported to 1.1

