# SetIntersectionProjection
Julia software for computing projections onto intersections of convex and non-convex constraint sets.

[Documentation](https://petersbas.github.io/SetIntersectionProjectionDocs/)


## Installation for Julia 1.1:

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
	- Wavelets
	
- The examples also use the packages:
	- [MAT](https://github.com/JuliaIO/MAT.jl)
	- [PyPlot](https://github.com/JuliaPy/PyPlot.jl)
	- [StatsBase](https://github.com/JuliaStats/StatsBase.jl)
 
## Status:

###  Feb 25 2019

 - the V06 branch works with Julia 0.6 and is intended only for the waveform inversion example that uses [WAVEFORM.jl](https://github.com/slimgroup/WAVEFORM.jl)
 - one of the examples requires data that is not available online yet
 - master branch works with Julia 1.1

