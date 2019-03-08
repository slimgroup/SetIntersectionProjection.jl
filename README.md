# SetIntersectionProjection (Julia 0.6 Branch)

Julia 0.6 software for computing projections onto intersections of convex and non-convex constraint sets.

[Documentation](https://petersbas.github.io/SetIntersectionProjectionDocs/)

The master branch of this repository is for Julia 1.1


## Installation for Julia 0.6:

 - install [Julia Operators LIbrary (JOLI)](https://github.com/slimgroup/JOLI.jl)
 - checkout the corresponding julia 06 branch: git checkout v06
 - then add SetIntersectionProjection from within Julia:
 
```
 Pkg.add("https://github.com/slimgroup/SetIntersectionProjection.jl.git")
```
 
 -  go to the julia package installation folder, e.g., /.julia/v0.6/SetIntersectionProjection and checkout the v06 branch: git checkout V06

 - SetIntersectionProjection also depends on the packages: 
 	- Parameters
	- Interpolations
	- DistributedArrays
	- SortingAlgorithms
	
- Download the WAVEFORM package:
```
Pkg.add("https://github.com/slimgroup/WAVEFORM.jl.git")
```

- The examples also use the packages:
	- [PyPlot](https://github.com/JuliaPy/PyPlot.jl)
 
## Status:

###  March 8 2019

 - the V06 branch works with Julia 0.6 and is intended only for the waveform inversion example that uses [WAVEFORM.jl](https://github.com/slimgroup/WAVEFORM.jl)
 - master branch works with Julia 1.1

