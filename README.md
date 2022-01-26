[![DOI](https://zenodo.org/badge/166865442.svg)](https://zenodo.org/badge/latestdoi/166865442)

# SetIntersectionProjection
Julia software for computing:
 - projections onto intersections of convex and non-convex constraint sets [Documentation](https://petersbas.github.io/SetIntersectionProjectionDocs/)
 - projections onto a generalization of the Minkowski set. [Documentation](https://petersbas.github.io/GeneralizedMinkowskiSetDocs/)

This software is based on the papers:

```bibtex
@article{peters2019algorithms,
  title={Algorithms and software for projections onto intersections of convex and non-convex sets with applications to inverse problems},
  author={Peters, Bas and Herrmann, Felix J},
  journal={arXiv preprint arXiv:1902.09699},
  year={2019}
}
```

and
```bibtex
@article{peters2019generalized,
  title={Generalized Minkowski sets for the regularization of inverse problems},
  author={Peters, Bas and Herrmann, Felix J},
  journal={arXiv preprint arXiv:1903.03942},
  year={2019}
}
``` 

## Installation for Julia 1.6:

 
 ```
 add SetIntersectionProjection.jl
 ``` 

- The examples also use the packages:
	- [MAT](https://github.com/JuliaIO/MAT.jl)
	- [PyPlot](https://github.com/JuliaPy/PyPlot.jl)
	- [StatsBase](https://github.com/JuliaStats/StatsBase.jl)
	- [InvertibleNetworks](https://github.com/slimgroup/InvertibleNetworks.jl)
 	- [TestImages](https://github.com/JuliaImages/TestImages.jl) 
	- [Images](https://github.com/JuliaImages/Images.jl) 
	- [Flux](https://github.com/FluxML/Flux.jl)
	
	
## Status:

###  January 2022

 - master branch works with Julia 1.5 & 1.6
 - timings for each part of the PARSDMM algorithm are now available as ``` log_PARSDMM.timing``` after solving a projection problem as ```(x1,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,set_Prop,P_sub,comp_grid,options)```
 - Recently added full support for custom JOLI operators, some examples can be found [here](https://github.com/slimgroup/SetIntersectionProjection.jl/blob/master/examples/ConstraintSetupExamples.jl).
 - see [https://github.com/slimgroup/ConstrainedFWIExamples](https://github.com/slimgroup/ConstrainedFWIExamples) for some more examples of how to use this package as the projector in combination with other software packages that compute function values and gradients for PDE-constrained optimization.
 - The waveform inversion example illustrates how SetIntersectionProjection can work together with other packages in other Julia versions (requires some package installations in Julia 0.6)
 - the V06 branch works with Julia 0.6 and is intended only for the waveform inversion example that uses [WAVEFORM.jl](https://github.com/slimgroup/WAVEFORM.jl)
