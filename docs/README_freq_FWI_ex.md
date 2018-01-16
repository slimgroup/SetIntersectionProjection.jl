# Frequency domain Full-Waveform Inversion (FWI) with constraints

[Julia script for this example](../examples/constrained_freq_FWI_simple.jl)


Full-waveform inversion is a partial-differential-equation (PDE) constrained optimization problem where we observe (`d_{observed}`) the acoustic pressure of (acoustic) waves emitted by controlled sources. The goal is to estimate parameters (acoustic velocity) `m` by minimizing a non-convex data-misfit `f(m)`. Here we use the non-linear least-squares misfit with additional intersection constraints.

This example illustrates that a) adding multiple constraints may be beneficial for the parameter estimation compared to one or two constraint sets; b) non-convex constraints connect more directly to certain types of prior knowledge about the model than convex constraints do; c) we can reliably solve problems with non-convex constraints. The constrained problem formulation enables us to set up this problem without fine tuning penalty parameters.

The Helmholtz equation models the wave-propagation, and a vertical-seismic-profiling experiment (sources at the surface, receivers in a well) acquires the frequency-domain data. All boundaries are perfectly-matched-layers (PML) that absorb outgoing waves as if the model is spatially unbounded. One-sided 'source illumination', limited frequency range (3-10 Hertz) and the nonconvexity of the data-misfit `f(m)` are challenges that we address by constraining the model estimate. 

We assume that our prior knowledge consists of: a) minimum and maximum velocities (2350 - 2650 m/s), b) The anomaly is rectangular, but we do not know the size, aspect ratio or location. We show how to use the software framework for projections onto an intersection of sets to add constraints to already available codes for seismic parameter estimation. 

Figure #Fig:FWI shows the true model, initial guess, and the estimated models using various combinations of constraints. The data acquisition geometry causes the model estimate with bound constraints to be an elongated diagonal anomaly that is incorrect in terms of size, shape, orientation, and parameter values. Total-variation seems like a good candidate to describe a blocky model with a few coefficients, but it may be difficult to select a total-variation constraint, i.e., the size of the 'TV-ball'. The results show that even in the unrealistic case that we use a TV constraint set to the TV of the true model, we obtain a model estimate (true TV & bounds in the figure) that is only a little bit improved compared to the estimation with bounds only. While many of the oscillations outside of the anomalous region are damped, the anomaly itself is still very poorly estimated. 

Multiple non-convex cardinality and rank constraints help the parameter estimation this example. From the prior information that the anomaly is rectangular, we deduce that the rank of the model is equal to two. We also know that the cardinality of the discrete gradient of each fiber (row or column) is less or equal to two as well. If we assume that the anomaly is not larger than half the total domain extent in each direction, we know that the cardinality of the discrete derivative of the matrix is not larger than the number of grid points in each direction. This is an overestimation of the true cardinality of the discrete gradient of the model. We also overestimate the rank constraint by setting it equal to three. The cardinality of the discrete gradient per column/row is not overestimated. 

The results in the figures show that even with partially overestimated non-convex constraints, the model estimates are better than the one obtained with the true total-variation constraint, while we did not use much prior knowledge. The result with rank constraints and both matrix and fiber-based cardinality constraints on the gradient is the most accurate in terms of anomaly shape. The variation between the results of the various combinations of non-convex constraints may not be very intuitive due to the non-uniqueness of projection onto a non-convex set and could be a disadvantage compared to convex sets. All results with non-convex sets estimate a lower than background velocity anomaly, while the results with convex sets also show higher than background velocity artifacts. 

![](images/FWI_figs/CFWI_simple_freq_m_est_true.png)
![](images/FWI_figs/CFWI_simple_freq_m_est_initial.png) 
![](images/FWI_figs/CFWI_simple_freq_m_est_bounds_only.png) 
![](images/FWI_figs/CFWI_simple_freq_m_est_trueTV_bounds.png) 
![](images/FWI_figs/CFWI_simple_freq_m_est_cardcol_bounds.png) 
![](images/FWI_figs/CFWI_simple_freq_m_est_cardmat_bounds.png) 
![](images/FWI_figs/CFWI_simple_freq_m_est_cardmat_cardcol_bounds.png) 
![](images/FWI_figs/CFWI_simple_freq_m_est_cardmat_cardcol_rank_bounds.png) 