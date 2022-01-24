#This script tests the time it takes to compute a projection vs model size in
#3D. Tests serial PARSDMM, parallel PARSDMM, multilevel serial PARSDMM, and
#multilevel parallel PARSDMM. Requires 5 Julia workers (julia -p 5). timings
#in the paper use export JULIA_NUM_THREADS=4

using Distributed
@everywhere using SetIntersectionProjection
@everywhere using LinearAlgebra

using HDF5

@everywhere mutable struct compgrid
  d :: Tuple
  n :: Tuple
end

width=[50 100 200 340]

test = Vector{Any}(undef,length(width))
N    = Vector{Any}(undef,length(width))

#PARSDMM options:
options              = PARSDMM_options()
options.FL           = Float32
options.evol_rel_tol = 10*eps(options.FL)
set_zero_subnormals(true)

#select working precision
# if options.FL==Float64
#   TF = Float64
#   TI = Int64
# elseif options.FL==Float32
#   TF = Float32
#   TI = Int32
# end
TF = Float32

#define constraints
constraint = Vector{SetIntersectionProjection.set_definitions}() #initialize

#bound constraints
m_min     = 1500.0
m_max     = 6000.0
set_type  = "bounds"
TD_OP     = "identity"
app_mode  = ("tensor","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

#monotonicity
m_min     = 0.0
m_max     = 1e6
set_type  = "bounds"
TD_OP     = "D_z"
app_mode  = ("tensor","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

#lateral smoothness via a bound on the gradient
m_min     = -1.0
m_max     = 1.0
set_type  = "bounds"
TD_OP     = "D_x"
app_mode  = ("tensor","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

m_min     = -1.0
m_max     = 1.0
set_type  = "bounds"
TD_OP     = "D_y"
app_mode  = ("tensor","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))


# Load velocity model
if ~isfile("overthrust_3D_true_model.h5")
  run(`wget ftp://slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/3DFWI/overthrust_3D_true_model.h5`)
end
n, d, o, m_full = read(h5open("overthrust_3D_true_model.h5","r"), "n", "d", "o", "m")


m_full .= 1000.0 ./ sqrt.(m_full);

#initialize arrays to save timings
log_T_serial   = Vector{Any}(undef,length(width))
T_tot_serial   = Vector{Any}(undef,length(width))
log_T_parallel = Vector{Any}(undef,length(width))
T_tot_parallel = Vector{Any}(undef,length(width))

log_T_serial_multilevel   = Vector{Any}(undef,length(width))
T_tot_serial_multilevel   = Vector{Any}(undef,length(width))
log_T_parallel_multilevel = Vector{Any}(undef,length(width))
T_tot_parallel_multilevel = Vector{Any}(undef,length(width))

options.rho_ini = [1.0;1000.0;1000.0;1000.0;1.0]

for i=length(width):-1:1
  print(i)

  m         = m_full[1:width[i],1:width[i],:];
  comp_grid = compgrid( (TF(d[1]), TF(d[2]), TF(d[3])),( size(m,1), size(m,2), size(m,3) ) )
  m         = convert(Vector{TF},vec(m))

  N[i]=prod(size(m));

  #parallel
  @everywhere GC.gc()
  println("")
  println("parallel")
  options.parallel=true
  @everywhere BLAS.set_num_threads(2)
  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)
  (TD_OP,AtA,l,y)        = PARSDMM_precompute_distribute(TD_OP,set_Prop,comp_grid,options)
  (x,log_PARSDMM)        = PARSDMM(m,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);
  val, t, bytes, gctime, memallocs = @timed (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);
  println(t)
  log_T_parallel[i] = log_PARSDMM
  T_tot_parallel[i] = t

  #serial
  @everywhere GC.gc()
  println("")
  println("serial")
  options.parallel=false
  #options.rho_ini = [10.0]
  BLAS.set_num_threads(4)
  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)
  (TD_OP,AtA,l,y)        = PARSDMM_precompute_distribute(TD_OP,set_Prop,comp_grid,options)
  (x,log_PARSDMM)        = PARSDMM(m,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);
  val, t, bytes, gctime, memallocs = @timed (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);
  println(t)
  log_T_serial[i] = log_PARSDMM
  T_tot_serial[i] = t

  #serial multilevel
  @everywhere GC.gc()
  println("")
  println("serial multilevel")
  BLAS.set_num_threads(4)
  #options.rho_ini = [1000.0]
  options.parallel  = false
  n_levels          = 3
  coarsening_factor = 2
  (TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels,constraint_level)=setup_multi_level_PARSDMM(m,n_levels,coarsening_factor,comp_grid,constraint,options)
  (x,log_PARSDMM) = PARSDMM_multi_level(m,TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels,options);
  val, t, bytes, gctime, memallocs = @timed (x,log_PARSDMM) = PARSDMM_multi_level(m,TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels,options);
  println(t)
  log_T_serial_multilevel[i] = log_PARSDMM
  T_tot_serial_multilevel[i] = t

  #parallel multilevel
  @everywhere GC.gc()
  println("")
  println("parallel multilevel")
  @everywhere BLAS.set_num_threads(2)
  options.parallel  = true
  n_levels          = 3
  coarsening_factor = 2
  (TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels,constraint_level)=setup_multi_level_PARSDMM(m,n_levels,coarsening_factor,comp_grid,constraint,options)
  (x,log_PARSDMM) = PARSDMM_multi_level(m,TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels,options);
  val, t, bytes, gctime, memallocs = @timed (x,log_PARSDMM) = PARSDMM_multi_level(m,TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels,options);
  println(t)
  log_T_parallel_multilevel[i] = log_PARSDMM
  T_tot_parallel_multilevel[i] = t

end

ENV["MPLBACKEND"]="qt5agg"
using PyPlot
#plot results
fig, ax = subplots()
ax[:loglog](N, T_tot_serial, marker="o", markersize=10, label="serial",linewidth=5)
ax[:loglog](N, T_tot_parallel, marker="x", markersize=10, label="parallel",linewidth=5)
ax[:loglog](N, T_tot_serial_multilevel, marker="v", markersize=10, label="serial multilevel",linewidth=5)
ax[:loglog](N, T_tot_parallel_multilevel, marker="^", markersize=10, label="parallel multilevel",linewidth=5)
ax[:legend]()
title(string("time 3D vs grid size, JuliaThreads=",Threads.nthreads(),", BLAS threads=",ccall((:openblas_get_num_threads64_, Base.libblas_name), Cint, ())), fontsize=12)
xlabel("N gridpoints", fontsize=15)
ylabel("time [seconds]", fontsize=15)
savefig("projection_intersection_timings3D_1.eps",bbox_inches="tight",dpi=1200)


#######################################################################################
#######################################################################################
