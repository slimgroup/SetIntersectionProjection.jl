@everywhere using SetIntersectionProjection
using HDF5
using PyPlot

@everywhere type compgrid
  d :: Tuple
  n :: Tuple
end

#test 2D
width=[50 100 200 340]

test =Vector{Any}(length(width))
N = Vector{Any}(length(width))

#PARSDMM options:
options=PARSDMM_options()
options.FL=Float32
options.evol_rel_tol =10*eps(options.FL)
set_zero_subnormals(true)
BLAS.set_num_threads(8)

#select working precision
if options.FL==Float64
  TF = Float64
  TI = Int64
elseif options.FL==Float32
  TF = Float32
  TI = Int32
end

constraint=Dict()

#bound constraints
constraint["use_bounds"]=true
constraint["m_min"]=1500.0
constraint["m_max"]=6000.0

constraint["use_TD_bounds_1"]=true;
constraint["TDB_operator_1"]="D_z";
constraint["TD_LB_1"]=0;
constraint["TD_UB_1"]=1e6;

constraint["use_TD_bounds_2"]=true;
constraint["TDB_operator_2"]="D_x";
constraint["TD_LB_2"]=-1.0;
constraint["TD_UB_2"]=1.0;

constraint["use_TD_bounds_3"]=true;
constraint["TDB_operator_3"]="D_y";
constraint["TD_LB_3"]=-1.0;
constraint["TD_UB_3"]=1.0;

# Load velocity model
#get model at:  ftp://slim.eos.ubc.ca/data/SoftwareRelease/WaveformInversion.jl/3DFWI/overthrust_3D_true_model.h5
n,d,o,m_full = h5open("overthrust_3D_true_model.h5","r") do file
  read(file, "n", "d", "o", "m")
end
m_full.=1000./sqrt.(m_full);

log_T_serial=Vector{Any}(length(width))
T_tot_serial=Vector{Any}(length(width))
log_T_parallel=Vector{Any}(length(width))
T_tot_parallel=Vector{Any}(length(width))
log_T_serial_multilevel=Vector{Any}(length(width))
T_tot_serial_multilevel=Vector{Any}(length(width))
log_T_parallel_multilevel=Vector{Any}(length(width))
T_tot_parallel_multilevel=Vector{Any}(length(width))

for i=1:length(width)
  print(i)

  m=m_full[1:width[i],1:width[i],:];
  comp_grid = compgrid( (TF(d[1]), TF(d[2]), TF(d[3])),( size(m,1), size(m,2), size(m,3) ) )
  m=convert(Vector{TF},vec(m))

  N[i]=prod(size(m));

  #serial
  println("")
  println("serial")
  options.parallel=false
  (P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,options.FL)
  (TD_OP,AtA,l,y) = PARSDMM_precompute_distribute(m,TD_OP,TD_Prop,options)
  (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
  val, t, bytes, gctime, memallocs = @timed (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
  log_T_serial[i]=log_PARSDMM;
  T_tot_serial[i]=t;

  #parallel
  println("")
  println("parallel")
  options.parallel=true
  (P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,options.FL)
  (TD_OP,AtA,l,y) = PARSDMM_precompute_distribute(m,TD_OP,TD_Prop,options)
  (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
  val, t, bytes, gctime, memallocs = @timed (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
  log_T_parallel[i]=log_PARSDMM;
  T_tot_parallel[i]=t;

  #serial multilevel
  println("")
  println("serial multilevel")
  options.parallel=false
  n_levels=2
  coarsening_factor=3
  (m_levels,TD_OP_levels,AtA_levels,P_sub_levels,TD_Prop_levels,comp_grid_levels)=setup_multi_level_PARSDMM(m,n_levels,coarsening_factor,comp_grid,constraint,options)
  (x,log_PARSDMM) = PARSDMM_multi_level(m_levels,TD_OP_levels,AtA_levels,P_sub_levels,TD_Prop_levels,comp_grid_levels,options);
  val, t, bytes, gctime, memallocs = @timed (x,log_PARSDMM) = PARSDMM_multi_level(m_levels,TD_OP_levels,AtA_levels,P_sub_levels,TD_Prop_levels,comp_grid_levels,options);
  log_T_serial_multilevel[i]=log_PARSDMM;
  T_tot_serial_multilevel[i]=t;

  #parallel multilevel
  println("")
  println("parallel multilevel")
  options.parallel=true
  n_levels=2
  coarsening_factor=3
  (m_levels,TD_OP_levels,AtA_levels,P_sub_levels,TD_Prop_levels,comp_grid_levels)=setup_multi_level_PARSDMM(m,n_levels,coarsening_factor,comp_grid,constraint,options)
  (x,log_PARSDMM) = PARSDMM_multi_level(m_levels,TD_OP_levels,AtA_levels,P_sub_levels,TD_Prop_levels,comp_grid_levels,options);
  val, t, bytes, gctime, memallocs = @timed (x,log_PARSDMM) = PARSDMM_multi_level(m_levels,TD_OP_levels,AtA_levels,P_sub_levels,TD_Prop_levels,comp_grid_levels,options);
  log_T_parallel_multilevel[i]=log_PARSDMM;
  T_tot_parallel_multilevel[i]=t;

end

# T_cg=Vector{Float64}(length(width))
# T_ini=Vector{Float64}(length(width))
# T_rhs=Vector{Float64}(length(width))
# T_adjust_rho_gamma=Vector{Float64}(length(width))
# T_y_l_update=Vector{Float64}(length(width))
#
# for i=1:length(width)
#   T_cg[i]=log_T[i].T_cg
#   T_rhs[i]=log_T[i].T_rhs
#   T_adjust_rho_gamma[i]=log_T[i].T_adjust_rho_gamma
#   T_y_l_update[i]=log_T[i].T_y_l_upd
#   T_ini[i]=log_T[i].T_ini
# end

#plot results
fig, ax = subplots()
ax[:loglog](N, T_tot_serial, label="serial",linewidth=5)
ax[:loglog](N, T_tot_parallel, label="parallel",linewidth=5)
ax[:loglog](N, T_tot_serial_multilevel, label="serial multilevel",linewidth=5)
ax[:loglog](N, T_tot_parallel_multilevel, label="parallel multilevel",linewidth=5)
ax[:legend]()
title(string("time 3D vs grid size, JuliaThreads=",Threads.nthreads(),", BLAS threads=",ccall((:openblas_get_num_threads64_, Base.libblas_name), Cint, ())), fontsize=12)
xlabel("N gridpoints", fontsize=15)
ylabel("time [seconds]", fontsize=15)
savefig("projection_intersection_timings3D_1.pdf",bbox_inches="tight")
savefig("projection_intersection_timings3D_1.png",bbox_inches="tight")

#
# #plot results
# fig, ax = subplots()
# ax[:loglog](N, T_cg, label="T cg")
# ax[:loglog](N, T_rhs, label="T rhs")
# ax[:loglog](N, T_adjust_rho_gamma, label="T adjust rho,gamma")
# ax[:loglog](N, T_y_l_update, label="T y-upd")
# ax[:loglog](N, T_ini, label="T ini")
# ax[:legend]()
# title("3D time vs grid size", fontsize=15)
# xlabel("N gridpoints", fontsize=15)
# ylabel("time [seconds]", fontsize=15)
# savefig("projection_intersection_timings_3D_2.pdf",bbox_inches="tight")
