using Distributed
@everywhere using SetIntersectionProjection
using MAT
ENV["MPLBACKEND"]="qt5agg"
using PyPlot
using LinearAlgebra


@everywhere mutable struct compgrid
  d :: Tuple
  n :: Tuple
end

#test 2D
width=[50 100 200 400 800 1600]

test = Vector{Any}(undef,length(width))
N    = Vector{Any}(undef,length(width))

#PARSDMM options:
options              = PARSDMM_options()
options.FL           = Float32
options.evol_rel_tol = 10*eps(options.FL)

set_zero_subnormals(true)
BLAS.set_num_threads(2)

#select working precision
if options.FL==Float64
  TF = Float64
  TI = Int64
elseif options.FL==Float32
  TF = Float32
  TI = Int32
end

#define constraints
constraint = Vector{SetIntersectionProjection.set_definitions}() #initialize

#bounds:
m_min     = 1450.0
m_max     = 4750.0
set_type  = "bounds"
TD_OP     = "identity"
app_mode  = ("matrix","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

#vertical monotonicity
m_min     = 0.0
m_max     = 1e6
set_type  = "bounds"
TD_OP     = "D_z"
app_mode  = ("matrix","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))


#some horizontal smoothness
m_min     = -1.0
m_max     = 1.0
set_type  = "bounds"
TD_OP     = "D_x"
app_mode  = ("matrix","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))


#initialize arrays to save timings
log_T_serial   = Vector{Any}(undef,length(width))
T_tot_serial   = Vector{Any}(undef,length(width))
log_T_parallel = Vector{Any}(undef,length(width))
T_tot_parallel = Vector{Any}(undef,length(width))

log_T_serial_multilevel   = Vector{Any}(undef,length(width))
T_tot_serial_multilevel   = Vector{Any}(undef,length(width))
log_T_parallel_multilevel = Vector{Any}(undef,length(width))
T_tot_parallel_multilevel = Vector{Any}(undef,length(width))

for i=1:length(width)
  print(i)

  file = matopen("../Data/compass_velocity.mat")
  m=read(file, "Data")
  close(file)
  m = m[1:341,1:width[i]]
  m = permutedims(a,[2,1])

  comp_grid = compgrid((TF(25), TF(25)),(size(m,1), size(m,2)))
  m    = convert(Vector{TF},vec(m))
  N[i] = prod(size(m))

  #serial
  @everywhere GC.gc()
  println("")
  println("serial")
  options.parallel = false
  (P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,options.FL)
  (TD_OP,AtA,l,y)       = PARSDMM_precompute_distribute(TD_OP,TD_Prop,comp_grid,options)
  (x,log_PARSDMM)       = PARSDMM(m,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
  val, t, bytes, gctime, memallocs = @timed (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
  println(t)
  log_T_serial[i]=log_PARSDMM;
  T_tot_serial[i]=t;

  #parallel
  @everywhere GC.gc()
  println("")
  println("parallel")
  options.parallel = true
  (P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,options.FL)
  (TD_OP,AtA,l,y)       = PARSDMM_precompute_distribute(TD_OP,TD_Prop,comp_grid,options)
  (x,log_PARSDMM)       = PARSDMM(m,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
  val, t, bytes, gctime, memallocs = @timed (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
  println(t)
  log_T_parallel[i] = log_PARSDMM
  T_tot_parallel[i] = t

  #serial multilevel
  @everywhere GC.gc()
  println("")
  println("serial multilevel")
  options.parallel  = false
  n_levels          = 2
  coarsening_factor = 3
  (TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels)=setup_multi_level_PARSDMM(m,n_levels,coarsening_factor,comp_grid,constraint,options)
  (x,log_PARSDMM) = PARSDMM_multi_level(m,TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels,options);
  val, t, bytes, gctime, memallocs = @timed (x,log_PARSDMM) = PARSDMM_multi_level(m,TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels,options);
  println(t)
  log_T_serial_multilevel[i] = log_PARSDMM
  T_tot_serial_multilevel[i] = t

  #parallel multilevel
  @everywhere GC.gc()
  println("")
  println("parallel multilevel")
  options.parallel  = true
  n_levels          = 2
  coarsening_factor = 3
  (TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels)=setup_multi_level_PARSDMM(m,n_levels,coarsening_factor,comp_grid,constraint,options)
  (x,log_PARSDMM) = PARSDMM_multi_level(m,TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels,options);
  val, t, bytes, gctime, memallocs = @timed (x,log_PARSDMM) = PARSDMM_multi_level(m,TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels,options);
  println(t)
  log_T_parallel_multilevel[i] = log_PARSDMM
  T_tot_parallel_multilevel[i] = t

  if i==1
    figure();imshow(reshape(x,(comp_grid.n[1],comp_grid.n[2]))');colorbar
    savefig("projection_intersection_timings_2D_fig1.eps",bbox_inches="tight",dpi=300)
    savefig("projection_intersection_timings_2D_fig1.png",bbox_inches="tight")
  elseif i==3
    figure();imshow(reshape(x,(comp_grid.n[1],comp_grid.n[2]))');colorbar
    savefig("projection_intersection_timings_2D_fig2.eps",bbox_inches="tight",dpi=300)
    savefig("projection_intersection_timings_2D_fig2.png",bbox_inches="tight")
  elseif i==6
    figure();imshow(reshape(x,(comp_grid.n[1],comp_grid.n[2]))');colorbar
    savefig("projection_intersection_timings_2D_fig3.eps",bbox_inches="tight",dpi=300)
    savefig("projection_intersection_timings_2D_fig3.png",bbox_inches="tight")
  end



end

#plot results
fig, ax = subplots()
ax[:loglog](N, T_tot_serial, marker="o", markersize=10, label="serial",linewidth=5)
ax[:loglog](N, T_tot_parallel, marker="x", markersize=10,label="parallel",linewidth=5)
ax[:loglog](N, T_tot_serial_multilevel, marker="v", markersize=10, label="serial multilevel",linewidth=5)
ax[:loglog](N, T_tot_parallel_multilevel, marker="^", markersize=10, label="parallel multilevel",linewidth=5)
ax[:legend]()
title(string("time 2D vs grid size, JuliaThreads=",Threads.nthreads(),", BLAS threads=",ccall((:openblas_get_num_threads64_, Base.libblas_name), Cint, ())), fontsize=12)
xlabel("N gridpoints", fontsize=15)
ylabel("time [seconds]", fontsize=15)
savefig("projection_intersection_timings2D_1.eps",bbox_inches="tight",dpi=1200)
savefig("projection_intersection_timings2D_1.png",bbox_inches="tight")

#######################################################################################
#######################################################################################
#
# #now with a different set of constraints:
# # transform-domain rank and bounds
# constraint=Dict()
#
# #bound constraints
# constraint["use_bounds"]=true
# constraint["m_min"]=1450
# constraint["m_max"]=4750
#
# #nuclear norm constraint on vertical derivative of the image
# constraint["use_TD_nuclear_1"]=true
# constraint["TD_nuclear_operator_1"]="D_z"
# #the max nuclear norm is adjusted to each model size, see below
#
# log_T_serial=Vector{Any}(length(width))
# T_tot_serial=Vector{Any}(length(width))
# log_T_parallel=Vector{Any}(length(width))
# T_tot_parallel=Vector{Any}(length(width))
# log_T_serial_multilevel=Vector{Any}(length(width))
# T_tot_serial_multilevel=Vector{Any}(length(width))
# log_T_parallel_multilevel=Vector{Any}(length(width))
# T_tot_parallel_multilevel=Vector{Any}(length(width))
#
# for i=1:length(width)
#   print(i)
#
#   file = matopen("compass_velocity.mat")
#   m=read(file, "Data")
#   close(file)
#   m=m[1:341,1:width[i]];
#   m=m';
#   comp_grid = compgrid((TF(25), TF(25)),(size(m,1), size(m,2)))
#
#   (Dz,dummy1,dummy2,TD_n,dummy4)=get_TD_operator(comp_grid,constraint["TD_nuclear_operator_1"],TF)
#   constraint["TD_nuclear_norm_1"]=0.25f0*norm(svdvals(reshape(Dz*vec(m),TD_n)),1)
#
#   m=convert(Vector{TF},vec(m));
#   N[i]=prod(size(m));
#
#   #serial
#   println("")
#   println("serial")
#   options.parallel=false
#   (P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,options.FL)
#   (TD_OP,AtA,l,y) = PARSDMM_precompute_distribute(TD_OP,TD_Prop,comp_grid,options)
#   (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
#   val, t, bytes, gctime, memallocs = @timed (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
#   println(t)
#   log_T_serial[i]=log_PARSDMM;
#   T_tot_serial[i]=t;
#
#   #parallel
#   println("")
#   println("parallel")
#   options.parallel=true
#   (P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,options.FL)
#   (TD_OP,AtA,l,y) = PARSDMM_precompute_distribute(TD_OP,TD_Prop,comp_grid,options)
#   (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
#   val, t, bytes, gctime, memallocs = @timed (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
#   println(t)
#   log_T_parallel[i]=log_PARSDMM;
#   T_tot_parallel[i]=t;
#
#   #serial multilevel
#   println("")
#   println("serial multilevel")
#   options.parallel=false
#   n_levels=2
#   coarsening_factor=3
#   (m_levels,TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels)=setup_multi_level_PARSDMM(m,n_levels,coarsening_factor,comp_grid,constraint,options)
#   (x,log_PARSDMM) = PARSDMM_multi_level(m_levels,TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels,options);
#   val, t, bytes, gctime, memallocs = @timed (x,log_PARSDMM) = PARSDMM_multi_level(m_levels,TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels,options);
#   println(t)
#   log_T_serial_multilevel[i]=log_PARSDMM;
#   T_tot_serial_multilevel[i]=t;
#
#   #parallel multilevel
#   println("")
#   println("parallel multilevel")
#   options.parallel=true
#   n_levels=2
#   coarsening_factor=3
#   (m_levels,TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels)=setup_multi_level_PARSDMM(m,n_levels,coarsening_factor,comp_grid,constraint,options)
#   (x,log_PARSDMM) = PARSDMM_multi_level(m_levels,TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels,options);
#   val, t, bytes, gctime, memallocs = @timed (x,log_PARSDMM) = PARSDMM_multi_level(m_levels,TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels,options);
#   println(t)
#   log_T_parallel_multilevel[i]=log_PARSDMM;
#   T_tot_parallel_multilevel[i]=t;
#
#   if i==1
#     figure();imshow(reshape(x,(comp_grid.n[1],comp_grid.n[2]))');colorbar
#     savefig("projection_intersection_timings_2D_fig1_b.eps",bbox_inches="tight")
#     savefig("projection_intersection_timings_2D_fig1_b.png",bbox_inches="tight")
#   elseif i==6
#     figure();imshow(reshape(x,(comp_grid.n[1],comp_grid.n[2]))');colorbar
#     savefig("projection_intersection_timings_2D_fig3_b.eps",bbox_inches="tight")
#     savefig("projection_intersection_timings_2D_fig3_b.png",bbox_inches="tight")
#   end
#
#
#
# end
#
# #plot results
# fig, ax = subplots()
# ax[:loglog](N, T_tot_serial, label="serial",linewidth=5)
# ax[:loglog](N, T_tot_parallel, label="parallel",linewidth=5)
# ax[:loglog](N, T_tot_serial_multilevel, label="serial multilevel",linewidth=5)
# ax[:loglog](N, T_tot_parallel_multilevel, label="parallel multilevel",linewidth=5)
# ax[:legend]()
# title(string("time 2D vs grid size, JuliaThreads=",Threads.nthreads(),", BLAS threads=",ccall((:openblas_get_num_threads64_, Base.libblas_name), Cint, ())), fontsize=12)
# xlabel("N gridpoints", fontsize=15)
# ylabel("time [seconds]", fontsize=15)
# savefig("projection_intersection_timings2D_2.eps",bbox_inches="tight")
# savefig("projection_intersection_timings2D_2.png",bbox_inches="tight")
