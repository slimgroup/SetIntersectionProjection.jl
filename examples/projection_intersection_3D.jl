#This script illustates how to set up constraints and project a 3D model onto an intersection
# with PARSDMM in serial, parallel or multilevel (serial or parallel)
# Bas Peters, 2017

using Distributed
@everywhere using LinearAlgebra
@everywhere using SetIntersectionProjection
using HDF5
using PyPlot

@everywhere mutable struct compgrid
  d :: Tuple
  n :: Tuple
end

# Load velocity model - overthrust model
if ~isfile("overthrust_3D_true_model.h5")
  run(`wget ftp://slim.gatech.edu/data/SoftwareRelease/WaveformInversion.jl/3DFWI/overthrust_3D_true_model.h5`)
end
n, d, o, m = read(h5open("overthrust_3D_true_model.h5","r"), "n", "d", "o", "m");

m .= 1000.0 ./ sqrt.(m);
m = m[50:200,50:200,:];

comp_grid = compgrid( (d[1], d[2], d[3]),( size(m,1), size(m,2), size(m,3) ) )
m=vec(m);

#PARSDMM options:
options=PARSDMM_options()
options.FL=Float64
#options=default_PARSDMM_options(options,options.FL)
options.adjust_gamma           = true
options.adjust_rho             = true
options.adjust_feasibility_rho = true
options.Blas_active            = true
options.maxit                  = 500

set_zero_subnormals(true)
@everywhere BLAS.set_num_threads(4)

#select working precision
if options.FL==Float64
  TF = Float64
  TI = Int64
elseif options.FL==Float32
  TF = Float32
  TI = Int32
end

#convert model and computational grid parameters
comp_grid = compgrid( ( convert(TF,comp_grid.d[1]),convert(TF,comp_grid.d[2]),convert(TF,comp_grid.d[3]) ),( convert(TI,comp_grid.n[1]),convert(TI,comp_grid.n[2]),convert(TI,comp_grid.n[3]) ) )
m         = convert(Vector{TF},m)

#define axis limits and colorbar limits
xmax = comp_grid.d[1]*comp_grid.n[1]
ymax = comp_grid.d[2]*comp_grid.n[2]
zmax = comp_grid.d[3]*comp_grid.n[3]
vmi=minimum(m)
vma=maximum(m)

#constraints
constraint = Vector{SetIntersectionProjection.set_definitions}() #initialize

#bounds:
m_min     = minimum(m)
m_max     = maximum(m)-200.0;
set_type  = "bounds"
TD_OP     = "identity"
app_mode  = ("matrix","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP));

#slope constraints (vertical)
m_min     = 0.0
m_max     = 1e6
set_type  = "bounds"
TD_OP     = "D_z"
app_mode  = ("matrix","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP));

options.parallel       = false
(P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL);
(TD_OP,AtA,l,y)        = PARSDMM_precompute_distribute(TD_OP,set_Prop,comp_grid,options);

println("")
println("PARSDMM serial (bounds and bounds on D_z):")
@time (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);
@time (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);
@time (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);

#plot PARSDMM logs
figure();
subplot(3, 3, 3);semilogy(log_PARSDMM.r_pri)          ;title("r primal")
subplot(3, 3, 4);semilogy(log_PARSDMM.r_dual)         ;title("r dual")
subplot(3, 3, 1);semilogy(log_PARSDMM.obj)            ;title(L"$ \frac{1}{2} || \mathbf{m}-\mathbf{x} ||_2^2 $")
subplot(3, 3, 2);semilogy(log_PARSDMM.set_feasibility);title("TD feasibility violation")
subplot(3, 3, 5);plot(log_PARSDMM.cg_it)              ;title("nr. of CG iterations")
subplot(3, 3, 6);semilogy(log_PARSDMM.cg_relres)      ;title("CG rel. res.")
subplot(3, 3, 7);semilogy(log_PARSDMM.rho)            ;title("rho")
subplot(3, 3, 8);plot(log_PARSDMM.gamma)              ;title("gamma")
subplot(3, 3, 9);semilogy(log_PARSDMM.evol_x)         ;title("x evolution")

#plot
m_plot = reshape(m,comp_grid.n);
figure();
subplot(3,1,1);imshow(m_plot[:,:,Int64(round(comp_grid.n[3]/2))],cmap="jet",vmin=vmi,vmax=vma,extent=[0,  xmax, ymax, 0]); title("model to project x-y slice")
subplot(3,1,2);imshow(permutedims(m_plot[:,Int64(round(comp_grid.n[2]/2)),:],[2,1]),cmap="jet",vmin=vmi,vmax=vma,extent=[0,  xmax, zmax, 0]); title("model to project x-z slice")
subplot(3,1,3);imshow(permutedims(m_plot[Int64(round(comp_grid.n[1]/2)),:,:],[2,1]),cmap="jet",vmin=vmi,vmax=vma,extent=[0,  ymax, zmax, 0]); title("model to project y-z slice")

x_plot = reshape(x,comp_grid.n);
figure();
subplot(3,1,1);imshow(x_plot[:,:,Int64(round(comp_grid.n[3]/2))],cmap="jet",vmin=vmi,vmax=vma,extent=[0,  xmax, ymax, 0]); title("projected model x-y slice")
subplot(3,1,2);imshow(permutedims(x_plot[:,Int64(round(comp_grid.n[2]/2)),:],[2,1]),cmap="jet",vmin=vmi,vmax=vma,extent=[0,  xmax, zmax, 0]); title("projected model x-z slice")
subplot(3,1,3);imshow(permutedims(x_plot[Int64(round(comp_grid.n[1]/2)),:,:],[2,1]),cmap="jet",vmin=vmi,vmax=vma,extent=[0,  ymax, zmax, 0]); title("projected model y-z slice")

#use multilevel-serial (2-levels)
#2 levels, the gird point spacing at level 2 is 3X that of the original (level 1) grid
options.parallel  = false
n_levels          = 2
coarsening_factor = 3

#set up all required quantities for each level
(TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels) = setup_multi_level_PARSDMM(m,n_levels,coarsening_factor,comp_grid,constraint,options);

println("")
println("PARSDMM multilevel-serial (bounds and bounds on D_z):")
@time (x,log_PARSDMM) = PARSDMM_multi_level(m,TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels,options);
@time (x,log_PARSDMM) = PARSDMM_multi_level(m,TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels,options);
@time (x,log_PARSDMM) = PARSDMM_multi_level(m,TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels,options);

#parallel single level
println("")
println("PARSDMM parallel (bounds and bounds on D_z):")
options.parallel       = true
(P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL);
(TD_OP,AtA,l,y)        = PARSDMM_precompute_distribute(TD_OP,set_Prop,comp_grid,options);

@time (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);
@time (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);
@time (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);

#use multilevel-parallel (2-levels)
options.parallel  = true
#2 levels, the grid point spacing at level 2 is 3X that of the original (level 1) grid
n_levels          = 2
coarsening_factor = 3

#set up all required quantities for each level
(TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels) = setup_multi_level_PARSDMM(m,n_levels,coarsening_factor,comp_grid,constraint,options);

println("PARSDMM multilevel-parallel (bounds and bounds on D_z):")
@time (x,log_PARSDMM) = PARSDMM_multi_level(m,TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels,options);
@time (x,log_PARSDMM) = PARSDMM_multi_level(m,TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels,options);
@time (x,log_PARSDMM) = PARSDMM_multi_level(m,TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels,options);
