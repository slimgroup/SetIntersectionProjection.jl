#This script illustate how to set up constraints and project a 2D model onto an intersection
# with PARSDMM in serial, parallel or multilevel (seria or parallel)
# Bas Peters, 2017

using Distributed
@everywhere using SetIntersectionProjection
@everywhere using LinearAlgebra
using MAT
#ENV["MPLBACKEND"]="qt5agg"
using PyPlot

@everywhere mutable struct compgrid
  d :: Tuple
  n :: Tuple
end

#PARSDMM options:
options    = PARSDMM_options()
options.FL = Float32
#options=default_PARSDMM_options(options,options.FL)
options.adjust_gamma           = true
options.adjust_rho             = true
options.adjust_feasibility_rho = true
options.Blas_active            = true
options.maxit                  = 500

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

#load image to project
file = matopen(joinpath(dirname(pathof(SetIntersectionProjection)), "../examples/Data/compass_velocity.mat"))
m    = read(file, "Data");close(file)
m    = m[1:341,200:600]
m    = permutedims(m,[2,1])

#set up computational grid (25 and 6 m are the original distances between grid points)
comp_grid = compgrid((TF(25.0), TF(6.0)),(size(m,1), size(m,2)))
m         = convert(Vector{TF},vec(m))

#define axis limits and colorbar limits
xmax = comp_grid.d[1]*comp_grid.n[1]
zmax = comp_grid.d[2]*comp_grid.n[2]
vmi  = 1500
vma  = 4500

#constraints (new setup)
constraint = Vector{SetIntersectionProjection.set_definitions}()

#bounds:
m_min     = 1500.0
m_max     = 4500.0
set_type  = "bounds"
TD_OP     = "identity"
app_mode  = ("matrix","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

#slope constraints (vertical)
m_min     = 0.0
m_max     = 1e6
set_type  = "bounds"
TD_OP     = "D_z"
app_mode  = ("matrix","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

options.parallel       = false
(P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)
(TD_OP,AtA,l,y)        = PARSDMM_precompute_distribute(TD_OP,set_Prop,comp_grid,options)

println("")
println("PARSDMM serial (bounds and bounds on D_z):")
@time (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);
@time (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);
@time (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);

#plot
figure();imshow(permutedims(reshape(m,(comp_grid.n[1],comp_grid.n[2])),[2,1]),cmap="jet",vmin=vmi,vmax=vma,extent=[0,  xmax, zmax, 0]); title("model to project")
savefig("original_model.png",bbox_inches="tight")
figure();imshow(permutedims(reshape(x,(comp_grid.n[1],comp_grid.n[2])),[2,1]),cmap="jet",vmin=vmi,vmax=vma,extent=[0,  xmax, zmax, 0]); title("Projection (bounds and bounds on D_z)")
savefig("projected_model.png",bbox_inches="tight")

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
tight_layout()
#tight_layout(pad=0.0, w_pad=0.0, h_pad=1.0)
savefig("PARSDMM_logs.png",bbox_inches="tight")

println("")
println("PARSDMM parallel (bounds and bounds on D_z):")
options.parallel       = true
(P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)
(TD_OP,AtA,l,y)        = PARSDMM_precompute_distribute(TD_OP,set_Prop,comp_grid,options)

@time (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);
@time (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);
@time (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,set_Prop,P_sub,comp_grid,options);

#plot
figure();
subplot(2,1,1);imshow(permutedims(reshape(m,(comp_grid.n[1],comp_grid.n[2])),[2,1]),cmap="jet",vmin=vmi,vmax=vma,extent=[0,  xmax, zmax, 0]); title("model to project")
subplot(2,1,2);imshow(permutedims(reshape(x,(comp_grid.n[1],comp_grid.n[2])),[2,1]),cmap="jet",vmin=vmi,vmax=vma,extent=[0,  xmax, zmax, 0]); title("Projection (bounds and bounds on D_z)")
savefig("projected_model_ParallelPARSDMM.png",bbox_inches="tight")

#use multilevel-serial (2-levels)
options.parallel = false

#2 levels, the gird point spacing at level 2 is 3X that of the original (level 1) grid
n_levels          = 2
coarsening_factor = 3

#set up all required quantities for each level
#(m_levels,TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels)=setup_multi_level_PARSDMM(m,n_levels,coarsening_factor,comp_grid,constraint,options)
(TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels)=setup_multi_level_PARSDMM(m,n_levels,coarsening_factor,comp_grid,constraint,options)

println("")
println("PARSDMM multilevel-serial (bounds and bounds on D_z):")
@time (x,log_PARSDMM) = PARSDMM_multi_level(m,TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels,options);
@time (x,log_PARSDMM) = PARSDMM_multi_level(m,TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels,options);
@time (x,log_PARSDMM) = PARSDMM_multi_level(m,TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels,options);

figure();
subplot(2,1,1);imshow(permutedims(reshape(m,(comp_grid.n[1],comp_grid.n[2])),[2,1]),cmap="jet",vmin=vmi,vmax=vma,extent=[0,  xmax, zmax, 0]); title("model to project")
subplot(2,1,2);imshow(permutedims(reshape(x,(comp_grid.n[1],comp_grid.n[2])),[2,1]),cmap="jet",vmin=vmi,vmax=vma,extent=[0,  xmax, zmax, 0]); title("Projection (bounds and bounds on D_z)")
savefig("projected_model_MultilevelSerialPARSDMM.png",bbox_inches="tight")

#now use multi-level with parallel PARSDMM
options.parallel=true

#2 levels, the gird point spacing at level 2 is 3X that of the original (level 1) grid
n_levels=2
coarsening_factor=3

#set up all required quantities for each level
(TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels)=setup_multi_level_PARSDMM(m,n_levels,coarsening_factor,comp_grid,constraint,options)

println("")
println("PARSDMM multilevel-parallel (bounds and bounds on D_z):")
BLAS.set_num_threads(2)
@time (x,log_PARSDMM) = PARSDMM_multi_level(m,TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels,options);
@time (x,log_PARSDMM) = PARSDMM_multi_level(m,TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels,options);
@time (x,log_PARSDMM) = PARSDMM_multi_level(m,TD_OP_levels,AtA_levels,P_sub_levels,set_Prop_levels,comp_grid_levels,options);

figure();
subplot(2,1,1);imshow(permutedims(reshape(m,(comp_grid.n[1],comp_grid.n[2])),[2,1]),cmap="jet",vmin=vmi,vmax=vma,extent=[0,  xmax, zmax, 0]); title("model to project")
subplot(2,1,2);imshow(permutedims(reshape(x,(comp_grid.n[1],comp_grid.n[2])),[2,1]),cmap="jet",vmin=vmi,vmax=vma,extent=[0,  xmax, zmax, 0]); title("Projection (bounds and bounds on D_z)")
savefig("projected_model_MultilevelParallelPARSDMM.png",bbox_inches="tight")
