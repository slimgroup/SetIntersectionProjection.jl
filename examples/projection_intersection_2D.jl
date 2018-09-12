#This script illustate how to set up constraints and project a 2D model onto an intersection
# with PARSDMM in serial, parallel or multilevel (seria or parallel)
# Bas Peters, 2017

@everywhere using SetIntersectionProjection
using MAT
using PyPlot

type compgrid
  d :: Tuple
  n :: Tuple
end

#PARSDMM options:
options=PARSDMM_options()
options.FL=Float32
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
file = matopen("compass_velocity.mat")
m=read(file, "Data")
close(file)
m=m[1:341,200:600];
m=m';

#set up computational grid (25 and 6 m are the original distances between grid points)
comp_grid = compgrid((TF(25), TF(6)),(size(m,1), size(m,2)))
m=convert(Vector{TF},vec(m))

#define axis limits and colorbar limits
xmax = comp_grid.d[1]*comp_grid.n[1]
zmax = comp_grid.d[2]*comp_grid.n[2]
vmi=1500
vma=4500

#constraints
constraint=Dict()

constraint["use_bounds"]=true
constraint["m_min"]=1500
constraint["m_max"]=4500

constraint["use_TD_bounds_1"]=true;
constraint["TDB_operator_1"]="D_z";
constraint["TD_LB_1"]=0;
constraint["TD_UB_1"]=1e6;

options.parallel             = false
(P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,options.FL)
(TD_OP,AtA,l,y) = PARSDMM_precompute_distribute(TD_OP,TD_Prop,comp_grid,options)

println("")
println("PARSDMM serial (bounds and bounds on D_z):")
@time (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
@time (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
@time (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);

#plot
figure();imshow(reshape(m,(comp_grid.n[1],comp_grid.n[2]))',cmap="jet",vmin=vmi,vmax=vma,extent=[0,  xmax, zmax, 0]); title("model to project")
savefig("original_model.png",bbox_inches="tight")
figure();imshow(reshape(x,(comp_grid.n[1],comp_grid.n[2]))',cmap="jet",vmin=vmi,vmax=vma,extent=[0,  xmax, zmax, 0]); title("Projection (bounds and bounds on D_z)")
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
options.parallel             = true
(P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,options.FL)
(TD_OP,AtA,l,y) = PARSDMM_precompute_distribute(TD_OP,TD_Prop,comp_grid,options)

@time (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
@time (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
@time (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);

#plot
figure();imshow(reshape(x,(comp_grid.n[1],comp_grid.n[2]))',cmap="jet",vmin=vmi,vmax=vma,extent=[0,  xmax, zmax, 0]); title("Projection (bounds and bounds on D_z)")

#use multilevel-serial (2-levels)
options.parallel = false

#2 levels, the gird point spacing at level 2 is 3X that of the original (level 1) grid
n_levels=2
coarsening_factor=3

#set up all required quantities for each level
#(m_levels,TD_OP_levels,AtA_levels,P_sub_levels,TD_Prop_levels,comp_grid_levels)=setup_multi_level_PARSDMM(m,n_levels,coarsening_factor,comp_grid,constraint,options)
(TD_OP_levels,AtA_levels,P_sub_levels,TD_Prop_levels,comp_grid_levels)=setup_multi_level_PARSDMM(m,n_levels,coarsening_factor,comp_grid,constraint,options)

println("")
println("PARSDMM multilevel-serial (bounds and bounds on D_z):")
@time (x,log_PARSDMM) = PARSDMM_multi_level(m,TD_OP_levels,AtA_levels,P_sub_levels,TD_Prop_levels,comp_grid_levels,options);
@time (x,log_PARSDMM) = PARSDMM_multi_level(m,TD_OP_levels,AtA_levels,P_sub_levels,TD_Prop_levels,comp_grid_levels,options);
@time (x,log_PARSDMM) = PARSDMM_multi_level(m,TD_OP_levels,AtA_levels,P_sub_levels,TD_Prop_levels,comp_grid_levels,options);

figure();imshow(reshape(x,(comp_grid.n[1],comp_grid.n[2]))',cmap="jet",vmin=vmi,vmax=vma,extent=[0,  xmax, zmax, 0]); title("Projection (bounds and bounds on D_z)")

#now use multi-level with parallel PARSDMM
options.parallel=true

#2 levels, the gird point spacing at level 2 is 3X that of the original (level 1) grid
n_levels=2
coarsening_factor=3

#set up all required quantities for each level
(TD_OP_levels,AtA_levels,P_sub_levels,TD_Prop_levels,comp_grid_levels)=setup_multi_level_PARSDMM(m,n_levels,coarsening_factor,comp_grid,constraint,options)

println("")
println("PARSDMM multilevel-parallel (bounds and bounds on D_z):")
BLAS.set_num_threads(2)
@time (x,log_PARSDMM) = PARSDMM_multi_level(m,TD_OP_levels,AtA_levels,P_sub_levels,TD_Prop_levels,comp_grid_levels,options);
@time (x,log_PARSDMM) = PARSDMM_multi_level(m,TD_OP_levels,AtA_levels,P_sub_levels,TD_Prop_levels,comp_grid_levels,options);
@time (x,log_PARSDMM) = PARSDMM_multi_level(m,TD_OP_levels,AtA_levels,P_sub_levels,TD_Prop_levels,comp_grid_levels,options);

figure();imshow(reshape(x,(comp_grid.n[1],comp_grid.n[2]))',cmap="jet",vmin=vmi,vmax=vma,extent=[0,  xmax, zmax, 0]); title("Projection (bounds and bounds on D_z)")

#
# #Now test TV and bounds
# options.parallel=true
# constraint=Dict()
#
# constraint["use_bounds"]=true
# constraint["m_min"]=1500
# constraint["m_max"]=4500
#
# constraint["use_TD_l1_1"]      = true
# constraint["TD_l1_operator_1"] = "TV"
# (TV_OP, AtA_diag, dense, TD_n)=get_TD_operator(comp_grid,"TV",TF)
# constraint["TD_l1_sigma_1"]    = 0.25*norm(TV_OP*m,1)
#
# (P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,options.FL)
# (TD_OP,AtA,l,y) = PARSDMM_precompute_distribute(TD_OP,TD_Prop,comp_grid,options)
#
# println("PARSDMM serial (bounds and TV):")
# @time (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
# @time (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
#
# figure();imshow(reshape(x,(comp_grid.n[1],comp_grid.n[2]))',cmap="jet",vmin=vmi,vmax=vma,extent=[0,  xmax, zmax, 0]); title("Projection (bounds and total-variation)")
#
# #plot PARSDMM logs
# figure();
# subplot(3, 3, 3);semilogy(log_PARSDMM.r_pri)          ;title("r primal")
# subplot(3, 3, 4);semilogy(log_PARSDMM.r_dual)         ;title("r dual")
# subplot(3, 3, 1);semilogy(log_PARSDMM.obj)            ;title(L"$ \frac{1}{2} || \mathbf{m}-\mathbf{x} ||_2^2 $")
# subplot(3, 3, 2);semilogy(log_PARSDMM.set_feasibility);title("TD feasibility violation")
# subplot(3, 3, 5);plot(log_PARSDMM.cg_it)              ;title("nr. of CG iterations")
# subplot(3, 3, 6);semilogy(log_PARSDMM.cg_relres)      ;title("CG rel. res.")
# subplot(3, 3, 7);semilogy(log_PARSDMM.rho)            ;title("rho")
# subplot(3, 3, 8);plot(log_PARSDMM.gamma)              ;title("gamma")
# subplot(3, 3, 9);semilogy(log_PARSDMM.evol_x)         ;title("x evolution")
#
#
# #Cardinality (matrix based) on the vertical gradient and an l2 constraint on the lateral gradient
# println("PARSDMM serial (bounds, cardinality on vertical gradient & lateral smoothness):")
# constraint=Dict()
#
# constraint["use_bounds"]=true
# constraint["m_min"]=1500
# constraint["m_max"]=4500
# #add more specific bounds for the top water layer which we always know relatively accurately
# constraint["water_depth"]=204.0
# constraint["water_max"]=1475.0
# constraint["water_min"]=1525.0
#
# constraint["use_TD_card_1"] = true
# constraint["TD_card_operator_1"]="D_z"
# constraint["card_1"] = 5*comp_grid.n[1] #allow 5 discontinuities when looking down from the surface
#
# constraint["use_TD_l2_1"] = true
# constraint["TD_l2_operator_1"] = "D_x"
# #observe the ||D_x m ||_2 of the model we want to project and use that as a constraint
# # This means we want to prevent that the projected model has less lateral smoothness than the original model
# (D_x, dummy1, dummy2, dummy3, dummy4)=get_TD_operator(comp_grid,"D_x")
# constraint["TD_l2_sigma_1"]=norm(D_x*m)
#
# #also add pointwise constraints on l=< Dx =< u to limit the lateral discontinuity magnitude
# constraint["use_TD_bounds_1"]=true
# constraint["TDB_operator_1"]="D_x"
# constraint["TD_LB_1"]=0.5*minimum(D_x*m)
# constraint["TD_UB_1"]=0.5*maximum(D_x*m)
#
#
# #set up constraints
# (P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,options.FL);
#
# parallel=true
# @time (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
# @time (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
#
# figure();imshow(reshape(x,(comp_grid.n[1],comp_grid.n[2]))',cmap="jet",vmin=vmi,vmax=vma,extent=[0,  xmax, zmax, 0]); title("Projection (bounds, l2 on lateral gradient, cardinality on vertical gradient")
# #plot PARSDMM logs
# figure();
# subplot(3, 3, 3);semilogy(log_PARSDMM.r_pri)          ;title("r primal")
# subplot(3, 3, 4);semilogy(log_PARSDMM.r_dual)         ;title("r dual")
# subplot(3, 3, 1);semilogy(log_PARSDMM.obj)            ;title(L"$ \frac{1}{2} || \mathbf{m}-\mathbf{x} ||_2^2 $")
# subplot(3, 3, 2);semilogy(log_PARSDMM.set_feasibility);title("TD feasibility violation")
# subplot(3, 3, 5);plot(log_PARSDMM.cg_it)              ;title("nr. of CG iterations")
# subplot(3, 3, 6);semilogy(log_PARSDMM.cg_relres)      ;title("CG rel. res.")
# subplot(3, 3, 7);semilogy(log_PARSDMM.rho)            ;title("rho")
# subplot(3, 3, 8);plot(log_PARSDMM.gamma)              ;title("gamma")
# subplot(3, 3, 9);semilogy(log_PARSDMM.evol_x)         ;title("x evolution")


# constraint["use_TD_bounds_1"]=true;
# constraint["TDB_operator_1"]="D_x";
# constraint["TD_LB_1"]=-1;
# constraint["TD_UB_1"]=1;
# (P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,options.FL);
# @time (x,log_PARSDMM) = compute_projection_intersection_PARSDMM(x,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
# #@time (x,log_PARSDMM) = compute_projection_intersection_PARSDMM(m,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
# using PyPlot
# figure();imshow(reshape(x,(comp_grid.n[1],comp_grid.n[2]))');colorbar
#

# #cardinality on columns of the model
# constraint["water_depth"]=204.0
# constraint["water_max"]=1500.0
# constraint["water_min"]=1500.0
#
# #lat derivative of original model:
# tdlb=Vector{Float64}(comp_grid.n[2])
# tdub=Vector{Float64}(comp_grid.n[2])
# test=Vector{Float64}(comp_grid.n[2])
# Dx_orig=diff(reshape(m,(comp_grid.n[1],comp_grid.n[2]))',2)./comp_grid.d[1];
# #Dx_test=diff(reshape(x,(comp_grid.n[1],comp_grid.n[2]))',2)./comp_grid.d[1];
# for i=1:comp_grid.n[2]
#   tdlb[i]=minimum(Dx_orig[i,:])
#   tdub[i]=maximum(Dx_orig[i,:])
# #  test[i]=maximum(Dx_test[i,:])
# end
# constraint["use_TD_bounds_1"]=false
# constraint["TDB_operator_1"]="D_x"
# constraint["TD_LB_1"]=vec(repmat(tdlb',comp_grid.n[1],1))
# constraint["TD_UB_1"]=vec(repmat(tdub',comp_grid.n[1],1))
#
# #lateral cardinality
#
#
#
# constraint["use_TD_card_fibre_z"]=true
# constraint["TD_card_fibre_z_operator"]="D_z"
# constraint["card_fibre_z"] = 5
#
# constraint["use_TD_card_fibre_x"]=true
# constraint["TD_card_fibre_x_operator"]="D_x"
# constraint["card_fibre_x"] = 20
#
# options.obj_tol=1e-5
# options.feas_tol=1e-5
# options.adjust_rho=true
# options.adjust_feasibility_rho=true
# (P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,options.FL);
# ini=copy(m)
#  @time (x,log_PARSDMM) = compute_projection_intersection_PARSDMM(m,ini,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options,FL,linear_inv_prob_flag);
#  using PyPlot
#   figure();imshow(reshape(x,(comp_grid.n[1],comp_grid.n[2]))');colorbar
