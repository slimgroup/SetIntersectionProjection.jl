#compare parallel Dykstra vs PARSDMM
#look at the total number of PARDMM iterations if
#we use PARSDMM to solve Dykstra subproblems.

@everywhere using SetIntersectionProjection
using MAT
using PyPlot

type compgrid
  d :: Tuple
  n :: Tuple
end

file = matopen("compass_velocity.mat")
m=read(file, "Data")
close(file)
m=m[1:341,200:300]#[1:341,200:600];
m=m';

#PARSDMM options:
options=PARSDMM_options()
options.FL=Float32
options=default_PARSDMM_options(options,options.FL)
set_zero_subnormals(true)

#select working precision
if options.FL==Float64
  TF = Float64
  TI = Int64
elseif options.FL==Float32
  TF = Float32
  TI = Int32
end

comp_grid = compgrid((TF(25), TF(6)),(size(m,1), size(m,2)))
m=convert(Vector{TF},vec(m))
m2=deepcopy(m)
m3=deepcopy(m)

#constraints
constraint=Dict()

constraint["use_bounds"]=true
constraint["m_min"]=1500
constraint["m_max"]=4000

constraint["use_TD_bounds_1"]=true;
constraint["TDB_operator_1"]="D_z";
(D_z, AtA_diag, dense, TD_n)=get_TD_operator(comp_grid,"D_z",TF)
constraint["TD_LB_1"]=0;
constraint["TD_UB_1"]=1e6;

constraint["use_TD_l1_1"]      = true
constraint["TD_l1_operator_1"] = "TV"
(TV_OP, AtA_diag, dense, TD_n)=get_TD_operator(comp_grid,"TV",TF)
constraint["TD_l1_sigma_1"]    = 0.25*norm(TV_OP*m,1)


options.obj_tol=10*eps(TF)
options.feas_tol=10*eps(TF)
BLAS.set_num_threads(2) #2 is fine for a small problem
(P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,options.FL)
(TD_OP,AtA,l,y) = PARSDMM_precompute_distribute(m,TD_OP,TD_Prop,options)

println("PARSDMM serial (bounds, TV and bounds on D_z):")
@time (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);

iteration_list=2:5:150
time_PARSDMM=Vector{Float64}(length(iteration_list))
PARSDMM_feas_log=Array{Float64,2}(length(iteration_list),3)
for i=1:length(iteration_list)
  options.maxit=iteration_list[i]
  temp = @timed (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
  time_PARSDMM[i]=temp[2]
  PARSDMM_feas_log[i,:]=log_PARSDMM.set_feasibility[end-1,:]
end
PARSDMM_obj_stop=abs.(diff(log_PARSDMM.obj[2:5:end])./log_PARSDMM.obj[2:5:5*length(diff(log_PARSDMM.obj[2:5:end]))])


cg_axis=[0 ; cumsum(log_PARSDMM.cg_it)];
cg_axis=cg_axis[1:10:end]

feas_axis=0:10:(10-1)*size(log_PARSDMM.set_feasibility,1)

#define axis limits and colorbar limits
xmax = comp_grid.d[1]*comp_grid.n[1]
zmax = comp_grid.d[2]*comp_grid.n[2]
vmi=1500
vma=4500
figure();imshow(reshape(m,(comp_grid.n[1],comp_grid.n[2]))',cmap="jet",vmin=vmi,vmax=vma,extent=[0,  xmax, zmax, 0]); title("model to project")
figure();imshow(reshape(x,(comp_grid.n[1],comp_grid.n[2]))',cmap="jet",vmin=vmi,vmax=vma,extent=[0,  xmax, zmax, 0]); title("Projection PARSDMM (bounds, bounds on D_z, TV))")

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


## try with convex.jl
using Convex
#using SCS
using ECOS  #segmentation fault if input is float32
#using Mosek
TF=Float64
TI=Int64
options.FL=Float64
(P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,options.FL)
(TV_OP, AtA_diag, dense, TD_n)=get_TD_operator(comp_grid,"TV",TF)
(D_z, AtA_diag, dense, TD_n)=get_TD_operator(comp_grid,"D_z",TF)
m=convert(Vector{TF},m)
tdlb=TF(0.0)
tdub=TF(1.0e6)
cv = Variable(length(m))
TV_lim=TF(0.25*norm(TV_OP*m,1));
problem = minimize(sumsquares(cv-m),[norm(TV_OP*cv,1)<=TV_lim ; D_z*cv<=tdub ; D_z*cv>=tdlb ; cv>=constraint["m_min"] ; cv<=constraint["m_max"]])
tol_general=10*eps(TF)

maxit_list=5:10:200
ECOS_obj_log=Vector{Float64}(length(maxit_list))
time_ECOS=Vector{Float64}(length(maxit_list))
ECOS_feas_log=Array{Float64,2}(length(maxit_list),3)
for i=1:length(maxit_list)
  tic()
  solve!(problem,ECOSSolver(maxit=maxit_list[i],delta=tol_general,eps=tol_general,feastol=tol_general,abstol=tol_general,reltol=tol_general))
  time_ECOS[i]=toc()
  ECOS_obj_log[i]=0.5*norm(vec(cv.value)-m,2).^2
  #don't need more than one run in this case becaues the running time does not decrease
  for j=1:3
    ECOS_feas_log[i,j]=norm(P_sub[j](TD_OP[j]*vec(cv.value))-TD_OP[j]*vec(cv.value))/norm(TD_OP[j]*vec(cv.value))
  end
end
ECOS_obj_stop=abs.(diff(ECOS_obj_log)./ECOS_obj_log[1:end-1])

fig, ax = subplots()
;ax[:semilogy](time_PARSDMM,PARSDMM_feas_log,label="PARSDMM")
;ax[:semilogy](time_ECOS,ECOS_feas_log,label="ECOS")

fig, ax = subplots()
;ax[:semilogy](time_PARSDMM[1:length(PARSDMM_obj_stop)],PARSDMM_obj_stop,label="PARSDMM")
;ax[:semilogy](time_ECOS,ECOS_obj_stop,label="ECOS")
savefig("Dykstra_vs_convexJL.pdf",bbox_inches="tight")
savefig("Dykstra_vs_convexJL.png",bbox_inches="tight")
#
# subplot(2, 1, 2);semilogy(log_PARSDMM.r_dual)         ;title("r dual")
# subplot(3, 3, 1);semilogy(log_PARSDMM.obj)            ;title(L"$ \frac{1}{2} || \mathbf{m}-\mathbf{x} ||_2^2 $")
# subplot(3, 3, 2);semilogy(log_PARSDMM.set_feasibility);title("TD feasibility violation")
# subplot(3, 3, 5);plot(log_PARSDMM.cg_it)              ;title("nr. of CG iterations")
# subplot(3, 3, 6);semilogy(log_PARSDMM.cg_relres)      ;title("CG rel. res.")
# subplot(3, 3, 7);semilogy(log_PARSDMM.rho)            ;title("rho")
# subplot(3, 3, 8);plot(log_PARSDMM.gamma)              ;title("gamma")
# subplot(3, 3, 9);semilogy(log_PARSDMM.evol_x)         ;title("x evolution")
# ;

#
# TF=Float64
# TI=Int64
# options.FL=Float64
# (P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,options.FL)
# (TV_OP, AtA_diag, dense, TD_n)=get_TD_operator(comp_grid,"TV",TF)
# (D_z, AtA_diag, dense, TD_n)=get_TD_operator(comp_grid,"D_z",TF)
# m=convert(Vector{TF},m)
# tdlb=TF(0.0)
# tdub=TF(1.0e6)
# cv = Variable(length(m))
# TV_lim=TF(0.25*norm(TV_OP*m,1));
# problem = minimize(sumsquares(cv-m),[norm(TV_OP*cv,1)<=TV_lim ; D_z*cv<=tdub ; D_z*cv>=tdlb ; cv>=constraint["m_min"] ; cv<=constraint["m_max"]])
# tol_general=10*eps(TF)
# maxit_list=5:5:60
# SCS_obj_log=Vector{Float64}(length(maxit_list))
# time_SCS=Vector{Float64}(length(maxit_list))
# SCS_feas_log=Array{Float64,2}(length(maxit_list),3)
# for i=1:length(maxit_list)
#   tic()
#   solve!(problem,SCSSolver(max_iters=maxit_list[i]))
#   time_SCS[i]=toc()
#   SCS_obj_log[i]=0.5*norm(vec(cv.value)-m,2).^2
#   #don't need more than one run in this case becaues the running time does not decrease
#   for j=1:3
#     SCS_feas_log[i,j]=norm(P_sub[j](TD_OP[j]*vec(cv.value))-TD_OP[j]*vec(cv.value))/norm(TD_OP[j]*vec(cv.value))
#   end
# end
# SCS_obj_stop=abs.(diff(SCS_obj_log)./SCS_obj_log[1:end-1])
#
#
# TF=Float64
# TI=Int64
# options.FL=Float64
# (P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,options.FL)
# (TV_OP, AtA_diag, dense, TD_n)=get_TD_operator(comp_grid,"TV",TF)
# (D_z, AtA_diag, dense, TD_n)=get_TD_operator(comp_grid,"D_z",TF)
# m=convert(Vector{TF},m)
# tdlb=TF(0.0)
# tdub=TF(1.0e6)
# cv = Variable(length(m))
# TV_lim=TF(0.25*norm(TV_OP*m,1));
# problem = minimize(sumsquares(cv-m),[norm(TV_OP*cv,1)<=TV_lim ; D_z*cv<=tdub ; D_z*cv>=tdlb ; cv>=constraint["m_min"] ; cv<=constraint["m_max"]])
# tol_general=10*eps(TF)
# maxit_list=5:5:60
# Mosek_obj_log=Vector{Float64}(length(maxit_list))
# time_Mosek=Vector{Float64}(length(maxit_list))
# Mosek_feas_log=Array{Float64,2}(length(maxit_list),3)
# for i=1:length(maxit_list)
#   tic()
#   solve!(problem,MosekSolver())
#   time_Mosek[i]=toc()
#   Mosek_obj_log[i]=0.5*norm(vec(cv.value)-m,2).^2
#   #don't need more than one run in this case becaues the running time does not decrease
#   for j=1:3
#     Mosek_feas_log[i,j]=norm(P_sub[j](TD_OP[j]*vec(cv.value))-TD_OP[j]*vec(cv.value))/norm(TD_OP[j]*vec(cv.value))
#   end
# end
# Mosek_obj_stop=abs.(diff(Mosek_obj_log)./Mosek_obj_log[1:end-1])


# problem.status
# figure();imshow(reshape(cv.value,(comp_grid.n[1],comp_grid.n[2]))',cmap="jet",vmin=vmi,vmax=vma,extent=[0,  xmax, zmax, 0]); title("Projection PARSDMM (bounds, bounds on D_z, TV))")
