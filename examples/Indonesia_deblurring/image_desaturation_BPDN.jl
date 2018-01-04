# test basis-pursuit denoise formulations for single image desaturation
# This script only serves as comparision for SetIntersectionProjection

@everywhere using SetIntersectionProjection
using MAT
using Interpolations

@everywhere type compgrid
  d :: Tuple
  n :: Tuple
end

#data directory for loading and writing results
data_dir = "/data/slim/bpeters/SetIntersection_data_results"

#select working precision
FL=32
if     FL==64
  TF = Float64
  TI = Int64
elseif FL==32
  TF = Float32
  TI = Int32
end

#load a very small data set (12 images only) (Mablab files for compatibility with matlab only solvers for comparison...)
file  = matopen(joinpath(data_dir,"Ternate_patch.mat"))
mtrue = read(file, "Ternate_patch")
mtrue = convert(Array{TF,3},mtrue)

#split data tensor in a training and evaluation data
m_train      = mtrue[[1:3 ; 5:6 ; 8 ; 10:13 ; 15:20],:,:]
m_evaluation = mtrue[[4, 7, 9, 14],:,:]
m_est_TV_BPDN        = zeros(TF,size(m_evaluation)) #allocate results


#computational grid for the training images (all images assumed to be on the same grid here)
comp_grid = compgrid((1, 1),(size(m_evaluation,2), size(m_evaluation,3)))
#convert model and computational grid parameters
comp_grid=compgrid( ( convert(TF,comp_grid.d[1]),convert(TF,comp_grid.d[2]) ), comp_grid.n )


#create 'observed data' by creating artificial saturation of images
d_obs = deepcopy(m_evaluation)
d_obs[d_obs.>125.0f0].=125.0f0
d_obs[d_obs.<60.0f0]=60.0f0

#define a few constraints and what to do with the observations
constraint=Dict()

constraint["use_bounds"]=true
constraint["m_min"]=0.0
constraint["m_max"]=255.0

#PARSDMM options:
options=PARSDMM_options()
options=default_PARSDMM_options(options,options.FL)
options.evol_rel_tol = 10*eps(TF)
options.feas_tol     = 10*eps(TF)
options.obj_tol      = 10*eps(TF)
options.adjust_gamma           = true
options.adjust_rho             = false
options.adjust_feasibility_rho = false
options.Blas_active            = true
options.maxit                  = 500
set_zero_subnormals(true)

options.linear_inv_prob_flag = true #compute projection of initial guess
options.parallel             = true
options.zero_ini_guess       = true
BLAS.set_num_threads(2)
FFTW.set_num_threads(2)

(P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,options.FL) #obtain projector and transform-domain operator pairs

#add the transform-domain matrix corresponding to the data constraint
push!(TD_OP,convert(SparseMatrixCSC{TF,TI},speye(TF,prod(comp_grid.n))))
push!(TD_Prop.AtA_offsets,[0]) #these are dummy values, actual ofsetts are automatically detected
push!(TD_Prop.ncvx,false)
push!(TD_Prop.banded,true)
push!(TD_Prop.AtA_diag,true)
push!(TD_Prop.dense,false)
push!(TD_Prop.tag,("data term","identity"))

#also add a projector onto the data constraint:
#i.e. , or l<=(A*x-m)<=u or, a norm ||A*x-m||=< sigma
data = vec(d_obs[1,:,:])
LBD=data.-2.0;  LBD=convert(Vector{TF},LBD);
UBD=data.+2.0;  UBD=convert(Vector{TF},UBD);
ind_min_clip=find(data.==60.0f0)
ind_max_clip=find(data.==125.0f0)
LBD[ind_min_clip].=0.0f0
UBD[ind_max_clip].=255.0f0

push!(P_sub,input -> project_bounds!(input,LBD,UBD))

#Now at the objective to minimize: ||Dx||_1 ( ||x||_tv )
# to do this, we add the transform-domain operator and a proximal mapping for the l1-norm
(TV_OP, dummy1, dummy2, TD_n_TV)=get_TD_operator(comp_grid,"TV",TF)
push!(TD_OP,TV_OP)
push!(TD_Prop.AtA_offsets,[0]) #these are dummy values, actual ofsetts are automatically detected
push!(TD_Prop.ncvx,false)
push!(TD_Prop.banded,true)
push!(TD_Prop.AtA_diag,false)
push!(TD_Prop.dense,false)
push!(TD_Prop.tag,("tv_obj","TV"))

push!(P_sub,input -> prox_l1!(input,1000f0))

options.adjust_rho             = false
options.adjust_feasibility_rho = false

dummy=zeros(TF,prod(comp_grid.n))
(TD_OP,AtA,l,y) = PARSDMM_precompute_distribute(dummy,TD_OP,TD_Prop,comp_grid,options)

for i=1:size(d_obs,1)
  SNR(in1,in2)=20*log10(norm(in1)/norm(in1-in2))

  @time (x,log_PARSDMM) = PARSDMM(dummy,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options)
  m_est_TV_BPDN[i,:,:]=reshape(x,comp_grid.n)
  println("SNR:", round(SNR(vec(m_evaluation[i,:,:]),vec(m_est_TV_BPDN[i,:,:])),2))

  if i+1<=size(d_obs,1)
    data = vec(d_obs[i+1,:,:])
    LBD=data.-2.0;  LBD=convert(Vector{TF},LBD);
    UBD=data.+2.0;  UBD=convert(Vector{TF},UBD);
    ind_min_clip=find(data.==60.0f0)
    ind_max_clip=find(data.==125.0f0)
    LBD[ind_min_clip].=0.0f0
    UBD[ind_max_clip].=255.0f0
    P_sub[2] = x -> project_bounds!(x,LBD,UBD)
  end
end

file = matopen(joinpath(data_dir,"m_est_TV_BPDN.mat"), "w")
write(file, "m_est_TV_BPDN", convert(Array{Float64,3},m_est_TV_BPDN))
close(file)

SNR(in1,in2)=20*log10(norm(in1)/norm(in1-in2))

using PyPlot


#plot results
for i=1:size(m_est_TV_BPDN,1)
    figure();imshow(d_obs[i,:,:],cmap="gray",vmin=0.0,vmax=255.0); title("observed");
    savefig(joinpath(data_dir,string("saturized_observed",i,".pdf")),bbox_inches="tight")
    savefig(joinpath(data_dir,string("saturized_observed",i,".png")),bbox_inches="tight")
    figure();imshow(m_est_TV_BPDN[i,:,:],cmap="gray",vmin=0.0,vmax=255.0); title(string("TV BPDN, SNR=", round(SNR(vec(m_evaluation[i,:,:]),vec(m_est_TV_BPDN[i,:,:])),2)))
    savefig(joinpath(data_dir,string("TV_BPDN_desaturation",i,".pdf")),bbox_inches="tight")
    savefig(joinpath(data_dir,string("TV_BPDN_desaturation",i,".png")),bbox_inches="tight")
    figure();imshow(m_evaluation[i,:,:],cmap="gray",vmin=0.0,vmax=255.0); title("True")
    savefig(joinpath(data_dir,string("desaturation_evaluation",i,".pdf")),bbox_inches="tight")
    savefig(joinpath(data_dir,string("desaturation_evaluation",i,".png")),bbox_inches="tight")
end
