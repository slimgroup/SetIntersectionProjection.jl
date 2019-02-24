# test projections onto intersection for julia for single image desaturation
# as set theoretical estimation (a feasibility problem)
# We will use a VERY simple learning appraoch to obtain 'good' constraints. This
# learing works with just a few even <10 training examples.
using Distributed
@everywhere using SetIntersectionProjection
using MAT
using Interpolations
using Statistics
using LinearAlgebra
using SparseArrays
using Random
using StatsBase

@everywhere mutable struct compgrid
  d :: Tuple
  n :: Tuple
end

#data directory for loading and writing results

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
file  = matopen(joinpath(dirname(pathof(SetIntersectionProjection)), "../examples/Data/Ternate_patch.mat"))
mtrue = read(file, "Ternate_patch")
mtrue = convert(Array{TF,3},mtrue)

#split data tensor in a training and evaluation data
m_train      = mtrue[[1:3 ; 5:6 ; 8 ; 10:13 ; 15:20],:,:]
m_evaluation = mtrue[[4, 7, 9, 14],:,:]
m_est        = zeros(TF,size(m_evaluation)) #allocate results


#computational grid for the training images (all images assumed to be on the same grid here)
comp_grid = compgrid((1, 1),(size(m_evaluation,2), size(m_evaluation,3)))
#convert model and computational grid parameters
comp_grid=compgrid( ( convert(TF,comp_grid.d[1]),convert(TF,comp_grid.d[2]) ), comp_grid.n )


#create 'observed data' by creating artificial saturation of images
d_obs = deepcopy(m_evaluation)
d_obs[d_obs.>125.0f0] .= 125.0f0
d_obs[d_obs.<60.0f0]  .= 60.0f0

# train by observing constraints on the data images in training data set
observations = constraint_learning_by_obseration(comp_grid,m_train)

#define a few constraints and what to do with the observations
constraint = Vector{SetIntersectionProjection.set_definitions}()

#bounds:
m_min     = 0.0
m_max     = 255.0
set_type  = "bounds"
TD_OP     = "identity"
app_mode  = ("matrix","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

#relaxed histogram constraint:
m_min     = observations["hist_min"]
m_max     = observations["hist_max"]
set_type  = "histogram"
TD_OP     = "identity"
app_mode  = ("matrix","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

#relaxed histogram constraint on discrete derivative of the image:
# m_min     = observations["hist_TV_min"]
# m_max     = observations["hist_TV_max"]
# set_type  = "histogram"
# TD_OP     = "TV"
# app_mode  = ("matrix","")
# custom_TD_OP = ([],false)
# push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

# #rank that preserves 95% of the training images:
# m_min     = 0
# observations["rank_095"]=sort(vec(observations["rank_095"]))
# m_max     = convert(TI,round(quantile(observations["rank_095"],0.25)))
# set_type  = "rank"
# TD_OP     = "identity"
# app_mode  = ("matrix","")
# custom_TD_OP = ([],false)
# push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

#nuclear norm constraint
m_min     = 0
m_max     = convert(TF,quantile(vec(observations["nuclear_norm"]),0.25))
set_type  = "nuclear"
TD_OP     = "identity"
app_mode  = ("matrix","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))


#nuclear norm constraint on the x-derivative of the image
m_min     = 0
m_max     = convert(TF,quantile(vec(observations["nuclear_Dx"]),0.25))
set_type  = "nuclear"
TD_OP     = "D_x"
app_mode  = ("matrix","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

#nuclear norm constraint on the z-derivative of the image
m_min     = 0
m_max     = convert(TF,quantile(vec(observations["nuclear_Dz"]),0.25))
set_type  = "nuclear"
TD_OP     = "D_z"
app_mode  = ("matrix","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

#anisotropic total-variation constraint:
m_min     = 0
m_max     = convert(TF,quantile(vec(observations["TV"]),0.25))
set_type  = "l1"
TD_OP     = "TV"
app_mode  = ("matrix","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

#l2 constraint on discrete derivatives of the image:
m_min     = 0
m_max     = convert(TF,quantile(vec(observations["D_l2"]),0.25))
set_type  = "l2"
TD_OP     = "TV"
app_mode  = ("matrix","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

#l1 constraint on DFT coefficients
m_min     = 0
m_max     = convert(TF,quantile(vec(observations["DFT_l1"]),0.50))
set_type  = "l1"
TD_OP     = "DFT"
app_mode  = ("matrix","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

#bound constraints on x-derivative
m_min     = convert(TF,quantile(vec(observations["D_x_min"]),0.15))
m_max     = convert(TF,quantile(vec(observations["D_x_max"]),0.85))
set_type  = "bounds"
TD_OP     = "D_x"
app_mode  = ("matrix","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

#bound constraints on z-derivative
m_min     = convert(TF,quantile(vec(observations["D_z_min"]),0.15))
m_max     = convert(TF,quantile(vec(observations["D_z_max"]),0.85))
set_type  = "bounds"
TD_OP     = "D_z"
app_mode  = ("matrix","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

#annulus constraint
m_min     = quantile(observations["annulus"],0.15)
m_max     = quantile(observations["annulus"],0.85)
set_type  = "annulus"
TD_OP     = "identity"
app_mode  = ("matrix","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

#annulus constraint on discrete gradients
m_min     = quantile(observations["TV_annulus"],0.15)
m_max     = quantile(observations["TV_annulus"],0.85)
set_type  = "annulus"
TD_OP     = "TV"
app_mode  = ("matrix","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))


#PARSDMM options:
options=PARSDMM_options()
options=default_PARSDMM_options(options,options.FL)
options.evol_rel_tol = 1f-6
options.feas_tol     = 0.001f0
options.obj_tol      = 0.0002f0
options.adjust_gamma           = true
options.adjust_rho             = true
options.adjust_feasibility_rho = true
options.Blas_active            = true
options.maxit                  = 300
set_zero_subnormals(true)

options.feasibility_only     = false #compute projection of initial guess
options.parallel             = true
options.zero_ini_guess       = false
BLAS.set_num_threads(2)
#FFTW.set_num_threads(2) #can't use this in Julia >0.6

(P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL) #obtain projector and transform-domain operator pairs

#add identity matrix as linear operator for the desaturation data-fit constraint
push!(TD_OP,SparseMatrixCSC{TF}(LinearAlgebra.I,comp_grid.n[1]*comp_grid.n[2],comp_grid.n[1]*comp_grid.n[2]))
push!(set_Prop.AtA_offsets,[0]) #these are dummy values, actual ofsetts are automatically detected
push!(set_Prop.ncvx,false)
push!(set_Prop.banded,true)
push!(set_Prop.AtA_diag,true)
push!(set_Prop.dense,false)
push!(set_Prop.tag,("bounds","identity","matrix",""))

#also add a projector onto the data constraint:
#i.e. , or l<=(A*x-m)<=u (or, a norm ||A*x-m||=< sigma)
data = vec(d_obs[1,:,:])
LBD  = data .- 2.0;  LBD=convert(Vector{TF},LBD);
UBD  = data .+ 2.0;  UBD=convert(Vector{TF},UBD);
ind_min_clip = findall(data.==60.0f0)
ind_max_clip = findall(data.==125.0f0)
LBD[ind_min_clip] .= 0.0f0
UBD[ind_max_clip] .= 255.0f0

push!(P_sub,input -> project_bounds!(input,LBD,UBD))

dummy=zeros(TF,prod(comp_grid.n))
(TD_OP,AtA,l,y) = PARSDMM_precompute_distribute(TD_OP,set_Prop,comp_grid,options)

x_ini               = vec(d_obs[1,:,:])
x_ini[ind_max_clip] .= 225.0f0
x_ini[ind_min_clip] .= 0.0f0

options.rho_ini      = ones(TF,length(TD_OP))*1000f0
for i=1:length(options.rho_ini)
  if set_Prop.ncvx[i]==true
    options.rho_ini[i] = 10f0
  end
end

for i=1:size(d_obs,1)
  SNR(in1,in2)=20*log10(norm(in1)/norm(in1-in2))
 [@spawnat pid y[:L][1]=TD_OP[:L][1]*x_ini for pid in y.pids]
 #[@spawnat pid l[:L][1]=randn(TF,length(l[:L][1])) for pid in l.pids]

  p2proj = deepcopy(x_ini) #don't couple initial guess and point to project.
  @time (x,log_PARSDMM) = PARSDMM(p2proj,AtA,TD_OP,set_Prop,P_sub,comp_grid,options,x_ini,[],y)
  m_est[i,:,:]=reshape(x,comp_grid.n)
  println("SNR:", round(SNR(vec(m_evaluation[i,:,:]),vec(m_est[i,:,:])),digits=2))
  println("PSNR:", round(psnr(vec(m_evaluation[i,:,:]),vec(m_est[i,:,:]),maximum(m_evaluation[i,:,:])),digits=2))

  if i+1<=size(d_obs,1)
    data = vec(d_obs[i+1,:,:])
    LBD = data.-2.0;  LBD=convert(Vector{TF},LBD);
    UBD = data.+2.0;  UBD=convert(Vector{TF},UBD);
    ind_min_clip = findall(data.==60.0f0)
    ind_max_clip = findall(data.==125.0f0)
    LBD[ind_min_clip] .= 0.0f0
    UBD[ind_max_clip] .= 255.0f0
    P_sub[end] = x -> project_bounds!(x,LBD,UBD)

    global x_ini = vec(d_obs[i+1,:,:])
    x_ini[ind_max_clip] .= 225.0f0
    x_ini[ind_min_clip] .= 0.0f0
  end

end

SNR(in1,in2)=20*log10(norm(in1)/norm(in1-in2))

ENV["MPLBACKEND"]="qt5agg"
using PyPlot

#all results in one figure
figure(figsize=(4.8, 3.15))
FS  = 5
LS  = 3
TML = 1
PD  = 1
for i=1:size(m_est,1)
  subplot(3,4,i);imshow(d_obs[i,:,:],cmap="gray",vmin=0.0,vmax=255.0); title("Observed",FontSize=FS);tick_params(labelsize=LS,length=TML,pad=PD)
end
for i=1:size(m_est,1)
  subplot(3,4,i+8);imshow(m_est[i,:,:],cmap="gray",vmin=0.0,vmax=255.0); title(string("PARSDMM, PSNR=", round(psnr(vec(m_evaluation[i,:,:]),vec(m_est[i,:,:]),maximum(m_evaluation[i,:,:])),digits=2)),FontSize=FS);;tick_params(labelsize=LS,length=TML,pad=PD)
end
for i=1:size(m_est,1)
  subplot(3,4,i+4);imshow(m_evaluation[i,:,:],cmap="gray",vmin=0.0,vmax=255.0); title("True",FontSize=FS);tick_params(labelsize=LS,length=TML,pad=PD)
end
tight_layout()
PyPlot.subplots_adjust(wspace=-0.5, hspace=0.3)
savefig("desaturation_results.png",bbox_inches="tight",dpi=600)
close("all")

#First 8 in 1 figure
figure(figsize=(4.8, 2.4))
for i=1:8
  subplot(2,4,i);imshow(m_train[i,:,:],cmap="gray",vmin=0.0,vmax=255.0);axis("off") #title("training image", fontsize=10)
end
tight_layout()
PyPlot.subplots_adjust(wspace=-0.5, hspace=0.1)
savefig("training_data_first8.png",bbox_inches="tight",dpi=600)
close()

#plot training images
figure();title("training image", fontsize=10)
for i=1:16
  subplot(4,4,i);imshow(m_train[i,:,:],cmap="gray",vmin=0.0,vmax=255.0);axis("off") #title("training image", fontsize=10)
end
savefig("training_data_all.eps",bbox_inches="tight",dpi=600)
close()

for i=1:16
  figure();title(string("training image", i), fontsize=10)
  imshow(m_train[i,:,:],cmap="gray",vmin=0.0,vmax=255.0);axis("off") #title("training image", fontsize=10)
  savefig(string("training_data_", i,".eps"),bbox_inches="tight",dpi=600)
  close()
end

#plot results individually
for i=1:size(m_est,1)
    figure();imshow(d_obs[i,:,:],cmap="gray",vmin=0.0,vmax=255.0); title("observed");
    savefig(string("saturized_observed",i,".eps"),bbox_inches="tight",dpi=600)
    figure();imshow(m_est[i,:,:],cmap="gray",vmin=0.0,vmax=255.0); title(string("PARSDMM, SNR=", round(SNR(vec(m_evaluation[i,:,:]),vec(m_est[i,:,:])),digits=2)))
    savefig(string("PARSDMM_desaturation",i,".eps"),bbox_inches="tight",dpi=600)
    figure();imshow(m_evaluation[i,:,:],cmap="gray",vmin=0.0,vmax=255.0); title("True")
    savefig(string("desaturation_evaluation",i,".eps"),bbox_inches="tight",dpi=600)
    close("all")
end

#save stuff
file = matopen("m_est.mat", "w")
write(file, "m_est", convert(Array{Float64,3},m_est))
close(file)

file = matopen("m_evaluation.mat", "w")
write(file, "m_evaluation", convert(Array{Float64,3},m_evaluation))
close(file)

file = matopen("m_train.mat", "w")
write(file, "m_train", convert(Array{Float64,3},m_train))
close(file)

file = matopen("d_obs.mat", "w")
write(file, "d_obs", convert(Array{Float64,3},d_obs))
close(file)
