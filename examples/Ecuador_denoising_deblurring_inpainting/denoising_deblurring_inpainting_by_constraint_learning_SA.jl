# test projections onto intersection for julia for simultaneous image inpainting&deblurring
# as set theoretical image recovery problem (a feasibility problem)
# We will use a VERY simple learning appraoch to obtain 'good' constraints. This
# learing works with just a few even <10 training examples.
# South America dataset
using Distributed
using LinearAlgebra
using SparseArrays
using Random
using Statistics
using StatsBase

@everywhere using SetIntersectionProjection
using MAT

@everywhere mutable struct compgrid
  d :: Tuple
  n :: Tuple
end

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
file = matopen(joinpath(dirname(pathof(SetIntersectionProjection)), "../examples/Data/SA_patches.mat"))
mtrue = read(file, "SA_patches")
mtrue = convert(Array{TF,3},mtrue)

#split data tensor in a training and evaluation data
#patches have been randomized already in matlab
m_train      = mtrue[1:35,:,:]
m_evaluation = mtrue[36:39,:,:]
m_est        = zeros(TF,size(m_evaluation))


#computational grid for the training images (all images assumed to be on the same grid here)
comp_grid = compgrid((1, 1),(size(m_evaluation,2), size(m_evaluation,3)))
#convert model and computational grid parameters
comp_grid=compgrid( ( convert(TF,comp_grid.d[1]),convert(TF,comp_grid.d[2]) ), comp_grid.n )


#create true observed data by blurring, setting pixels to zero, and adding random zero mean noise

#first, motion blur as a sparse matrix
bkl   = 25; #blurring kernel length
d_obs = zeros(TF,size(m_evaluation,1),comp_grid.n[1]-bkl,comp_grid.n[2])
#temp  = zeros(TF,comp_grid.n[1]-bkl,comp_grid.n[2])

n1 = comp_grid.n[1]
Bx = SparseMatrixCSC{TF}(LinearAlgebra.I,n1,n1)./bkl
for i=2:bkl
  temp =  spdiagm(i => ones(TF,n1)) ./ bkl #spdiagm(ones(n1)./bkl,i)
  temp =  temp[1:n1,1:n1];
  global Bx   += temp;
end
Bx = Bx[1:end-bkl,:]
Iz = SparseMatrixCSC{TF}(LinearAlgebra.I,comp_grid.n[2],comp_grid.n[2])# speye(TF,comp_grid.n[2]);
BF = kron(Iz,Bx);
#BF = convert(SparseMatrixCSC{TF,TI},BF);

#second, subsample
(e1,e2,e3) = size(d_obs)
mask       = ones(TF,e2*e3)
s          = randperm(e3*e2)
zero_ind   = s[1:Int(8.0 .* round((e2*e3)/10))]
mask[zero_ind] .= TF(0.0)
mask       = spdiagm(0 => mask)#spdiagm(mask,0)
FWD_OP     = mask*BF#convert(SparseMatrixCSC{TF,TI},mask*BF)

#blur and subsample to create observed data
for i=1:size(d_obs,1)
  temp = FWD_OP*vec(m_evaluation[i,:,:])

  #add noise (integers between -10 and 10 for each nonzero observation point)
  noise         = rand((-10:10),count(!iszero, temp))#countnz(temp))
  nz_ind        = findall(x->x!=0, temp)
  temp[nz_ind] .= temp[nz_ind].+noise
  d_obs[i,:,:]  = reshape(temp,comp_grid.n[1]-bkl,comp_grid.n[2])
end
println("finished creating observed data")

#"train" by observing constraints on the data images in training data set
observations = constraint_learning_by_obseration(comp_grid,m_train)
println("finished learning by observation")

#define a few constraints
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
# m_max     = convert(TI,round(quantile(observations["rank_095"],0.33)))
# set_type  = "rank"
# TD_OP     = "identity"
# app_mode  = ("matrix","")
# custom_TD_OP = ([],false)
# push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

#nuclear norm constraint
m_min     = 0
m_max     = convert(TF,quantile(vec(observations["nuclear_norm"]),0.33))
set_type  = "nuclear"
TD_OP     = "identity"
app_mode  = ("matrix","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

#nuclear norm constraint on the x-derivative of the image
m_min     = 0
m_max     = convert(TF,quantile(vec(observations["nuclear_Dx"]),0.33))
set_type  = "nuclear"
TD_OP     = "D_x"
app_mode  = ("matrix","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

#nuclear norm constraint on the z-derivative of the image
m_min     = 0
m_max     = convert(TF,quantile(vec(observations["nuclear_Dz"]),0.33))
set_type  = "nuclear"
TD_OP     = "D_z"
app_mode  = ("matrix","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

#anisotropic total-variation constraint:
m_min     = 0
m_max     = convert(TF,quantile(vec(observations["TV"]),0.33))
set_type  = "l1"
TD_OP     = "TV"
app_mode  = ("matrix","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

#anisotropic total-variation constraint:
m_min     = 0
m_max     = convert(TF,quantile(vec(observations["wavelet_l1"]),0.33))
set_type  = "l1"
TD_OP     = "wavelet"
app_mode  = ("matrix","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

#l2 constraint on discrete derivatives of the image:
m_min     = 0
m_max     = convert(TF,quantile(vec(observations["D_l2"]),0.33))
set_type  = "l2"
TD_OP     = "TV"
app_mode  = ("matrix","")
custom_TD_OP = ([],false)
push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

# constraint["use_TD_l1_2"]=false
# constraint["TD_l1_operator_2"]="curvelet"
# constraint["TD_l1_sigma_2"] = 0.5f0*convert(TF,quantile(vec(observations["curvelet_l1"]),0.33))
#
# constraint["use_TD_l1_3"]=false
# constraint["TD_l1_operator_3"]="DFT"
# constraint["TD_l1_sigma_3"] = convert(TF,quantile(vec(observations["DFT_l1"]),0.33))

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

#curvelet cardinality constraint
# m_min     = 0
# m_max     = convert(TI,round(quantile(vec(observations["curvelet_card_095"]),0.85)))
# set_type  = "cardinality"
# TD_OP     = "curvelet"
# app_mode  = ("matrix","")
# custom_TD_OP = ([],false)
# push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

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

options.feasibility_only     = true #this is important to set
options.parallel             = true
options.zero_ini_guess       = true
BLAS.set_num_threads(2)
#FFTW.set_num_threads(2) (this is not supported in Julia 0.7)


(P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)

#add the mask*blurring filer sparse matrix as a transform domain matrix
push!(TD_OP,FWD_OP)
push!(set_Prop.AtA_offsets,[0])#convert(Vector{TI},0:bkl)) #these are dummy values, actual ofsetts are automatically detected
push!(set_Prop.ncvx,false)
push!(set_Prop.banded,true)
push!(set_Prop.AtA_diag,true)
push!(set_Prop.dense,false)
push!(set_Prop.tag,("bounds","subsampled x-motion-blur","matrix",""))

#also add a projector onto the data constraint: i.e. ||A*x-m||=< sigma, or l<=(A*x-m)<=u
data = vec(d_obs[1,:,:])
LBD=data.-15.0;  LBD=convert(Vector{TF},LBD)
UBD=data.+15.0;  UBD=convert(Vector{TF},UBD)
push!(P_sub,x -> project_bounds!(x,LBD,UBD))
println("finished setting up constraints")


dummy=zeros(TF,size(BF,2))
(TD_OP,AtA,l,y) = PARSDMM_precompute_distribute(TD_OP,set_Prop,comp_grid,options)
println("finished precomputing and distributing")

options.rho_ini      = ones(TF,length(TD_OP))*TF(1000.0)
for i=1:length(options.rho_ini)
  if set_Prop.ncvx[i]==true
    options.rho_ini[i] = TF(10.0)
  end
end


for i=1:size(d_obs,1)
  SNR(in1,in2)=20*log10(norm(in1)/norm(in1-in2))
  @time (x,log_PARSDMM) = PARSDMM(dummy,AtA,TD_OP,set_Prop,P_sub,comp_grid,options)
  m_est[i,:,:]=reshape(x,comp_grid.n)
  println("SNR:", round(SNR(vec(m_evaluation[i,(bkl*2):end-(bkl*2),:]),vec(m_est[i,(bkl*2):end-(bkl*2),:])),digits=2))
  println("PSNR:", round(psnr(vec(m_evaluation[i,(bkl*2):end-(bkl*2),:]),vec(m_est[i,(bkl*2):end-(bkl*2),:]),maximum(m_evaluation[i,:,:])),digits=2))
  if i+1<=size(d_obs,1)
    data = vec(d_obs[i+1,:,:])
    LBD=data.-15.0;  LBD=convert(Vector{TF},LBD);
    UBD=data.+15.0;  UBD=convert(Vector{TF},UBD);
    P_sub[end] = x -> project_bounds!(x,LBD,UBD)
  end
end


# for i=1:35
#   figure();title(string("training image", i), fontsize=10)
#   imshow(m_train[i,:,:],cmap="gray",vmin=0.0,vmax=255.0);axis("off") #title("training image", fontsize=10)
#   savefig(string("training_data_", i,".eps"),bbox_inches="tight",dpi=600)
#   savefig(string("training_data_", i,".eps"),bbox_inches="tight")
#   close()
# end
# close("all")

ENV["MPLBACKEND"]="qt5agg"
using PyPlot

#plot first 8 training in 1 figure
figure();
for i=1:8
  subplot(2,4,i);imshow(m_train[i,:,:],cmap="gray",vmin=0.0,vmax=255.0);axis("off") #title("training image", fontsize=10)
end
savefig("training_data_first8.eps",bbox_inches="tight",dpi=300)
close()

SNR(in1,in2)=20*log10(norm(in1)/norm(in1-in2))

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

file = matopen("FWD_OP.mat", "w")
write(file, "FWD_OP", convert(SparseMatrixCSC{Float64,Int64},FWD_OP))
close(file)

file = matopen("x_SPGL1_wavelet_save_SA.mat")
x_SPGL_wavelet_save=read(file, "x_SPGL1_wavelet_save_SA")
x_SPGL_wavelet_save = convert(Array{TF,3},x_SPGL_wavelet_save)

figure(figsize=(4.8, 4.8))
FS  = 5
LS  = 3
TML = 1
PD  = 1
for i=1:size(m_est,1)
  subplot(4,4,i);imshow(d_obs[i,(bkl*2):end-(bkl*2),(bkl*2):end-(bkl*2)],cmap="gray",vmin=0.0,vmax=255.0); title("observed",FontSize=FS,verticalalignment="bottom");tick_params(labelsize=LS,length=TML,pad=PD)
end
for i=1:size(m_est,1)
  subplot(4,4,i+8);imshow(m_est[i,(bkl*2):end-(bkl*2),(bkl*2):end-(bkl*2)],cmap="gray",vmin=0.0,vmax=255.0); title(FontSize=FS,string("PARSDMM, PSNR=", round(psnr(vec(m_evaluation[i,(bkl*2):end-(bkl*2),(bkl*2):end-(bkl*2)]),vec(m_est[i,(bkl*2):end-(bkl*2),(bkl*2):end-(bkl*2)]),maximum(m_evaluation[i,(bkl*2):end-(bkl*2),(bkl*2):end-(bkl*2)])),digits=2)));tick_params(labelsize=LS,length=TML,pad=PD)
end
for i=1:size(m_est,1)
  subplot(4,4,i+4);imshow(m_evaluation[i,(bkl*2):end-(bkl*2),(bkl*2):end-(bkl*2)],cmap="gray",vmin=0.0,vmax=255.0); title(FontSize=FS,"True");tick_params(labelsize=LS,length=TML,pad=PD)
end
for i=1:size(m_est,1)
  subplot(4,4,i+12);imshow(x_SPGL_wavelet_save[i,(bkl*2):end-(bkl*2),(bkl*2):end-(bkl*2)],cmap="gray",vmin=0.0,vmax=255.0); title(FontSize=FS,string("BPDN-wavelet, PSNR=", round(psnr(vec(m_evaluation[i,(bkl*2):end-(bkl*2),(bkl*2):end-(bkl*2)]),vec(x_SPGL_wavelet_save[i,(bkl*2):end-(bkl*2),(bkl*2):end-(bkl*2)]),maximum(m_evaluation[i,(bkl*2):end-(bkl*2),(bkl*2):end-(bkl*2)])),digits=2)));tick_params(labelsize=LS,length=TML,pad=PD)
end
tight_layout()
PyPlot.subplots_adjust(wspace=-0.05, hspace=0.25)
savefig("deblurring_inpainting_results.png",bbox_inches="tight",dpi=600)


#First 8 in 1 figure
figure(figsize=(6.4, 3.2))
for i=1:8
  subplot(2,4,i);imshow(m_train[i,:,:],cmap="gray",vmin=0.0,vmax=255.0);axis("off") #title("training image", fontsize=10)
end
tight_layout()
PyPlot.subplots_adjust(wspace=-0.15, hspace=0.1)
savefig("training_data_first8.png",bbox_inches="tight",dpi=600)
close()

for i=1:size(m_est,1)
    figure();imshow(d_obs[i,(bkl*2):end-(bkl*2),:],cmap="gray",vmin=0.0,vmax=255.0); title("observed");
    savefig(string("deblurring_inpainting_observed",i,".eps"),bbox_inches="tight",dpi=600)
    figure();imshow(m_est[i,(bkl*2):end-(bkl*2),:],cmap="gray",vmin=0.0,vmax=255.0); title(string("PARSDMM, SNR=", round(SNR(vec(m_evaluation[i,(bkl*2):end-(bkl*2),:]),vec(m_est[i,(bkl*2):end-(bkl*2),:])),digits=2)))
    savefig(string("PARSDMM_deblurring_inpainting",i,".eps"),bbox_inches="tight",dpi=600)
    figure();imshow(m_evaluation[i,(bkl*2):end-(bkl*2),:],cmap="gray",vmin=0.0,vmax=255.0); title("True")
    savefig(string("deblurring_inpainting_evaluation",i,".eps"),bbox_inches="tight",dpi=600)
    close("all")
end

#all results in one figure (PARSDMM + SPGL1)
# figure()
# for i=1:size(m_est,1)
#   subplot(4,4,i);imshow(d_obs[i,(bkl*2):end-(bkl*2),(bkl*2):end-(bkl*2)],cmap="gray",vmin=0.0,vmax=255.0); title("observed");
# end
# for i=1:size(m_est,1)
#   subplot(4,4,i+8);imshow(m_est[i,(bkl*2):end-(bkl*2),(bkl*2):end-(bkl*2)],cmap="gray",vmin=0.0,vmax=255.0); title(string("PARSDMM, SNR=", round(SNR(vec(m_evaluation[i,(bkl*2):end-(bkl*2),(bkl*2):end-(bkl*2)]),vec(m_est[i,(bkl*2):end-(bkl*2),(bkl*2):end-(bkl*2)])),digits=2)))
# end
# for i=1:size(m_est,1)
#   subplot(4,4,i+4);imshow(m_evaluation[i,(bkl*2):end-(bkl*2),(bkl*2):end-(bkl*2)],cmap="gray",vmin=0.0,vmax=255.0); title("True")
# end
# for i=1:size(m_est,1)
#   subplot(4,4,i+12);imshow(x_SPGL_wavelet_save[i,(bkl*2):end-(bkl*2),(bkl*2):end-(bkl*2)],cmap="gray",vmin=0.0,vmax=255.0); title(string("BPDN-wavelet, SNR=", round(SNR(vec(m_evaluation[i,(bkl*2):end-(bkl*2),(bkl*2):end-(bkl*2)]),vec(x_SPGL_wavelet_save[i,(bkl*2):end-(bkl*2),(bkl*2):end-(bkl*2)])),digits=2)))
# end

#plot training images
figure();title("training image", fontsize=10)
for i=1:16
  subplot(4,4,i);imshow(m_train[i,:,:],cmap="gray",vmin=0.0,vmax=255.0);axis("off") #title("training image", fontsize=10)
end
savefig("training_data_all.eps",bbox_inches="tight",dpi=600)
close()

SNR(in1,in2)=20*log10(norm(in1)/norm(in1-in2))
for i=1:size(m_evaluation,1)
  figure()
  imshow(x_SPGL_wavelet_save[i,(bkl*2):end-(bkl*2),(bkl*2):end-(bkl*2)],cmap="gray",vmin=0.0,vmax=255.0); title(string("SPGL1 BPDN-wavelet, SNR=", round(SNR(vec(m_evaluation[i,(bkl*2):end-(bkl*2),(bkl*2):end-(bkl*2)]),vec(x_SPGL_wavelet_save[i,(bkl*2):end-(bkl*2),(bkl*2):end-(bkl*2)])),digits=2)))
  savefig(string("SPGL1_wavelet_inpainting",i,".eps"),bbox_inches="tight",dpi=600)
  close()
end
