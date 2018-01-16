# test projections onto intersection for julia for simultaneous image inpainting&deblurring
# as set theoretical image recovery problem (a feasibility problem)
# We will use a VERY simple learning appraoch to obtain 'good' constraints. This
# learing works with just a few even <10 training examples.
# South America dataset

@everywhere using SetIntersectionProjection
using MAT

@everywhere type compgrid
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

#data directory for loading and writing results
data_dir = "/data/slim/bpeters/SetIntersection_data_results"

#load a very small data set (12 images only) (Mablab files for compatibility with matlab only solvers for comparison...)
file  = matopen(joinpath(data_dir,"SA_patches.mat"))
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


#create true observed data by blurring and setting pixels to zero
bkl = 25; #blurring kernel length
d_obs = zeros(TF,size(m_evaluation,1),comp_grid.n[1]-bkl,comp_grid.n[2])
temp  = zeros(TF,comp_grid.n[1]-bkl,comp_grid.n[2])

n1=comp_grid.n[1]
Bx=speye(n1)./bkl
for i=2:bkl
  temp=  spdiagm(ones(n1)./bkl,i)
  temp=temp[1:n1,1:n1];
  Bx+= temp;
end
Bx=Bx[1:end-bkl,:]
Iz=speye(TF,comp_grid.n[2]);
BF=kron(Iz,Bx);
BF=convert(SparseMatrixCSC{TF,TI},BF);

#second: subsample
(e1,e2,e3)=size(d_obs)
mask = ones(TF,e2*e3)
s    = randperm(e3*e2)
zero_ind = s[1:Int(8.*round((e2*e3)/10))]
mask[zero_ind].=0.0f0
mask = spdiagm(mask,0)
FWD_OP = convert(SparseMatrixCSC{TF,TI},mask*BF)

#blur and subsample to create observed data
for i=1:size(d_obs,1)
  temp = FWD_OP*vec(m_evaluation[i,:,:])
  #add noise (integers between -2 and 2 for each  nonzero observation point)
  noise = rand([-2,-1,0,1,2],countnz(temp));
  nz_ind = find(temp)
  temp[nz_ind].= temp[nz_ind].+noise
  d_obs[i,:,:] = reshape(temp,comp_grid.n[1]-bkl,comp_grid.n[2])
end
println("finished creating observed data")

#"train" by observing constraints on the data images in training data set
observations = constraint_learning_by_obseration(comp_grid,m_train)
println("finished learning by observation")

#define a few constraints and what to do with the observations
constraint=Dict()
constraint["use_bounds"]=true
constraint["m_min"]=0.0
constraint["m_max"]=255.0

constraint["use_TD_rank_1"]=false;
observations["rank_095"]=sort(vec(observations["rank_095"]))
constraint["TD_max_rank_1"] = convert(TI,round(quantile(observations["rank_095"],0.50)))
constraint["TD_rank_operator_1"]="identity"

constraint["use_TD_nuclear_1"]=true;
constraint["TD_nuclear_operator_1"]="identity"
constraint["TD_nuclear_norm_1"] = convert(TF,quantile(vec(observations["nuclear_norm"]),0.50))

constraint["use_TD_nuclear_2"]=true;
constraint["TD_nuclear_operator_2"]="D_x"
constraint["TD_nuclear_norm_2"] = convert(TF,quantile(vec(observations["nuclear_Dx"]),0.50))

constraint["use_TD_nuclear_3"]=true;
constraint["TD_nuclear_operator_3"]="D_z"
constraint["TD_nuclear_norm_3"] = convert(TF,quantile(vec(observations["nuclear_Dz"]),0.50))

constraint["use_TD_l1_1"]=true
constraint["TD_l1_operator_1"]="TV"
constraint["TD_l1_sigma_1"] = convert(TF,quantile(vec(observations["TV"]),0.50))

constraint["use_TD_l2_1"]=true
constraint["TD_l2_operator_1"]="TV"
constraint["TD_l2_sigma_1"] = convert(TF,quantile(vec(observations["D_l2"]),0.50))

constraint["use_TD_l1_2"]=false
constraint["TD_l1_operator_2"]="curvelet"
constraint["TD_l1_sigma_2"] = 0.5f0*convert(TF,quantile(vec(observations["curvelet_l1"]),0.50))

constraint["use_TD_l1_3"]=false
constraint["TD_l1_operator_3"]="DFT"
constraint["TD_l1_sigma_3"] = convert(TF,quantile(vec(observations["DFT_l1"]),0.50))

constraint["use_TD_bounds_1"]=true
constraint["TDB_operator_1"]="D_x"
constraint["TD_LB_1"]=convert(TF,quantile(vec(observations["D_x_min"]),0.15))
constraint["TD_UB_1"]=convert(TF,quantile(vec(observations["D_x_max"]),0.85))

constraint["use_TD_bounds_2"]=true
constraint["TDB_operator_2"]="D_z"
constraint["TD_LB_2"]=convert(TF,quantile(vec(observations["D_z_min"]),0.15))
constraint["TD_UB_2"]=convert(TF,quantile(vec(observations["D_z_max"]),0.85))

constraint["use_TD_bounds_fiber_x"]=true
constraint["TD_bounds_fiber_x_operator"]="DCT"
constraint["TD_LB_fiber_x"]=observations["DCT_x_LB"]
constraint["TD_UB_fiber_x"]=observations["DCT_x_UB"]

constraint["use_TD_bounds_fiber_y"]=true
constraint["TD_bounds_fiber_y_operator"]="DCT"
constraint["TD_LB_fiber_y"]=observations["DCT_y_LB"]
constraint["TD_UB_fiber_y"]=observations["DCT_y_UB"]

constraint["use_TD_card_1"]=false
constraint["TD_card_operator_1"]="curvelet"
constraint["card_1"]=convert(TI,round(quantile(vec(observations["curvelet_card_095"]),0.85)))

constraint["use_TD_card_2"]=false
constraint["TD_card_operator_2"]="TV"
constraint["card_2"]=convert(TI,round(quantile(vec(observations["TV_card_095"]),0.85)))

constraint["use_TD_annulus_1"]=true
constraint["TD_annulus_operator_1"]="identity"
constraint["TD_annulus_sigma_max_1"]=maximum(observations["annulus"])
constraint["TD_annulus_sigma_min_1"]=minimum(observations["annulus"])

constraint["use_TD_annulus_2"]=true
constraint["TD_annulus_operator_2"]="TV"
constraint["TD_annulus_sigma_max_2"]=maximum(observations["TV_annulus"])
constraint["TD_annulus_sigma_min_2"]=minimum(observations["TV_annulus"])

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
options.maxit                  = 5000
set_zero_subnormals(true)

options.linear_inv_prob_flag = true #this is important to set
options.parallel             = true
options.zero_ini_guess       = true
BLAS.set_num_threads(2)
FFTW.set_num_threads(2)


(P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,options.FL)

#add the mask*blurring filer sparse matrix as a transform domain matrix
push!(TD_OP,FWD_OP)
push!(TD_Prop.AtA_offsets,convert(Vector{TI},0:bkl)) #these are dummy values, actual ofsetts are automatically detected
push!(TD_Prop.ncvx,false)
push!(TD_Prop.banded,true)
push!(TD_Prop.AtA_diag,true)
push!(TD_Prop.dense,false)
push!(TD_Prop.tag,("subsampling blurring filer","x-motion-blur"))

#also add a projector onto the data constraint: i.e. ||A*x-m||=< sigma, or l<=(A*x-m)<=u
data = vec(d_obs[1,:,:])
LBD=data.-2.0;  LBD=convert(Vector{TF},LBD)
UBD=data.+2.0;  UBD=convert(Vector{TF},UBD)
push!(P_sub,x -> project_bounds!(x,LBD,UBD))
println("finished setting up constraints")


dummy=zeros(TF,size(BF,2))
(TD_OP,AtA,l,y) = PARSDMM_precompute_distribute(TD_OP,TD_Prop,comp_grid,options)
println("finished precomputing and distributing")

for i=1:size(d_obs,1)
  SNR(in1,in2)=20*log10(norm(in1)/norm(in1-in2))
  @time (x,log_PARSDMM) = PARSDMM(dummy,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options)
  m_est[i,:,:]=reshape(x,comp_grid.n)
  println("SNR:", round(SNR(vec(m_evaluation[i,(bkl*2):end-(bkl*2),:]),vec(m_est[i,(bkl*2):end-(bkl*2),:])),2))
  if i+1<=size(d_obs,1)
    data = vec(d_obs[i+1,:,:])
    LBD=data.-2.0;  LBD=convert(Vector{TF},LBD);
    UBD=data.+2.0;  UBD=convert(Vector{TF},UBD);
    P_sub[end] = x -> project_bounds!(x,LBD,UBD)
  end
end

using PyPlot

#plot training images
figure();title("training image", fontsize=10)
for i=1:16
  subplot(4,4,i);imshow(m_train[i,:,:],cmap="gray",vmin=0.0,vmax=255.0);axis("off") #title("training image", fontsize=10)
end
savefig(joinpath(data_dir,"training_data_all.pdf"),bbox_inches="tight")
savefig(joinpath(data_dir,"training_data_all.png"),bbox_inches="tight")
close()

for i=1:35
  figure();title(string("training image", i), fontsize=10)
  imshow(m_train[i,:,:],cmap="gray",vmin=0.0,vmax=255.0);axis("off") #title("training image", fontsize=10)
  savefig(joinpath(data_dir,string("training_data_", i,".pdf")),bbox_inches="tight")
  savefig(joinpath(data_dir,string("training_data_", i,".png")),bbox_inches="tight")
  close()
end
close("all")

SNR(in1,in2)=20*log10(norm(in1)/norm(in1-in2))

for i=1:size(m_est,1)
    figure();imshow(d_obs[i,(bkl*2):end-(bkl*2),:],cmap="gray",vmin=0.0,vmax=255.0); title("observed");
    savefig(joinpath(data_dir,string("deblurring_inpainting_observed",i,".pdf")),bbox_inches="tight")
    savefig(joinpath(data_dir,string("deblurring_inpainting_observed",i,".png")),bbox_inches="tight")
    figure();imshow(m_est[i,(bkl*2):end-(bkl*2),:],cmap="gray",vmin=0.0,vmax=255.0); title(string("PARSDMM, SNR=", round(SNR(vec(m_evaluation[i,(bkl*2):end-(bkl*2),:]),vec(m_est[i,(bkl*2):end-(bkl*2),:])),2)))
    savefig(joinpath(data_dir,string("PARSDMM_deblurring_inpainting",i,".pdf")),bbox_inches="tight")
    savefig(joinpath(data_dir,string("PARSDMM_deblurring_inpainting",i,".png")),bbox_inches="tight")
    figure();imshow(m_evaluation[i,(bkl*2):end-(bkl*2),:],cmap="gray",vmin=0.0,vmax=255.0); title("True")
    savefig(joinpath(data_dir,string("deblurring_inpainting_evaluation",i,".pdf")),bbox_inches="tight")
    savefig(joinpath(data_dir,string("deblurring_inpainting_evaluation",i,".png")),bbox_inches="tight")
end

#test TV and bounds only, to see if all the other constraints contribute anything in reconstructuion quality
#
# #constraints
# constraint=Dict()
# constraint["use_bounds"]=true
# constraint["m_min"]=0.0
# constraint["m_max"]=255.0
#
# constraint["use_TD_l1_1"]=true
# constraint["TD_l1_operator_1"]="TV"
# constraint["TD_l1_sigma_1"] = convert(TF,quantile(vec(observations["TV"]),0.50))
#
# (m_est,mask_save)=ICLIP_inpainting(FWD_OP,d_obs,m_evaluation,constraint,comp_grid,options,multi_level,n_levels,coarsening_factor)
#
# for i=1:size(m_est,1)
#   figure()
#   subplot(3,1,1);imshow(d_obs[i,:,:],cmap="gray",vmin=0.0,vmax=255.0); title("observed")
#   subplot(3,1,2);imshow(m_est[i,:,:],cmap="gray",vmin=0.0,vmax=255.0); title("Reconstruction")
#   subplot(3,1,3);imshow(m_evaluation[i,:,:],cmap="gray",vmin=0.0,vmax=255.0); title("True")
# end


file = matopen(joinpath(data_dir,"m_est.mat"), "w")
write(file, "m_est", convert(Array{Float64,3},m_est))
close(file)

file = matopen(joinpath(data_dir,"m_evaluation.mat"), "w")
write(file, "m_evaluation", convert(Array{Float64,3},m_evaluation))
close(file)

file = matopen(joinpath(data_dir,"m_train.mat"), "w")
write(file, "m_train", convert(Array{Float64,3},m_train))
close(file)

file = matopen(joinpath(data_dir,"d_obs.mat"), "w")
write(file, "d_obs", convert(Array{Float64,3},d_obs))
close(file)

file = matopen(joinpath(data_dir,"FWD_OP.mat"), "w")
write(file, "FWD_OP", convert(SparseMatrixCSC{Float64,Int64},FWD_OP))
close(file)


#load TFOCS matlab results and plot
 file = matopen("x_TFOCS_tv_save_SA.mat")
 x_TFOCS_tv_save=read(file, "x_TFOCS_tv_save_SA")
SNR(in1,in2)=20*log10(norm(in1)/norm(in1-in2))
for i=1:size(m_evaluation,1)
  figure()
  imshow(x_TFOCS_tv_save[i,(bkl*2):end-(bkl*2),:],cmap="gray",vmin=0.0,vmax=255.0); title(string("TFOCS BPDN-TV, SNR=", round(SNR(vec(m_evaluation[i,(bkl*2):end-(bkl*2),:]),vec(x_TFOCS_tv_save[i,(bkl*2):end-(bkl*2),:])),2)))
  savefig(string("TFOCS_TV_inpainting",i,".pdf"),bbox_inches="tight")
end
#

# file = matopen("x_TFOCS_W_save_SA.mat")
# x_TFOCS_tv_save=read(file, "x_TFOCS_W_save_SA")
# x_TFOCS_tv_save=reshape(x_TFOCS_tv_save,4,comp_grid.n[1],comp_grid.n[2])
#
# SNR(in1,in2)=20*log10(norm(in1)/norm(in1-in2))
# for i=1:size(m_evaluation,1)
#   figure()
#   imshow(x_TFOCS_tv_save[i,(bkl*2):end-(bkl*2),:],cmap="gray",vmin=0.0,vmax=255.0); title(string("TFOCS BPDN-Wavelet, SNR=", round(SNR(vec(m_evaluation[i,26:end-26,:]),vec(x_TFOCS_tv_save[i,26:end-26,:])),2)))
#   savefig(string("TFOCS_Wavelet_inpainting",i,".pdf"),bbox_inches="tight")
# end


file = matopen("x_SPGL1_wavelet_save_SA.mat")
x_SPGL_wavelet_save=read(file, "x_SPGL1_wavelet_save_SA")
#x_TFOCS_tv_save=reshape(x_TFOCS_tv_save,4,comp_grid.n[1],comp_grid.n[2])

SNR(in1,in2)=20*log10(norm(in1)/norm(in1-in2))
for i=1:size(m_evaluation,1)
  figure()
  imshow(x_SPGL_wavelet_save[i,:,:],cmap="gray",vmin=0.0,vmax=255.0); title(string("SPGL1 BPDN-wavelet, SNR=", round(SNR(vec(m_evaluation[i,(bkl*2):end-(bkl*2),(bkl*2):end-(bkl*2)]),vec(x_SPGL_wavelet_save[i,:,:])),2)))
  savefig(joinpath(data_dir,string("SPGL1_wavelet_inpainting",i,".pdf")),bbox_inches="tight")
  savefig(joinpath(data_dir,string("SPGL1_wavelet_inpainting",i,".png")),bbox_inches="tight")
end
