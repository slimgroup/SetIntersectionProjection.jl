# test projections onto intersection for julia for single image desaturation
# as set theoretical estimation (a feasibility problem)
# We will use a VERY simple learning appraoch to obtain 'good' constraints. This
# learing works with just a few even <10 training examples.

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
m_est        = zeros(TF,size(m_evaluation)) #allocate results


#computational grid for the training images (all images assumed to be on the same grid here)
comp_grid = compgrid((1, 1),(size(m_evaluation,2), size(m_evaluation,3)))
#convert model and computational grid parameters
comp_grid=compgrid( ( convert(TF,comp_grid.d[1]),convert(TF,comp_grid.d[2]) ), comp_grid.n )


#create 'observed data' by creating artificial saturation of images
d_obs = deepcopy(m_evaluation)
d_obs[d_obs.>125.0f0].=125.0f0
d_obs[d_obs.<60.0f0]=60.0f0

# train by observing constraints on the data images in training data set
observations = constraint_learning_by_obseration(comp_grid,m_train)

#define a few constraints and what to do with the observations
constraint=Dict()

constraint["use_bounds"]=false
constraint["m_min"]=0.0
constraint["m_max"]=255.0

constraint["use_TD_hist_eq_relax_1"]=true
constraint["hist_eq_LB_1"] = observations["hist_min"]
constraint["hist_eq_UB_1"] = observations["hist_max"]
constraint["TD_hist_eq_operator_1"]= "identity"

constraint["use_TD_hist_eq_relax_2"]=true
constraint["hist_eq_LB_2"] = observations["hist_TV_min"]
constraint["hist_eq_UB_2"] = observations["hist_TV_max"]
constraint["TD_hist_eq_operator_2"]= "TV"

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

constraint["use_TD_l1_3"]=true
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


constraint["use_TD_card_1"]=false
constraint["TD_card_operator_1"]="curvelet"
constraint["card_1"]=convert(TI,round(quantile(vec(observations["curvelet_card_095"]),0.85)))

constraint["use_TD_bounds_fiber_x"]=false
constraint["TD_bounds_fiber_x_operator"]="DCT"
constraint["TD_LB_fiber_x"]=observations["DCT_x_LB"]
constraint["TD_UB_fiber_x"]=observations["DCT_x_UB"]

constraint["use_TD_bounds_fiber_y"]=false
constraint["TD_bounds_fiber_y_operator"]="DCT"
constraint["TD_LB_fiber_y"]=observations["DCT_y_LB"]
constraint["TD_UB_fiber_y"]=observations["DCT_y_UB"]

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

options.linear_inv_prob_flag = false #compute projection of initial guess
options.parallel             = true
options.zero_ini_guess       = false
BLAS.set_num_threads(2)
FFTW.set_num_threads(2)

(P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,options.FL) #obtain projector and transform-domain operator pairs

#add the blurring filer sparse matrix as a transform domain matrix, along with the properties
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

dummy=zeros(TF,prod(comp_grid.n))
(TD_OP,AtA,l,y) = PARSDMM_precompute_distribute(TD_OP,TD_Prop,comp_grid,options)
x_ini= vec(d_obs[1,:,:])
x_ini[ind_max_clip]=225.0f0
x_ini[ind_min_clip]=0.0f0

options.rho_ini      = ones(TF,length(TD_OP))*1000f0
for i=1:length(options.rho_ini)
  if TD_Prop.ncvx[i]==true
    options.rho_ini[i]=10f0
  end
end

for i=1:size(d_obs,1)
  SNR(in1,in2)=20*log10(norm(in1)/norm(in1-in2))
 [@spawnat pid y[:L][1]=TD_OP[:L][1]*x_ini for pid in y.pids]
 #[@spawnat pid l[:L][1]=randn(TF,length(l[:L][1])) for pid in l.pids]


  p2proj = deepcopy(x_ini) #don't couple initial guess and point to project.
  @time (x,log_PARSDMM) = PARSDMM(p2proj,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options,x_ini,[],y)
  m_est[i,:,:]=reshape(x,comp_grid.n)
  println("SNR:", round(SNR(vec(m_evaluation[i,:,:]),vec(m_est[i,:,:])),2))

  if i+1<=size(d_obs,1)
    data = vec(d_obs[i+1,:,:])
    LBD=data.-2.0;  LBD=convert(Vector{TF},LBD);
    UBD=data.+2.0;  UBD=convert(Vector{TF},UBD);
    ind_min_clip=find(data.==60.0f0)
    ind_max_clip=find(data.==125.0f0)
    LBD[ind_min_clip].=0.0f0
    UBD[ind_max_clip].=255.0f0
    P_sub[end] = x -> project_bounds!(x,LBD,UBD)

    x_ini= vec(d_obs[i+1,:,:])
    x_ini[ind_max_clip]=225.0f0
    x_ini[ind_min_clip]=0.0f0
  end
end
#
# FWD_OP="mask"
# multi_level=false
# n_levels=2
# coarsening_factor=2.0
# (m_est,mask_save)=ICLIP_inpainting(FWD_OP,d_obs,m_evaluation,constraint,comp_grid,options,multi_level,n_levels,coarsening_factor)

SNR(in1,in2)=20*log10(norm(in1)/norm(in1-in2))

using PyPlot

#plot training images
figure();title("training image", fontsize=10)
for i=1:16
  subplot(4,4,i);imshow(m_train[i,:,:],cmap="gray",vmin=0.0,vmax=255.0);axis("off") #title("training image", fontsize=10)
end
savefig(joinpath(data_dir,"training_data_all.eps"),bbox_inches="tight",dpi=600)
savefig(joinpath(data_dir,"training_data_all.png"),bbox_inches="tight")
close()

for i=1:16
  figure();title(string("training image", i), fontsize=10)
  imshow(m_train[i,:,:],cmap="gray",vmin=0.0,vmax=255.0);axis("off") #title("training image", fontsize=10)
  savefig(joinpath(data_dir,string("training_data_", i,".eps")),bbox_inches="tight",dpi=600)
  savefig(joinpath(data_dir,string("training_data_", i,".png")),bbox_inches="tight")
  close()
end

#plot results
for i=1:size(m_est,1)
    figure();imshow(d_obs[i,:,:],cmap="gray",vmin=0.0,vmax=255.0); title("observed");
    savefig(joinpath(data_dir,string("saturized_observed",i,".eps")),bbox_inches="tight",dpi=600)
    savefig(joinpath(data_dir,string("saturized_observed",i,".png")),bbox_inches="tight")
    figure();imshow(m_est[i,:,:],cmap="gray",vmin=0.0,vmax=255.0); title(string("PARSDMM, SNR=", round(SNR(vec(m_evaluation[i,:,:]),vec(m_est[i,:,:])),2)))
    savefig(joinpath(data_dir,string("PARSDMM_desaturation",i,".eps")),bbox_inches="tight",dpi=600)
    savefig(joinpath(data_dir,string("PARSDMM_desaturation",i,".png")),bbox_inches="tight")
    figure();imshow(m_evaluation[i,:,:],cmap="gray",vmin=0.0,vmax=255.0); title("True")
    savefig(joinpath(data_dir,string("desaturation_evaluation",i,".eps")),bbox_inches="tight",dpi=600)
    savefig(joinpath(data_dir,string("desaturation_evaluation",i,".png")),bbox_inches="tight")
    close()
end

#plot histograms of reconstructed on top of true
nbins=10
for i=1:size(m_est,1)
  fig = figure("pyplot_histogram") # Not strictly required
  ax = axes() # Not strictly required
  h = plt[:hist](vec(m_evaluation[i,:,:]),nbins) # Histogram of true image
  h = plt[:hist](vec(m_est[i,:,:]),nbins,alpha=0.5) # Histogram of estimated image
  xlabel("pixel value", fontsize=12)
  ylabel("count", fontsize=12)
  title("Histograms of true and estimated image (PARSDMM)", fontsize=12)
  savefig(joinpath(data_dir,string("hist_desaturation_PARSDMM",i,".eps")),bbox_inches="tight",dpi=300)
  savefig(joinpath(data_dir,string("hist_desaturation_PARSDMM",i,".png")),bbox_inches="tight")
  savefig(joinpath(data_dir,string("hist_desaturation_PARSDMM_HQPNG",i,".png")),bbox_inches="tight",dpi=1200)
  close()
end


# file = matopen(joinpath(data_dir,"x.mat"), "w")
# write(file, "x", convert(Array{Float64,1},x))
# close(file)

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
#
# file = matopen(joinpath(data_dir,"BF.mat"), "w")
# write(file, "BF", convert(SparseMatrixCSC{Float64,Int64},BF))
# close(file)

# (TV_OP, dummy1, dummy2, dummy3)=get_TD_operator(comp_grid,"TV",TF)
# file = matopen("TV_OP.mat", "w")
# write(file, "TV_OP", convert(SparseMatrixCSC{Float64,Int64},TV_OP))
# close(file)
#
# #load TFOCS matlab results and plot
# file = matopen("x_TFOCS_tv_save2.mat")
# x_TFOCS_tv_save=read(file, "x_TFOCS_tv_save2")
# x_TFOCS_tv_save=reshape(x_TFOCS_tv_save,4,comp_grid.n[1],comp_grid.n[2])
#
# file = matopen("m_evaluation.mat")
# m_evaluation=read(file, "m_evaluation")
#
# SNR(in1,in2)=20*log10(norm(in1)/norm(in1-in2))
# for i=1:size(m_est,1)
#   figure()
#   imshow(x_TFOCS_tv_save[i,51:end-51,:],cmap="gray",vmin=0.0,vmax=255.0); title(string("TFOCS BPDN-TV, SNR=", round(SNR(vec(m_evaluation[i,51:end-51,:]),vec(x_TFOCS_tv_save[i,51:end-51,:])),2)))
#   savefig(string("TFOCS_TV_inpainting",i,".eps"),bbox_inches="tight",dpi=600)
# end

#
# #plot zoomed section
# z_parsdmm_1 = m_est[1,51:400,1700:end]
# z_TFOCS_1   = reshape(x_TFOCS_tv_save[1,:,:],comp_grid.n)
# z_TFOCS_1   = z_TFOCS_1[51:400,1700:2051];
# z_true_1    = m_evaluation[1,1:400,1700:end];
#
# z_parsdmm_2 = m_est[2,500:end,600:800]
# z_TFOCS_2   = reshape(x_TFOCS_tv_save[2,:,:],comp_grid.n)
# z_TFOCS_2   = z_TFOCS_2[500:926,600:800];
# z_true_2    = m_evaluation[2,500:end,600:800];
#
# figure();imshow(z_parsdmm_1,cmap="gray",vmin=0.0,vmax=255.0); title("PARSDMM zoomed")
# savefig(string("PARSDMM_inpainting_zoomed_1.eps"),bbox_inches="tight",dpi=600)
# figure();imshow(z_parsdmm_2,cmap="gray",vmin=0.0,vmax=255.0); title("PARSDMM zoomed")
# savefig(string("PARSDMM_inpainting_zoomed_2.eps"),bbox_inches="tight",dpi=600)
#
# figure();imshow(z_TFOCS_1,cmap="gray",vmin=0.0,vmax=255.0); title("BPDN-TV zoomed")
# savefig(string("TFOCS_inpainting_zoomed_1.eps"),bbox_inches="tight",dpi=600)
# figure();imshow(z_TFOCS_2,cmap="gray",vmin=0.0,vmax=255.0); title("BPDN-TV zoomed")
# savefig(string("TFOCS_inpainting_zoomed_2.eps"),bbox_inches="tight",dpi=600)
#
# figure();imshow(z_true_1,cmap="gray",vmin=0.0,vmax=255.0); title("true zoomed")
# savefig(string("true_inpainting_zoomed_1.eps"),bbox_inches="tight",dpi=600)
# figure();imshow(z_true_2,cmap="gray",vmin=0.0,vmax=255.0); title("true zoomed")
# savefig(string("true_inpainting_zoomed_2.eps"),bbox_inches="tight",dpi=600)
