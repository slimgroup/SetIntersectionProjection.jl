# test projections onto intersection for julia for image deblurring
# as set theoretical estimation (a feasibility problem)
# We will use a VERY simple learning appraoch to obtain 'good' constraints. This
# learing works with just a few even <10 training examples.

@everywhere using SetIntersectionProjection
@everywhere using MAT
@everywhere using PyPlot

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

#plot training images
figure();title("training image", fontsize=10)
for i=1:16
  subplot(4,4,i);imshow(m_train[i,:,:],cmap="gray",vmin=0.0,vmax=255.0);axis("off") #title("training image", fontsize=10)
end
savefig(joinpath(data_dir,"training_data_all.pdf"),bbox_inches="tight")
savefig(joinpath(data_dir,"training_data_all.png"),bbox_inches="tight")

for i=1:16
  figure();title(string("training image", i), fontsize=10)
  imshow(m_train[i,:,:],cmap="gray",vmin=0.0,vmax=255.0);axis("off") #title("training image", fontsize=10)
  savefig(joinpath(data_dir,string("training_data_", i,".pdf")),bbox_inches="tight")
  savefig(joinpath(data_dir,string("training_data_", i,".png")),bbox_inches="tight")
end


#computational grid for the training images (all images assumed to be on the same grid here)
comp_grid = compgrid((1, 1),(size(m_evaluation,2), size(m_evaluation,3)))
#convert model and computational grid parameters
comp_grid=compgrid( ( convert(TF,comp_grid.d[1]),convert(TF,comp_grid.d[2]) ), comp_grid.n )

# Now set up the de-motion blur examples (as a sparse matrix)
bkl = 50; #blurring kernel length
n1=comp_grid.n[1]
Bx=speye(n1)./bkl
for i=1:bkl
  temp=  spdiagm(ones(n1)./bkl,i)
  temp=temp[1:n1,1:n1];
  Bx+= temp;
end
Bx=Bx[1:end-bkl,:]
Iz=speye(TF,comp_grid.n[2]);
BF=kron(Iz,Bx);
BF=convert(SparseMatrixCSC{TF,TI},BF);

#create 'observed data' by blurring evaluation images
d_obs = zeros(TF,size(m_evaluation,1),comp_grid.n[1]-bkl,comp_grid.n[2])
for i=1:size(d_obs,1)
  d_obs[i,:,:] = reshape(BF*vec(m_evaluation[i,:,:]),comp_grid.n[1]-bkl,comp_grid.n[2])
end

# train by observing constraints on the data images in training data set
observations = constraint_learning_by_obseration(comp_grid,m_train)

#define a few constraints and what to do with the observations
constraint=Dict()

constraint["use_bounds"]=true
constraint["m_min"]=0.0
constraint["m_max"]=255.0

constraint["use_TD_rank"]=true;
observations["rank_095"]=sort(vec(observations["rank_095"]))
constraint["TD_max_rank"] = convert(TI,round(quantile(observations["rank_095"],0.50)))
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
#
# constraint["use_TD_bounds_3"]=true
# constraint["TDB_operator_3"]="DCT"
# constraint["TD_LB_3"]=observations["DCT_LB"]
# constraint["TD_UB_3"]=observations["DCT_UB"]
#
# constraint["TD_UB_3"]=reshape(constraint["TD_UB_3"],comp_grid.n)
# constraint["TD_LB_3"]=reshape(constraint["TD_LB_3"],comp_grid.n)
# temp1=zeros(TF,comp_grid.n)
# temp2=zeros(TF,comp_grid.n)
# for j=2:size(constraint["TD_LB_3"],1)-1
#   for k=2:size(constraint["TD_LB_3"],2)-1
#     temp1[j,k].=minimum(constraint["TD_LB_3"][j-1:j+1,k-1:k+1]);
#     temp2[j,k].=maximum(constraint["TD_UB_3"][j-1:j+1,k-1:k+1]);
#   end
# end
# temp1[1,:]=constraint["TD_LB_3"][1,:];
# temp1[:,1]=constraint["TD_LB_3"][:,1]
# temp1[end,:]=constraint["TD_LB_3"][end,:]
# temp1[:,end]=constraint["TD_LB_3"][:,end]
#
# temp2[1,:]=constraint["TD_UB_3"][1,:]
# temp2[:,1]=constraint["TD_UB_3"][:,1]
# temp2[end,:]=constraint["TD_UB_3"][end,:]
# temp2[:,end]=constraint["TD_UB_3"][:,end]
#
# for j=1:size(constraint["TD_LB_3"],1)-1
#   for k=1:size(constraint["TD_LB_3"],2)-1
#     temp1[j,k].=minimum(constraint["TD_LB_3"][j:j+1,k:k+1]);
#     temp2[j,k].=maximum(constraint["TD_UB_3"][j:j+1,k:k+1]);
#   end
# end
# for j=2:size(constraint["TD_LB_3"],1)
#   for k=2:size(constraint["TD_LB_3"],2)
#     temp1[j,k].=minimum(constraint["TD_LB_3"][j-1:j,k-1:k]);
#     temp2[j,k].=maximum(constraint["TD_UB_3"][j-1:j,k-1:k]);
#   end
# end
#
# constraint["TD_LB_3"]=vec(temp1)
# constraint["TD_UB_3"]=vec(temp2)


constraint["use_TD_card_1"]=false
constraint["TD_card_operator_1"]="curvelet"
constraint["card_1"]=convert(TI,round(quantile(vec(observations["curvelet_card_095"]),0.85)))

constraint["use_TD_card_2"]=false
constraint["TD_card_operator_2"]="TV"
constraint["card_2"]=convert(TI,round(quantile(vec(observations["TV_card_095"]),0.85)))

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

(P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,options.FL) #obtain projector and transform-domain operator pairs

#add the blurring filer sparse matrix as a transform domain matrix, along with the properties
push!(TD_OP,BF)
push!(TD_Prop.AtA_offsets,convert(Vector{TI},0:bkl))
push!(TD_Prop.ncvx,false)
push!(TD_Prop.banded,true)
push!(TD_Prop.AtA_diag,true)
push!(TD_Prop.dense,false)
push!(TD_Prop.tag,("blurring filer","x-motion-blur"))

#also add a projector onto the data constraint: i.e. ||A*x-m||=< sigma, or l<=(A*x-m)<=u
data = vec(d_obs[1,:,:])
LBD=data.-2.0;  LBD=convert(Vector{TF},LBD);
UBD=data.+2.0;  UBD=convert(Vector{TF},UBD);
push!(P_sub,x -> project_bounds!(x,LBD,UBD))

dummy=zeros(TF,size(BF,2))
(TD_OP,AtA,l,y) = PARSDMM_precompute_distribute(dummy,TD_OP,TD_Prop,options)

for i=1:size(d_obs,1)
  SNR(in1,in2)=20*log10(norm(in1)/norm(in1-in2))
  @time (x,log_PARSDMM) = PARSDMM(dummy,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options)
  m_est[i,:,:]=reshape(x,comp_grid.n)
  println("SNR:", round(SNR(vec(m_evaluation[i,(bkl+1):end-(bkl+1),:]),vec(m_est[i,(bkl+1):end-(bkl+1),:])),2))

  if i+1<=size(d_obs,1)
    data = vec(d_obs[i+1,:,:])
    LBD=data.-2.0;  LBD=convert(Vector{TF},LBD);
    UBD=data.+2.0;  UBD=convert(Vector{TF},UBD);
    P_sub[end] = x -> project_bounds!(x,LBD,UBD)
  end
end
#
# FWD_OP="mask"
# multi_level=false
# n_levels=2
# coarsening_factor=2.0
# (m_est,mask_save)=ICLIP_inpainting(FWD_OP,d_obs,m_evaluation,constraint,comp_grid,options,multi_level,n_levels,coarsening_factor)

SNR(in1,in2)=20*log10(norm(in1)/norm(in1-in2))

for i=1:size(m_est,1)
    figure();imshow(d_obs[i,(bkl+1):end-(bkl+1),:],cmap="gray",vmin=0.0,vmax=255.0); title("observed");
    savefig(joinpath(data_dir,string("deblurring_observed",i,".pdf")),bbox_inches="tight")
    savefig(joinpath(data_dir,string("deblurring_observed",i,".png")),bbox_inches="tight")
    figure();imshow(m_est[i,(bkl+1):end-(bkl+1),:],cmap="gray",vmin=0.0,vmax=255.0); title(string("PARSDMM, SNR=", round(SNR(vec(m_evaluation[i,(bkl+1):end-(bkl+1),:]),vec(m_est[i,(bkl+1):end-(bkl+1),:])),2)))
    savefig(joinpath(data_dir,string("PARSDMM_deblurring",i,".pdf")),bbox_inches="tight")
    savefig(joinpath(data_dir,string("PARSDMM_deblurring",i,".png")),bbox_inches="tight")
    figure();imshow(m_evaluation[i,(bkl+1):end-(bkl+1),:],cmap="gray",vmin=0.0,vmax=255.0); title("True")
    savefig(joinpath(data_dir,string("deblurring_evaluation",i,".pdf")),bbox_inches="tight")
    savefig(joinpath(data_dir,string("deblurring_evaluation",i,".png")),bbox_inches="tight")
end


file = matopen(joinpath(data_dir,"m_est.mat"), "w")
write(file, "m_est", convert(Array{Float64,3},m_est))
close(file)

file = matopen("m_evaluation.mat", "w")
write(file, "m_evaluation", convert(Array{Float64,3},m_evaluation))
close(file)

file = matopen(joinpath(data_dir,"m_train.mat"), "w")
write(file, "m_train", convert(Array{Float64,3},m_train))
close(file)

file = matopen(joinpath(data_dir,"d_obs.mat"), "w")
write(file, "d_obs", convert(Array{Float64,3},d_obs))
close(file)

file = matopen(joinpath(data_dir,"BF.mat"), "w")
write(file, "BF", convert(SparseMatrixCSC{Float64,Int64},BF))
close(file)

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
#   savefig(string("TFOCS_TV_inpainting",i,".pdf"),bbox_inches="tight")
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
# savefig(string("PARSDMM_inpainting_zoomed_1.pdf"),bbox_inches="tight")
# figure();imshow(z_parsdmm_2,cmap="gray",vmin=0.0,vmax=255.0); title("PARSDMM zoomed")
# savefig(string("PARSDMM_inpainting_zoomed_2.pdf"),bbox_inches="tight")
#
# figure();imshow(z_TFOCS_1,cmap="gray",vmin=0.0,vmax=255.0); title("BPDN-TV zoomed")
# savefig(string("TFOCS_inpainting_zoomed_1.pdf"),bbox_inches="tight")
# figure();imshow(z_TFOCS_2,cmap="gray",vmin=0.0,vmax=255.0); title("BPDN-TV zoomed")
# savefig(string("TFOCS_inpainting_zoomed_2.pdf"),bbox_inches="tight")
#
# figure();imshow(z_true_1,cmap="gray",vmin=0.0,vmax=255.0); title("true zoomed")
# savefig(string("true_inpainting_zoomed_1.pdf"),bbox_inches="tight")
# figure();imshow(z_true_2,cmap="gray",vmin=0.0,vmax=255.0); title("true zoomed")
# savefig(string("true_inpainting_zoomed_2.pdf"),bbox_inches="tight")
