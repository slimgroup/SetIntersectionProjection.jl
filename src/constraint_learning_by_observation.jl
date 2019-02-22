export constraint_learning_by_obseration

"""
Compute matrix/image properties and return the results in a dictionary.
m is a tensor where the first index is the image index.
i.e., m[i,:,:] is one image
"""
function constraint_learning_by_obseration(comp_grid,m_train::Array{TF}) where {TF<:Real}

if TF==Float64
  TI=Int64
else
  TI=Int32
end


if length(size(m_train))==3
  (n_train_ex,t2,t3)=size(m_train)
else
  (t2,t3)=size(m_train)
  n_train_ex=1
end

#a non exhaustive list of things to learn/observe. Add some more...
observations=Dict()

observations["nuclear_norm"] = zeros(TF,n_train_ex)
observations["nuclear_Dx"]   = zeros(TF,n_train_ex)
observations["nuclear_Dz"]   = zeros(TF,n_train_ex)
observations["rank_095"]     = zeros(TI,n_train_ex)
observations["TV"]           = zeros(TF,n_train_ex)
observations["wavelet_l1"]   = zeros(TF,n_train_ex)
observations["Dx_l1"]        = zeros(TF,n_train_ex)
observations["Dz_l1"]        = zeros(TF,n_train_ex)
#observations["curvelet_l1"]  = zeros(TF,n_train_ex)
observations["DFT_l1"]       = zeros(TF,n_train_ex)
observations["DFT_card_095"] = zeros(TI,n_train_ex)
#observations["curvelet_card_095"]= zeros(TI,n_train_ex)
observations["TV_card_095"]  = zeros(TI,n_train_ex)
observations["annulus"]      = zeros(TF,n_train_ex)
observations["TV_annulus"]   = zeros(TF,n_train_ex)
observations["D_l2"]         = zeros(TF,n_train_ex)
observations["D_x_min"]      = zeros(TF,n_train_ex)
observations["D_x_max"]      = zeros(TF,n_train_ex)
observations["D_z_min"]      = zeros(TF,n_train_ex)
observations["D_z_max"]      = zeros(TF,n_train_ex)

observations["DCT_x_LB"]      = zeros(TF,t2) .+ 1e8
observations["DCT_x_UB"]      = zeros(TF,t2)

observations["DCT_y_LB"]      = zeros(TF,t3) .+ 1e8 #add because we take the min looping over every training image,
observations["DCT_y_UB"]      = zeros(TF,t3)

observations["hist_min"]      = zeros(TF,t2*t3) .+ 1e8
observations["hist_max"]      = zeros(TF,t2*t3)

observations["hist_TV_min"]   = zeros(TF,(t2-1)*t3+t2*(t3-1)) .+ 1e8
observations["hist_TV_max"]   = zeros(TF,(t2-1)*t3+t2*(t3-1))

#get transform-domain operators
#need to install CurveLab to use the Curvelet transform
(Dx_OP, dummy1, dummy2, Dx_TD_n) = get_TD_operator(comp_grid,"D_x",TF)
(Dz_OP, dummy1, dummy2, Dz_TD_n) = get_TD_operator(comp_grid,"D_z",TF)
(TV_OP, dummy1, dummy2, TD_n_TV) = get_TD_operator(comp_grid,"TV",TF)
#(C_OP, dummy1, dummy2, dummy3)   = get_TD_operator(comp_grid,"curvelet",TF)
(DFT, dummy1, dummy2, dummy3)    = get_TD_operator(comp_grid,"DFT",TF)
(DWT, dummy1, dummy2, dummy3)    = get_TD_operator(comp_grid,"wavelet",TF)

#DFT = x-> vec(fft(reshape(x,comp_grid.n)))
DCT = x-> vec(dct(reshape(x,comp_grid.n)))

DCT_x = x-> vec(dct(reshape(x,comp_grid.n),1))
DCT_y = x-> vec(dct(reshape(x,comp_grid.n),2))

for i=1:n_train_ex #can be changed to a parallel loop for larger datasets

    training_image = vec(m_train[i,:,:])

    #observe sorted values for mean histogram
    observations["hist_min"].=min.(observations["hist_min"],sort(training_image))
    observations["hist_max"].=max.(observations["hist_max"],sort(training_image))

    #observe sorted values for mean histogram
    observations["hist_TV_min"].=min.(observations["hist_TV_min"],sort(TV_OP*training_image))
    observations["hist_TV_max"].=max.(observations["hist_TV_max"],sort(TV_OP*training_image))

    #observe Nuclear norm
    sv=svdvals(reshape(training_image,comp_grid.n))
    observations["nuclear_norm"][i]=norm(sv,1)

    #observe nuclear norm in transform-domain (TV-operator, total-nuclear-variation)
    observations["nuclear_Dx"][i]=norm(svdvals(reshape(Dx_OP*training_image,Dx_TD_n)),1)
    observations["nuclear_Dz"][i]=norm(svdvals(reshape(Dz_OP*training_image,Dz_TD_n)),1)

    #observe rank that represents 95% of each image
    sv=cumsum(sv)
    sv.=sv./maximum(sv)
    observations["rank_095"][i]=findfirst(sv.>0.95)

    #observe slope in x direction
    observations["D_x_min"][i]=minimum(Dx_OP*training_image)
    observations["D_x_max"][i]=maximum(Dx_OP*training_image)

    #observe slope in z direction
    observations["D_z_min"][i]=minimum(Dz_OP*training_image)
    observations["D_z_max"][i]=maximum(Dz_OP*training_image)

    #observe anisotropic total-variation
    observations["TV"][i]=norm(TV_OP*training_image,1)

    #Wavelet l1
    observations["wavelet_l1"][i]=norm(DWT*training_image,1)

    #different direction separately
    observations["Dx_l1"][i] = norm(Dx_OP*training_image,1)
    observations["Dz_l1"][i] = norm(Dz_OP*training_image,1)

    #observe l2 norm of discrete gradients (i.e., l2 version of anisotripic TV)
    observations["D_l2"][i]=norm(TV_OP*training_image,2)

    #observe annulus
    observations["annulus"][i]=norm(vec(training_image),2)
    observations["TV_annulus"][i]=norm(TV_OP*vec(training_image),2)

    #observe curvelet domain l1 norm
    #observations["curvelet_l1"][i]=norm(C_OP*training_image,1)

    #observe Fourier domain l1 norm
    observations["DFT_l1"][i]=norm(DFT*training_image,1)

    #observe nr or required DFT atoms to represent 95%
    temp=DFT*training_image; temp=abs.(temp); temp=sort(temp); temp=cumsum(temp); temp=temp./maximum(temp);
    observations["DFT_card_095"][i]=length(temp)-findfirst(temp.>0.05)

    #observe nr or required curvelet atoms to represent95%
    # temp=C_OP*training_image; temp=abs.(temp); temp=sort(temp); temp=cumsum(temp); temp=temp./maximum(temp);
    # observations["curvelet_card_095"][i]=length(temp)-findfirst(temp.>0.05)

    #observe nr or required TV atoms to represent95%
    temp=TV_OP*training_image; temp=abs.(temp); temp=sort(temp); temp=cumsum(temp); temp=temp./maximum(temp);
    observations["TV_card_095"][i]=length(temp)-findfirst(temp.>0.05)

    #observe DCT spectrum per column (x)
    temp_img=dct(reshape(training_image,t2,t3),1);
    (temp_vals,indx)=findmin(temp_img,dims=2)
    observations["DCT_x_LB"].=min.(temp_vals[:],observations["DCT_x_LB"])
    (temp_vals,indx)=findmax(temp_img,dims=2)
    observations["DCT_x_UB"].=max.(temp_vals[:],observations["DCT_x_UB"])

    #observe DCT spectrum per row (y)
    temp_img=dct(reshape(training_image,t2,t3),2);
    (temp_vals,indx)=findmin(temp_img,dims=1)
    observations["DCT_y_LB"].=min.(temp_vals[:],observations["DCT_y_LB"])
    (temp_vals,indx)=findmax(temp_img,dims=1)
    observations["DCT_y_UB"].=max.(temp_vals[:],observations["DCT_y_UB"])
end


return observations
end
