export ICLIP_inpainting

function ICLIP_inpainting{TF<:Real}(FWD_OP::String,#::Union{String,SparseMatrixCSC{TF,Integer}},
                         d_obs::Array{TF,3},
                         m_evaluation::Array{TF,3},
                         constraint,
                         comp_grid,
                         options,
                         multi_level=false :: Bool,
                         n_levels=3 ::Integer,
                         coarsening_factor=2.0)

#intersection constrained linear inverse problem

if     TF==Float64
  TI = Int64
elseif TF==Float32
  TI = Int32
end

#always true
options.linear_inv_prob_flag = true #this is important to set

m_est=zeros(TF,size(m_evaluation))

mask_save=Vector{SparseMatrixCSC{TF,TI}}(size(d_obs,1))

#start with first image
m = d_obs[1,:,:]
m = convert(Vector{TF},vec(m))
mt=vec(m_evaluation[1,:,:])

if multi_level==false
  (P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,options.FL)
else
  #set up all required quantities for each level
  (m_levels,TD_OP_levels,AtA_levels,P_sub_levels,TD_Prop_levels,comp_grid_levels)=setup_multi_level_PARSDMM(m,n_levels,coarsening_factor,comp_grid,constraint,options)
end

if FWD_OP=="mask" && multi_level == false
  #add the observation matrix/mask for the data as a transform domain matrix
  mask = zeros(TF,length(m))
  mask[find(m)]=TF(1.0) #a 1 for observed entries
  mask_mat=spdiagm(mask,0); #create diagonal sparse matrix
  mask_mat=convert(SparseMatrixCSC{TF,TI},mask_mat)
  #also add transform-domain properties
  push!(TD_OP,mask_mat)
  push!(TD_Prop.AtA_offsets,[0])
  push!(TD_Prop.ncvx,false)
  push!(TD_Prop.banded,true)
  push!(TD_Prop.AtA_diag,true)
  push!(TD_Prop.dense,false)
  push!(TD_Prop.tag,("Lin forward OP","mask"))
  push!(TD_Prop.TD_n,comp_grid.n )

  #add a projector onto the data constraint: i.e. ||A*x-m||=< sigma, or l<=(A*x-m)<=u . m is the observed part of the image
  nonzero_ind=find(m);
  LBD=ones(TF,length(m)).*constraint["m_min"]; LBD[nonzero_ind].=m[nonzero_ind].-1.0 ;  LBD=convert(Vector{TF},LBD);
  UBD=ones(TF,length(m)).*constraint["m_max"]; UBD[nonzero_ind].=m[nonzero_ind].+1.0  ; UBD=convert(Vector{TF},UBD);
  #add the data constraint
  push!(P_sub,x -> project_bounds!(x,LBD,UBD))

elseif FWD_OP=="mask" && multi_level == true
  LBD=Vector{Vector{TF}}(n_levels)
  UBD=Vector{Vector{TF}}(n_levels)
  for i=1:n_levels
    #add the observation matrix/mask for the data as a transform domain matrix
    mask = zeros(TF,length(m_levels[i]))
    mask[find(m_levels[i])]=TF(1.0) #a 1 for observed entries
    mask_mat=spdiagm(mask,0); #create diagonal sparse matrix
    mask_mat=convert(SparseMatrixCSC{TF,TI},mask_mat)
    #also add transform-domain properties
    push!(TD_OP_levels[i],mask_mat)
    push!(TD_Prop_levels[i].AtA_offsets,[0])
    push!(TD_Prop_levels[i].ncvx,false)
    push!(TD_Prop_levels[i].banded,true)
    push!(TD_Prop_levels[i].AtA_diag,true)
    push!(TD_Prop_levels[i].dense,false)
    push!(TD_Prop_levels[i].tag,("Lin forward OP","mask"))
    push!(TD_Prop_levels[i].TD_n,comp_grid_levels[i].n )

    #add a projector onto the data constraint: i.e. ||A*x-m||=< sigma, or l<=(A*x-m)<=u . m is the observed part of the image
    nonzero_ind=find(m_levels[i]);
    LBD[i]=ones(TF,length(m_levels[i])).*constraint["m_min"]; LBD[i][nonzero_ind].=m_levels[i][nonzero_ind].-2.0 ;  LBD[i]=convert(Vector{TF},LBD[i]);
    UBD[i]=ones(TF,length(m_levels[i])).*constraint["m_max"]; UBD[i][nonzero_ind].=m_levels[i][nonzero_ind].+2.0  ; UBD[i]=convert(Vector{TF},UBD[i]);
    #add the data constraint
    #push!(P_sub_levels[i],0.0) #do this as intermediate step, somehow it doesnt work if you do it directly
    #P_sub_levels[i][end]= x -> project_bounds!(x,LBD,UBD)
    push!(P_sub_levels[i],x -> project_bounds!(x,LBD[i],UBD[i]))
    #P_sub_levels[i]=vcat

    #add new precomputed TD_OP'*TD_OP in compressed diagonal format:
    (temp_CDS_mat,offset)=mat2CDS(mask_mat'*mask_mat)
     push!(AtA_levels[i],temp_CDS_mat)
     #note that in general we also need to update the offsets of AtA[end] (in TD_Prop.AtA_offsets) but masks are always diagonal in this case so we leave it out
  end
end


mask_save[1]=mask_mat

if multi_level==false
  #precompute and distribute some stuff
  (TD_OP,AtA,l,y) = PARSDMM_precompute_distribute(m,TD_OP,TD_Prop,options)
end

println("imaging inpainting with PARSDMM serial (with many constraints):")
for i=1:size(m_evaluation,1)

  if multi_level==false
    @time (x,log_PARSDMM) = PARSDMM(m,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options);
    m_est[i,:,:].=reshape(x,comp_grid.n)
  else
    @time (x,log_PARSDMM) = PARSDMM_multi_level(m_levels,TD_OP_levels,AtA_levels,P_sub_levels,TD_Prop_levels,comp_grid_levels,options);
    m_est[i,:,:].=reshape(x,comp_grid.n)
  end

  mt=vec(m_evaluation[i,:,:])
  println("evaluation image: ",i)
  #println("relative l2 error:", norm(x-vec(mt))/norm(vec(mt)) )
  SNR(in1,in2)=20*log10(norm(in1)/norm(in1-in2))
  println("SNR:", SNR(vec(mt),x))


  #set up new mask for next image
  if (i+1)<=size(d_obs,1)
    m.=vec(d_obs[i+1,:,:])
    m=convert(Vector{TF},m)

    if FWD_OP=="mask" && multi_level == false


      #add the observation matrix/mask for the data as a transform domain matrix
      mask = zeros(TF,length(m))
      mask[find(m)]=TF(1.0) #a 1 for observed entries
      mask_mat=spdiagm(mask,0); #create diagonal sparse matrix
      mask_mat=convert(SparseMatrixCSC{TF,TI},mask_mat)
      mask_save[i+1]=mask_mat
      #also add a projector onto the data constraint: i.e. ||A*x-m||=< sigma, or l<=(A*x-m)<=u . m is the observed part of the image
      nonzero_ind=find(m);
      LBD=ones(TF,length(m)).*constraint["m_min"]; LBD[nonzero_ind].=m[nonzero_ind].-2.0 ;  LBD=convert(Vector{TF},LBD);
      UBD=ones(TF,length(m)).*constraint["m_max"]; UBD[nonzero_ind].=m[nonzero_ind].+2.0  ; UBD=convert(Vector{TF},UBD);

      #overwrite with new mask and data constraint projector
      #@spawnat P_sub.pids[end] P_sub[:L][1] = x -> project_bounds!(x,LBD,UBD)
      P_sub[end] = x -> project_bounds!(x,LBD,UBD)
      if options.parallel==true
        @spawnat TD_OP.pids[end] TD_OP[:L][1] = mask_mat
      else
         TD_OP[end]=mask_mat
      end
      #add new precomputed TD_OP'*TD_OP in compressed diagonal format:
      (temp_CDS_mat,offset)=mat2CDS(mask_mat'*mask_mat)
       AtA[end]=temp_CDS_mat
       #note that in general we also need to update the offsets of AtA[end] (in TD_Prop.AtA_offsets) but masks are always diagonal in this case so we leave it out
  elseif FWD_OP=="mask" && multi_level == true
    LBD=Vector{Vector{TF}}(n_levels)
    UBD=Vector{Vector{TF}}(n_levels)
    for j=1:n_levels
      #add the observation matrix/mask for the data as a transform domain matrix
      mask = zeros(TF,length(m_levels[j]))
      mask[find(m_levels[j])]=TF(1.0) #a 1 for observed entries
      mask_mat=spdiagm(mask,0); #create diagonal sparse matrix
      mask_mat=convert(SparseMatrixCSC{TF,TI},mask_mat)

      #add a projector onto the data constraint: i.e. ||A*x-m||=< sigma, or l<=(A*x-m)<=u . m is the observed part of the image
      nonzero_ind=find(m_levels[j]);
      LBD[j]=ones(TF,length(m_levels[j])).*constraint["m_min"]; LBD[j][nonzero_ind].=m_levels[j][nonzero_ind].-2.0 ;  LBD[j]=convert(Vector{TF},LBD[j]);
      UBD[j]=ones(TF,length(m_levels[j])).*constraint["m_max"]; UBD[j][nonzero_ind].=m_levels[j][nonzero_ind].+2.0  ; UBD[j]=convert(Vector{TF},UBD[j]);
      #add the data constraint
      P_sub_levels[j][end] = x -> project_bounds!(x,LBD[j],UBD[j])
      if options.parallel==true
        @spawnat TD_OP_levels.pids[end] TD_OP_levels[j][:L][1] = mask_mat
      else
         TD_OP_levels[j][end]=mask_mat
      end
      #add new precomputed TD_OP'*TD_OP in compressed diagonal format:
      (temp_CDS_mat,offset)=mat2CDS(mask_mat'*mask_mat)
       AtA_levels[j][end]=temp_CDS_mat
    end
end
end
  # if i==4
  #   #plot PARSDMM logs for last image
  #   figure();
  #   subplot(3, 3, 3);semilogy(log_PARSDMM.r_pri)          ;title("r primal")
  #   subplot(3, 3, 4);semilogy(log_PARSDMM.r_dual)         ;title("r dual")
  #   subplot(3, 3, 1);semilogy(log_PARSDMM.obj)            ;title(L"$ \frac{1}{2} || \mathbf{m}-\mathbf{x} ||_2^2 $")
  #   subplot(3, 3, 2);semilogy(log_PARSDMM.set_feasibility);title("TD feasibility violation")
  #   subplot(3, 3, 5);plot(log_PARSDMM.cg_it)              ;title("nr. of CG iterations")
  #   subplot(3, 3, 6);semilogy(log_PARSDMM.cg_relres)      ;title("CG rel. res.")
  #   subplot(3, 3, 7);semilogy(log_PARSDMM.rho)            ;title("rho")
  #   subplot(3, 3, 8);plot(log_PARSDMM.gamma)              ;title("gamma")
  #   subplot(3, 3, 9);semilogy(log_PARSDMM.evol_x)         ;title("x evolution")
  # end

end




return m_est,mask_save
end
