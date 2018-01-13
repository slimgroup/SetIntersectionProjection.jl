export setup_constraints

function setup_constraints(constraint,comp_grid,TF)
# input: constraint: structure with information about which constraints to use and their specifications
#      : comp_grid: structure with computational grid information; comp_grid.n = [x z] number of gridpoints in each direction; comp_grid.d = [d1 d2] spacing between gridpoints in each direction.
#      : FL : Floating point precision (either 32 or 64)
# output : P :
#        : P_sub
#        : TD_OP
# Bas Peters

if    TF == Float64
  TI = Int64
elseif TF == Float32
  TI = Int32
end

[@spawnat pid comp_grid for pid in workers()]

#convert all integers and floats to the desired precision
keylist=collect(keys(constraint))
for i=1:length(keylist)
  if typeof(constraint[keylist[i]])<:Real && length(constraint[keylist[i]])==1 && ((typeof(constraint[keylist[i]])<:Integer)==false) #convert floats
    constraint[keylist[i]]=convert(TF,constraint[keylist[i]])
  elseif typeof(constraint[keylist[i]])<:Vector #convert vectors
    constraint[keylist[i]]=convert(Vector{TF},constraint[keylist[i]])
  elseif typeof(constraint[keylist[i]])<:Integer  #convert integers
    constraint[keylist[i]]=convert(TI,constraint[keylist[i]])
  end
end

#allocate
P_sub = Vector{Any}(99)
TD_OP = Vector{Union{SparseMatrixCSC{TF,TI},JOLI.joLinearFunction{TF,TF}}}(99);
AtA   = Vector{SparseMatrixCSC{TF,TI}}(99);

TD_Prop=transform_domain_properties(zeros(99),zeros(99),zeros(99),Vector{Tuple{TI,TI}}(99),Vector{Tuple{String,String}}(99),zeros(99),Vector{Vector{TI}}(99))

special_operator_list = ["DFT" "DCT"]
N =  prod(comp_grid.n)
counter=1;

# bound constraints
if haskey(constraint,"use_bounds") && constraint["use_bounds"] == true
    if length(constraint["m_max"])==1 # test if the input is scalar
        constraint["m_min"]=convert(TF,constraint["m_min"])
        constraint["m_max"]=convert(TF,constraint["m_max"])
        (LB,UB)=get_bound_constraints(comp_grid,constraint)
    else #use per grid point provided upper and lower bound models
        LB = constraint["m_min"];
        UB = constraint["m_max"];
    end
    P_sub[counter]        = x -> project_bounds!(x,LB,UB)
    TD_OP[counter]        = convert(SparseMatrixCSC{TF,TI},speye(TF,N))
    TD_Prop.ncvx[counter] = false ;TD_Prop.AtA_diag[counter] = true ; TD_Prop.dense[counter] = false ; TD_Prop.TD_n[counter] = comp_grid.n ; TD_Prop.banded[counter] = true
    TD_Prop.tag[counter]  = ("bounds","identity")
    counter               = counter+1;
end

#loop for 1 to 3, to allow for 3 different instances of the same type of constraints,
#one corresponding to each physical dimension

#Transform domain bound-constraints & slope constraints
for i=1:3
  if haskey(constraint,string("use_TD_bounds_",i)) && constraint[string("use_TD_bounds_",i)]==true
    (P_sub, TD_OP, TD_Prop) = setup_transform_domain_bound_constraints(counter,P_sub,TD_OP,TD_Prop,comp_grid,constraint[string("TDB_operator_",i)],constraint[string("TD_LB_",i)],constraint[string("TD_UB_",i)],TF,special_operator_list)
    TD_Prop.ncvx[counter]   = false
    TD_Prop.tag[counter]    = (string("TD-bounds ",i), constraint[string("TDB_operator_",i)])
    counter                 = counter+1;
  end
end

#Transform domain bounds constraints for each column/row/fiber for a 2D or 3D model
for i in ["x","y","z"]
  if haskey(constraint,string("use_TD_bounds_fiber_",i)) && constraint[string("use_TD_bounds_fiber_",i)]== true
    (P_sub, TD_OP, TD_Prop) = setup_transform_domain_bound_constraints_fiber(i,counter,P_sub,TD_OP,TD_Prop,comp_grid,constraint[string("TD_bounds_fiber_",i,"_operator")],constraint[string("TD_LB_fiber_",i)],constraint[string("TD_UB_fiber_",i)],TF,special_operator_list)
    TD_Prop.ncvx[counter]   = false
    TD_Prop.tag[counter]    = ( string("TD_bounds_fiber_",i) , constraint[string("TD_bounds_fiber_",i,"_operator")] )
    counter                 = counter+1;
  end
end

#Transform domain l1-norm constraints (including total-variation)
for i=1:3
  if haskey(constraint,string("use_TD_l1_",i)) && constraint[string("use_TD_l1_",i)]==true
    (P_sub, TD_OP, TD_Prop)=setup_transform_domain_l1_constraints(counter,P_sub,TD_OP,TD_Prop,comp_grid,constraint[string("TD_l1_operator_",i)],constraint[string("TD_l1_sigma_",i)],TF,special_operator_list)
    TD_Prop.ncvx[counter]= false
    TD_Prop.tag[counter]  = (string("TDl1 ",i), constraint[string("TD_l1_operator_",i)])
    counter              = counter+1;
  end
end

#Transform domain l2-norm constraints
for i=1:3
  if haskey(constraint,string("use_TD_l2_",i)) && constraint[string("use_TD_l2_",i)]==true
    (P_sub, TD_OP, TD_Prop) = setup_transform_domain_l2_constraints(counter,P_sub,TD_OP,TD_Prop,comp_grid,constraint[string("TD_l2_operator_",i)],constraint[string("TD_l2_sigma_",i)],TF,special_operator_list)
    TD_Prop.ncvx[counter]   = false
    TD_Prop.tag[counter]    = (string("TDl2 ",i), constraint[string("TD_l2_operator_",i)])
    counter                 = counter+1;
  end
end

#Transform domain annulus constraints
for i=1:3
  if haskey(constraint,string("use_TD_annulus_",i)) && constraint[string("use_TD_annulus_",i)]==true
    (P_sub, TD_OP, TD_Prop) = setup_transform_domain_annulus_constraints(counter,P_sub,TD_OP,TD_Prop,comp_grid,constraint[string("TD_annulus_operator_",i)],constraint[string("TD_annulus_sigma_min_",i)],constraint[string("TD_annulus_sigma_max_",i)],TF,special_operator_list)
    TD_Prop.ncvx[counter]   = true
    TD_Prop.tag[counter]    = (string("TD_annulus ",i), constraint[string("TD_annulus_operator_",i)])
    counter                 = counter+1;
  end
end

#Transform domain cardinality constraints (global for the full computational domain)
for i=1:3
  if haskey(constraint,string("use_TD_card_",i)) && constraint[string("use_TD_card_",i)]== true
    (P_sub, TD_OP, TD_Prop) = setup_transform_domain_card_constraints(counter,P_sub,TD_OP,TD_Prop,comp_grid,constraint[string("TD_card_operator_",i)],constraint[string("card_",i)],TF,special_operator_list)
    TD_Prop.ncvx[counter]   = true
    TD_Prop.tag[counter]    = (string("card_",i), constraint[string("TD_card_operator_",i)])
    counter                 = counter+1;
  end
end

#Transform domain cardinality constraints for each column/row/fiber for a 2D or 3D model
for i in ["x","y","z"]
  if haskey(constraint,string("use_TD_card_fiber_",i)) && constraint[string("use_TD_card_fiber_",i)]== true
    (P_sub, TD_OP, TD_Prop) = setup_transform_domain_card_constraints_fiber(i,counter,P_sub,TD_OP,TD_Prop,comp_grid,constraint[string("TD_card_fiber_",i,"_operator")],constraint[string("card_fiber_",i)],TF,special_operator_list)
    TD_Prop.ncvx[counter]   = true
    TD_Prop.tag[counter]    = ( string("TD_card_fiber_",i) , constraint[string("TD_card_fiber_",i,"_operator")] )
    counter                 = counter+1;
  end
end

#subspace constraints
if haskey(constraint,"use_subspace") && constraint["use_subspace"]== true
  P_sub[counter]            = x -> project_subspace!(x,constraint["A"],constraint["subspace_orthogonal"])
  TD_Prop.ncvx[counter]     = false
  TD_OP[counter]            = convert(SparseMatrixCSC{TF,TI},speye(TF,N))
  TD_Prop.AtA_diag[counter] = true
  TD_Prop.dense[counter]    = false
  TD_Prop.banded[counter]   = true
  TD_Prop.TD_n[counter]     = comp_grid.n
  TD_Prop.tag[counter]      = ("subspace", "identity")
  counter                   = counter+1;
end

#Nuclear norm constraints for a matrix
for i=1:3
  if  haskey(constraint,string("use_TD_nuclear_",i)) && constraint[string("use_TD_nuclear_",i)]== true
    if length(comp_grid.n)==3 && comp_grid.n[3]>1 #test if provided model is 3D
       error("provided model is 3D, select 'use_nuclear_slice' instead of 'use_nuclear' ")
    end
    (P_sub, TD_OP, TD_Prop)=setup_transform_domain_nuclear_constraints(counter,P_sub,TD_OP,TD_Prop,comp_grid,constraint[string("TD_nuclear_operator_",i)],constraint[string("TD_nuclear_norm_",i)],TF,special_operator_list)
    TD_Prop.ncvx[counter]= false
    TD_Prop.tag[counter]  = (string("TD_nuclear_",i), constraint[string("TD_nuclear_operator_",i)])
    counter              = counter+1;
  end
end

#Nuclear norm constraints for each slice (x-y,x-z or y-z) from a 3D model
for i in ["x","y","z"]
  if haskey(constraint,string("use_nuclear_slice_",i)) && constraint[string("use_nuclear_slice_",i)]== true
    if length(comp_grid.n)==2 | comp_grid.n[3]==1 #test if provided model is 3D
       error("provided model is 2D, select 'use_nuclear' instead of 'use_nuclear_slice' ")
    end
    P_sub[counter]            = x -> project_nuclear!(reshape(x,comp_grid.n),constraint[string("nuclear_norm_slice_",i)],i)
    TD_Prop.ncvx[counter]     = false
    TD_OP[counter]            = convert(SparseMatrixCSC{TF,TI},speye(TF,N))
    TD_Prop.AtA_diag[counter] = true
    TD_Prop.dense[counter]    = false
    TD_Prop.banded[counter]   = true
    TD_Prop.TD_n[counter]     = comp_grid.n
    TD_Prop.tag[counter]      = (string("nuclear_norm_slice_",i), "identity")
    counter                   = counter+1;
  end
end

#rank constraint for 2D matrix
for i=1:3
  if haskey(constraint,string("use_TD_rank_",i)) && constraint[string("use_TD_rank_",i)]== true
    if length(comp_grid.n)==3 && comp_grid.n[3]>1 #test if provided model is 3D
       error("provided model is 3D, select 'use_rank_slice' instead of 'use_rank' ")
    end
    (P_sub, TD_OP, TD_Prop)=setup_transform_domain_rank_constraints(counter,P_sub,TD_OP,TD_Prop,comp_grid,constraint[string("TD_rank_operator_",i)],constraint[string("TD_max_rank_",i)],TF,special_operator_list)
    TD_Prop.ncvx[counter]= true
    TD_Prop.tag[counter]  = (string("TD_rank_",i), constraint[string("TD_rank_operator_",i)])
    counter              = counter+1;
  end
end

#rank constraint each slice of a 3D matrix
for i in ["x","y","z"]
  if haskey(constraint,string("use_rank_slice_",i)) && constraint[string("use_rank_slice_",i)]== true
    if length(comp_grid.n)==2 | comp_grid.n[3]==1 #test if provided model is 3D
       error("provided model is 2D, select 'use_rank' instead of 'use_rank_slice' ")
    end
    P_sub[counter]            = x -> project_rank!(reshape(x,comp_grid.n),constraint[string("max_rank_slice_",i)],i)
    TD_Prop.ncvx[counter]     = true
    TD_OP[counter]            = convert(SparseMatrixCSC{TF,TI},speye(TF,N))
    TD_Prop.AtA_diag[counter] = true
    TD_Prop.dense[counter]    = false
    TD_Prop.banded[counter]   = true
    TD_Prop.TD_n[counter]     = comp_grid.n
    TD_Prop.tag[counter]      = (string("use_rank_slice_",i), "identity")
    counter                   = counter+1;
  end
end

P_sub = P_sub[1:counter-1]
TD_OP = TD_OP[1:counter-1]

TD_Prop.ncvx        = TD_Prop.ncvx[1:counter-1]
TD_Prop.AtA_diag    = TD_Prop.AtA_diag[1:counter-1]
TD_Prop.dense       = TD_Prop.dense[1:counter-1]
TD_Prop.TD_n        = TD_Prop.TD_n[1:counter-1]
TD_Prop.tag         = TD_Prop.tag[1:counter-1]
TD_Prop.banded      = TD_Prop.banded[1:counter-1]
TD_Prop.AtA_offsets = TD_Prop.AtA_offsets[1:counter-1]

return P_sub,TD_OP,TD_Prop
end #end setup_constraints

function setup_transform_domain_bound_constraints(ind,P_sub,TD_OP,TD_Prop,comp_grid,operator_type,TD_LB,TD_UB,TF,special_operator_list)
  if length(TD_UB)==1
    TD_UB=convert(TF,TD_UB); TD_LB=convert(TF,TD_LB)
  else
    TD_UB=convert(Vector{TF},TD_UB); TD_LB=convert(Vector{TF},TD_LB)
  end
  (A,AtA_diag,dense,TD_n,banded)= get_TD_operator(comp_grid,operator_type,TF)

  if operator_type in special_operator_list
    if operator_type=="DCT"
    else
      error("temporality no support for bound constraints in a transform domain that results in complex coefficients (can only choose DCT for now)")
    end
    if TF==Float64; TI=Int64; else; TI=Int32; end
    TD_OP[ind]              = convert(SparseMatrixCSC{TF,TI},speye(TF,prod(comp_grid.n)))
    TD_Prop.AtA_diag[ind]   = true
    TD_Prop.dense[ind]      = false
    TD_Prop.TD_n[ind]       = comp_grid.n
    TD_Prop.banded[ind]     = true
    P_sub[ind]              = x -> copy!(x,A'*project_bounds!(A*x,TD_LB,TD_UB))
    TD_OP[ind]              = convert(SparseMatrixCSC{TF,TI},speye(TF,prod(comp_grid.n)))
  else
    P_sub[ind]              =  x -> project_bounds!(x,TD_LB,TD_UB)
    TD_OP[ind]              = A
    TD_Prop.AtA_diag[ind]   = AtA_diag
    TD_Prop.dense[ind]      = dense
    TD_Prop.TD_n[ind]       = TD_n
    TD_Prop.banded[ind]     = banded
  end
  return P_sub, TD_OP, TD_Prop
end

function setup_transform_domain_bound_constraints_fiber(mode,ind,P_sub,TD_OP,TD_Prop,comp_grid,operator_type,LB,UB,TF,special_operator_list)

  if operator_type in special_operator_list
    if operator_type=="DCT"
      # #get 1D DCT
      # if mode=="x"
      #   A = joDCT(convert(Int64,comp_grid.n[1]);planned=false,DDT=TF,RDT=TF)
      # elseif mode=="y"
      #   A = joDCT(convert(Int64,comp_grid.n[2]);planned=false,DDT=TF,RDT=TF)
      # elseif mode=="z"
      #   A = joDCT(convert(Int64,comp_grid.n[3]);planned=false,DDT=TF,RDT=TF)
      # end
    else
      error("temporality no support for bound constraints in a transform domain that results in complex coefficients (can only choose DCT for now)")
    end
    if TF==Float64; TI=Int64; else; TI=Int32; end
    TD_OP[ind]              = convert(SparseMatrixCSC{TF,TI},speye(TF,prod(comp_grid.n)))
    TD_Prop.AtA_diag[ind]   = true
    TD_Prop.dense[ind]      = false
    TD_Prop.TD_n[ind]       = comp_grid.n
    TD_Prop.banded[ind]     = true
    #P_sub[ind]              = x -> copy!(x,A'*project_bounds!(A*reshape(x,comp_grid.n),TD_LB,TD_UB))
    if mode=="x"; coord=1; elseif  mode=="y"; coord=2; elseif mode=="z"; coord=3; end
    P_sub[ind]               = x -> copy!(x,vec(idct(project_bounds!(dct(reshape(x,comp_grid.n),coord),TD_LB,TD_UB,mode),coord)))
    TD_OP[ind]              = convert(SparseMatrixCSC{TF,TI},speye(TF,prod(comp_grid.n)))
  else
    (A,AtA_diag,dense,TD_n,banded)  = get_TD_operator(comp_grid,operator_type,TF)
    P_sub[ind]              =  x -> project_bounds!(reshape(x,TD_n),LB,UB,mode)
    TD_OP[ind]              = A
    TD_Prop.AtA_diag[ind]   = AtA_diag
    TD_Prop.dense[ind]      = dense
    TD_Prop.TD_n[ind]       = TD_n
    TD_Prop.banded[ind]     = banded
  end
  return P_sub, TD_OP, TD_Prop
end

function setup_transform_domain_nuclear_constraints(ind,P_sub,TD_OP,TD_Prop,comp_grid,operator_type,nuclear_norm,TF,special_operator_list)
    (A,AtA_diag,dense,TD_n,banded)= get_TD_operator(comp_grid,operator_type,TF)
    P_sub[ind]              = x -> project_nuclear!(reshape(x,TD_n),nuclear_norm)
    TD_OP[ind]              = A
    TD_Prop.AtA_diag[ind]   = AtA_diag
    TD_Prop.dense[ind]      = dense
    TD_Prop.TD_n[ind]       = TD_n
    TD_Prop.banded[ind]     = banded
  return P_sub, TD_OP, TD_Prop
end

function setup_transform_domain_rank_constraints(ind,P_sub,TD_OP,TD_Prop,comp_grid,operator_type,max_rank,TF,special_operator_list)
    if operator_type in ["TV" "D2D" "D3D"]
      error("TV, D2D or D3D not available for transform-domain rank constraint, use two or three separate constraints with different derivatives (Dx,Dy,Dz)")
    end
    (A,AtA_diag,dense,TD_n,banded)= get_TD_operator(comp_grid,operator_type,TF)
    P_sub[ind]              = x -> project_rank!(reshape(x,TD_n),max_rank)
    TD_OP[ind]              = A
    TD_Prop.AtA_diag[ind]   = AtA_diag
    TD_Prop.dense[ind]      = dense
    TD_Prop.TD_n[ind]       = TD_n
    TD_Prop.banded[ind]     = banded
  return P_sub, TD_OP, TD_Prop
end

function setup_transform_domain_l1_constraints(ind,P_sub,TD_OP,TD_Prop,comp_grid,operator_type,sigma,TF,special_operator_list)
  (A,AtA_diag,dense,TD_n,banded)= get_TD_operator(comp_grid,operator_type,TF)
  #P_sub[ind]         =  x -> projnorm1(x,convert(Float64,sigma))

  if operator_type in special_operator_list
    #P_sub[ind]              =  x -> (x=A*x; project_l1_Duchi!(x,sigma); x=A'*x; return x)
    P_sub[ind]              =  x -> copy!(x,A'*project_l1_Duchi!(A*x,sigma))
    if TF==Float64; TI=Int64; else; TI=Int32; end
    TD_OP[ind]              = convert(SparseMatrixCSC{TF,TI},speye(TF,prod(comp_grid.n)))
    TD_Prop.AtA_diag[ind]   = true
    TD_Prop.dense[ind]      = false
    TD_Prop.TD_n[ind]       = comp_grid.n
    TD_Prop.banded[ind]     = true
  else
    P_sub[ind]              = x -> project_l1_Duchi!(x,sigma)
    TD_OP[ind]              = A
    TD_Prop.AtA_diag[ind]   = AtA_diag
    TD_Prop.dense[ind]      = dense
    TD_Prop.TD_n[ind]       = TD_n
    TD_Prop.banded[ind]     = banded
  end
  return P_sub, TD_OP, TD_Prop
end

function setup_transform_domain_l2_constraints(ind,P_sub,TD_OP,TD_Prop,comp_grid,operator_type,sigma,TF,special_operator_list)
  (A,AtA_diag,dense,TD_n,banded)= get_TD_operator(comp_grid,operator_type,TF)

  if operator_type in special_operator_list
    P_sub[ind]              =  x -> copy!(x,A'*project_l2!(A*x,sigma))
    if TF==Float64; TI=Int64; else; TI=Int32; end
    TD_OP[ind]              = convert(SparseMatrixCSC{TF,TI},speye(TF,prod(comp_grid.n)))
    TD_Prop.AtA_diag[ind]   = true
    TD_Prop.dense[ind]      = false
    TD_Prop.TD_n[ind]       = comp_grid.n
    TD_Prop.banded[ind]     = true
  else
    P_sub[ind]              =  x -> project_l2!(x,sigma)
    TD_OP[ind]              = A
    TD_Prop.AtA_diag[ind]   = AtA_diag
    TD_Prop.dense[ind]      = dense
    TD_Prop.TD_n[ind]       = TD_n
    TD_Prop.banded[ind]     = banded
  end
  return P_sub, TD_OP, TD_Prop
end

function setup_transform_domain_annulus_constraints(ind,P_sub,TD_OP,TD_Prop,comp_grid,operator_type,sigma_min,sigma_max,TF,special_operator_list)
  (A,AtA_diag,dense,TD_n,banded)= get_TD_operator(comp_grid,operator_type,TF)

  if operator_type in special_operator_list
    P_sub[ind]              =  x -> copy!(x,A'*project_annulus!(A*x,sigma_min,sigma_max))
    if TF==Float64; TI=Int64; else; TI=Int32; end
    TD_OP[ind]              = convert(SparseMatrixCSC{TF,TI},speye(TF,prod(comp_grid.n)))
    TD_Prop.AtA_diag[ind]   = true
    TD_Prop.dense[ind]      = false
    TD_Prop.TD_n[ind]       = comp_grid.n
    TD_Prop.banded[ind]     = true
  else
    P_sub[ind]              =  x -> project_annulus!(x,sigma_min,sigma_max)
    TD_OP[ind]              = A
    TD_Prop.AtA_diag[ind]   = AtA_diag
    TD_Prop.dense[ind]      = dense
    TD_Prop.TD_n[ind]       = TD_n
    TD_Prop.banded[ind]     = banded
  end
  return P_sub, TD_OP, TD_Prop
end

function setup_transform_domain_card_constraints(ind,P_sub,TD_OP,TD_Prop,comp_grid,operator_type,k,TF,special_operator_list)
  (A,AtA_diag,dense,TD_n,banded)= get_TD_operator(comp_grid,operator_type,TF)

  if operator_type in special_operator_list
    P_sub[ind]              =  x -> copy!(x,A'*project_cardinality!(A*x,convert(Integer,k)))
    if TF==Float64; TI=Int64; else; TI=Int32; end
    TD_OP[ind]              = convert(SparseMatrixCSC{TF,TI},speye(TF,prod(comp_grid.n)))
    TD_Prop.AtA_diag[ind]   = true
    TD_Prop.dense[ind]      = false
    TD_Prop.TD_n[ind]       = comp_grid.n
    TD_Prop.banded[ind]     = true
  else
    P_sub[ind]              =  x -> project_cardinality!(x,convert(Integer,k))
    TD_OP[ind]              = A
    TD_Prop.AtA_diag[ind]   = AtA_diag
    TD_Prop.dense[ind]      = dense
    TD_Prop.TD_n[ind]       = TD_n
    TD_Prop.banded[ind]     = banded
  end
  return P_sub, TD_OP, TD_Prop
end

function setup_transform_domain_card_constraints_fiber(mode,ind,P_sub,TD_OP,TD_Prop,comp_grid,operator_type,k,TF,special_operator_list)

  if operator_type in special_operator_list
    error("temporality no support for cardinality constraints in a transform domain on tensor fibers")
  else
    (A,AtA_diag,dense,TD_n,banded)  = get_TD_operator(comp_grid,operator_type,TF)
    P_sub[ind]              =  x -> project_cardinality!(reshape(x,TD_n),convert(Integer,k),mode)
    TD_OP[ind]              = A
    TD_Prop.AtA_diag[ind]   = AtA_diag
    TD_Prop.dense[ind]      = dense
    TD_Prop.TD_n[ind]       = TD_n
    TD_Prop.banded[ind]     = banded
  end
  return P_sub, TD_OP, TD_Prop
end
