export PARSDMM_precompute_distribute

function PARSDMM_precompute_distribute{TF<:Real,TI<:Integer}(
                                      m        ::Vector{TF},
                                      TD_OP    ::Vector{Union{SparseMatrixCSC{TF,TI},JOLI.joLinearFunction{TF,TF}}},
                                      TD_Prop,
                                      options
                                      )

const N           = length(m)

#add the identity matrix as the operator for the squared distance from the point we want to project
if options.linear_inv_prob_flag==false
  push!(TD_OP,convert(SparseMatrixCSC{TF,TI},speye(TF,N)));
  push!(TD_Prop.AtA_offsets,[0])
  push!(TD_Prop.banded,true)
  push!(TD_Prop.AtA_diag,true)
  push!(TD_Prop.dense,false)
  push!(TD_Prop.tag,("distance squared","identity"))
end


const p     = length(TD_OP);

AtA=Vector{SparseMatrixCSC{TF,TI}}(p)
for i=1:p
  if TD_Prop.dense[i]==true
    if TD_Prop.AtA_diag[i]==true
      AtA[i]=speye(N)
    else
      error("provided a dense non orthogoal transform domaian operator")
    end
  else
    AtA[i] = TD_OP[i]'*TD_OP[i]
  end
end

#if all AtA are banded -> convert to CDS (DIA) format
if sum(TD_Prop.banded[1:p].=true)==p
  for i=1:p
    (AtA[i],TD_Prop.AtA_offsets[i])  = mat2CDS(AtA[i])
    TD_Prop.AtA_offsets[i]=convert(Vector{TI},TD_Prop.AtA_offsets[i])
  end
   AtA=convert(Vector{Array{TF,2}},AtA);
  TD_Prop.AtA_offsets=TD_Prop.AtA_offsets[1:p]
end

#allocate arrays of vectors
y       = Vector{Vector{TF}}(p);
l       = Vector{Vector{TF}}(p);

for ii=1:p #initialize all rho's, gamma's, y's and l's
    y[ii]       = zeros(size(TD_OP[ii],1))
    ly          = length(y[ii])
    l[ii]       = zeros(ly);#0.*y[ii];
end

if options.parallel==true
  TD_OP = distribute(TD_OP)
  y     = distribute(y)
  l     = distribute(l)
end
return TD_OP,AtA,l,y
end
