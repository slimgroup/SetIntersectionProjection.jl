export Q_update!

"""
update the Q matrix in SparseMatrixCSC or CDS format. Q=sum_{i=1}^p AtA[i]rho[i]
if any of the rho[i] change
"""
function Q_update!(
                  Q            ::Union{Array{TF,2},SparseMatrixCSC{TF,Int64},joAbstractLinearOperator{TF, TF}},
                  AtA          ::Union{Vector{Array{TF, 2}}, Vector{SparseMatrixCSC{TF, TI}}, Vector{joAbstractLinearOperator{TF, TF}}, Vector{Union{Array{TF, 2}, SparseMatrixCSC{TF, TI}, joAbstractLinearOperator{TF, TF}}}},
                  set_Prop,
                  rho          ::Vector{TF},
                  ind_updated  ::Vector{TI},
                  log_PARSDMM,
                  i            ::Integer,
                  Q_offsets=[] ::Vector{TI}
                  ) where {TF<:Real,TI<:Integer}

  if isempty(ind_updated) == false
    if typeof(Q) <: joAbstractLinearOperator #some of the AtA are JOLI operators
      #JOLI operators cannot be defined recursively for many iterations for many operators
      #because this causes memory issues and slow performance -> re-assemble Q from scratch
      # for ii=1:length(ind_updated)
      #   if typeof(AtA[ind_updated[ii]]) <: joAbstractLinearOperator
      #     Q = Q + ( AtA[ind_updated[ii]] )*( rho[ind_updated[ii]]-log_PARSDMM.rho[i,ind_updated[ii]] );
      #   else
      #     Q = Q + ( joMatrix(AtA[ind_updated[ii]]) )*( rho[ind_updated[ii]]-log_PARSDMM.rho[i,ind_updated[ii]] );
      #   end
      # end
      if typeof(AtA[1]) <: joAbstractLinearOperator
        Q = rho[1]*AtA[1]
      else
        Q = rho[1]*joMatrix(AtA[1])
      end
      for i=2:length(AtA)
        if typeof(AtA[i]) <: joAbstractLinearOperator
          Q = Q + rho[i]*AtA[i]
        else
          Q = Q + rho[i]*joMatrix(AtA[i])
        end
      end
    elseif typeof(Q) == SparseMatrixCSC{TF,Int64} #all AtA are sparse arrays
      for ii=1:length(ind_updated)
          Q = Q + ( AtA[ind_updated[ii]] )*( rho[ind_updated[ii]]-log_PARSDMM.rho[i,ind_updated[ii]] );
      end
    else #all AtA are in CDS format
      for ii=1:length(ind_updated)
        CDS_scaled_add!(Q,AtA[ind_updated[ii]],Q_offsets,set_Prop.AtA_offsets[ind_updated[ii]],rho[ind_updated[ii]] .- log_PARSDMM.rho[i,ind_updated[ii]] )
      end
    end
  end

return Q
end
