export Q_update!
function Q_update!{TF<:Real,TI<:Integer}(
                  Q            ::Union{Array{TF,2},SparseMatrixCSC{TF,TI}},
                  AtA          ::Union{Vector{Array{TF,2}},Vector{SparseMatrixCSC{TF,TI}}},
                  TD_Prop,
                  rho          ::Vector{TF},
                  ind_updated  ::Vector{TI},
                  log_PARSDMM,
                  i            ::Integer,
                  Q_offsets=[]
                  )

  if isempty(ind_updated) == false
    if typeof(Q)==SparseMatrixCSC{TF,TI}
      for ii=1:length(ind_updated)
          Q = Q + ( AtA[ind_updated[ii]] )*( rho[ind_updated[ii]]-log_PARSDMM.rho[i,ind_updated[ii]] );
      end
    else
      for ii=1:length(ind_updated)
        CDS_scaled_add!(Q,AtA[ind_updated[ii]],Q_offsets,TD_Prop.AtA_offsets[ind_updated[ii]],rho[ind_updated[ii]].-log_PARSDMM.rho[i,ind_updated[ii]] )
        #for j=1:length(AtA_offsets[ind_updated[ii]])
        #  Q_update_col = findin(Q_offsets,AtA_offsets[ind_updated[ii]][j])
        #  Q[:,Q_update_col] .= Q[:,Q_update_col] .+ ( AtA[ind_updated[ii][:,AtA_offsets[ind_updated][j]]] ).*( rho[ind_updated[ii]].-log_PARSDMM.rho[i,ind_updated[ii]] );
        #end
      end
    end
  end

return Q
end
