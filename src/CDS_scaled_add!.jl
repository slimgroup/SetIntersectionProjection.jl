export CDS_scaled_add!


"""
 Computes A = A + alpha * B for A and B in the compressed diagonal storage format (CDS/DIA)
 TO DO: make this function multi-threaded per column of A and B, or use BLAS functions
"""
function CDS_scaled_add!(
                  A           ::Array{TF,2},
                  B           ::Array{TF,2},
                  A_offsets   ::Vector{TI},
                  B_offsets   ::Vector{TI},
                  alpha       ::TF
                  ) where {TF<:Real,TI<:Integer}

for k=1:length(B_offsets)
  A_update_col = findall((in)(B_offsets[k]),A_offsets)
  if isempty(A_update_col) == true
    error("attempted to update a diagonal in A in CDS storage that does not exist. A and B need to have the same nonzero diagonals")
  end
  A[:,A_update_col] .= A[:,A_update_col] .+ alpha .* B[:,k];
end



end
