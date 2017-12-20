export project_subspace!

function project_subspace!{TF<:Real}(
                          input ::Vector{TF},
                          A     ::Union{Array{TF,2},SparseMatrixCSC{Integer,TF}},
                          orth  ::Bool
                          )
  if orth == true
     input .= A*(A'*input) ::Vector{TF}
  else
    input .= A*((A'*A)\(A'*input)) ::Vector{TF}
  end

end
