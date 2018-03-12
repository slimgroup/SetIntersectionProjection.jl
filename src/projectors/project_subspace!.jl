export project_subspace!

function project_subspace!{TF<:Real}(
                          input ::Vector{TF},
                          A     ::Union{Array{TF,2},SparseMatrixCSC{Integer,TF}},
                          orth  ::Bool
                          )
#If we want to project each slice of a tensor onto the subspace spanned by the columns of A
#the input is a 2D array, because every slice of the tensor is already vectorized. This avoids repeated reshape operations

  if orth == true
     input .= A*(A'*input) ::Vector{TF}
  else
    input .= A*((A'*A)\(A'*input)) ::Vector{TF}
  end

end

function project_subspace!{TF<:Real}(
                          input ::Array{TF,3},
                          A     ::Union{Array{TF,2},SparseMatrixCSC{Integer,TF}},
                          orth  ::Bool,
                          mode  ::String
                          )
#we want to project each slice of a tensor onto the subspace spanned by the columns of A
#the input is a 3D array and we need to permute and reshape it such that we project the
#correct slices
if mode == "x"

elseif mode == "y"

elseif mode == "z"

end
  if orth == true
     input .= A*(A'*input) ::Vector{TF}
  else
    input .= A*((A'*A)\(A'*input)) ::Vector{TF}
  end

input = vec(input)
end
