export project_subspace!

function project_subspace!(
                          x     ::Vector{TF},
                          A     ::Union{Array{TF,2},SparseMatrixCSC{Integer,TF}},
                          orth  ::Bool
                          ) where {TF<:Real}

#project a vector x onto the subspace spanned by the columns of A.
#This is a simple implementation that solves linear systems on the fly. We can
#add options to factor A'*A once and reuse if memory permits. Iterative factorization
#free methods are also an option. Orthogonalize subspace beforehand to obtain an A
# such that A'*A=I to avoid linear system solves all together in this stage.
  if orth == true
     x .= A*(A'*x) ::Vector{TF}
  else
    x .= A*((A'*A)\(A'*x)) ::Vector{TF}
  end

end

function project_subspace!(
                          x     ::Array{TF,2},
                          A     ::Union{Array{TF,2},SparseMatrixCSC{Integer,TF}},
                          orth  ::Bool,
                          mode  ::String
                          ) where {TF<:Real}
(n1,n2)=size(x)
if mode == "x" #project each x[:,i] onto the subspace, so each column of x
  if orth == true
     x .= A*(A'*x)
  else
    x .= A*((A'*A)\(A'*x))
  end
elseif mode == "y" #project each x[i,:] onto the subspace, so each row of x
  if orth == true
     x .= (A*(A'*x'))'
  else
    x .= (A*((A'*A)\(A'*x')))'
  end
end

x = vec(x)
end

function project_subspace!{TF<:Real}(
                          x     ::Array{TF,3},
                          A     ::Union{Array{TF,2},SparseMatrixCSC{Integer,TF}},
                          orth  ::Bool,
                          mode  ::String
                          )
#we want to project each slice of a tensor onto the subspace spanned by the columns of A
#x is a 3D array and we need to permute and reshape it such that we project the
#correct slices. Every slice needs to become a column of the matrix (x) we projection onto the subspace in column-wise sense

  (n1,n2,n3)=size(x)
  #permute and reshape
  if mode == "x"
    permutedims(x,[2;3;1]);
    x = reshape(x,n2*n3,n1)
  elseif mode == "y"
    permutedims(x,[1;3;2]);
    x = reshape(x,n1*n3,n2)
  elseif mode == "z"
    x = reshape(x,n1*n2,n3) #there are n3 slices of dimension n1*n2
  end

  #project every column of matrix x (a slice of the original tensor)
  if orth == true
     x .= A*(A'*x) ::Array{TF,2}
  else
    x .= A*((A'*A)\(A'*x)) ::Array{TF,2}
  end

  #permute back and vectorize
  if mode == "x"
    x = reshape(x,n2,n3,n1)
    permutedims(x,[3;1;2]);
  elseif mode == "y"
    x = reshape(x,n1,n3,n2)
    permutedims(x,[1;3;2]);
  elseif mode == "z"
    x = reshape(x,n1,n2,n3) #there are n3 slices of dimension n1*n2
  end

x = vec(x)
end
