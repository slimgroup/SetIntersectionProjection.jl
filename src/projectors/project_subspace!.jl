export project_subspace!

"""
project a vector x onto the subspace spanned by the columns of A.
This is a simple implementation that solves linear systems on the fly. We can
add options to factor A' * A once and reuse if memory permits. Iterative factorization
free methods are also an option. Orthogonalize subspace beforehand to obtain an A
such that A' * A = I to avoid linear system solves all together in this stage.
"""
function project_subspace!(
                          x     ::Vector{TF},
                          A     ::Union{Array{Any},Array{TF,2}},
                          orth  ::Bool
                          ) where {TF<:Real}
  if orth == true
     x .= A*(A'*x)
  else
    x .= A*((A'*A)\(A'*x))
  end

end

function project_subspace!(
                          x     ::Array{TF,2},
                          A     ::Union{Array{Any},Array{TF,2}},
                          orth  ::Bool,
                          mode  ::Tuple{String,String}
                          ) where {TF<:Real}

(n1,n2) = deepcopy(size(x))

if mode[1] == "slice"
  error("mode[1] for project_subspace! must be: fiber")
end

if mode[2] == "x" #project each x[:,i] onto the subspace, so each column of x
  if orth == true
     x .= A*(A'*x)
  else
    x .= A*((A'*A)\(A'*x))
  end
elseif mode[2] == "z" #project each x[i,:] onto the subspace, so each row of x
  if orth == true
     x .= (A*(A'*x'))'
  else
    x .= (A*((A'*A)\(A'*x')))'
  end
else
  error("mode[2] for project_subspace! with 2D array input must be: x, or z")
end

x = vec(x)
end

"""
we want to project each slice of a tensor onto the subspace spanned by the columns of A
x is a 3D array and we need to permute and reshape it such that we project the
correct slices. Every slice needs to become a column of the matrix (x) we projection onto the subspace in column-wise sense
"""
function project_subspace!(
                          x     ::Array{TF,3},
                          A     ::Union{Array{Any},Array{TF,2}},
                          orth  ::Bool,
                          mode  ::Tuple{String,String}
                          ) where {TF<:Real}

  (n1,n2,n3) = deepcopy(size(x))

if mode[1] == "slice"
  #permute and reshape
  if mode[2] == "x"
    x = permutedims(x,[2;3;1]);
    x = reshape(x,n2*n3,n1)
  elseif mode[2] == "y"
    x = permutedims(x,[1;3;2]);
    x = reshape(x,n1*n3,n2)
  elseif mode[2] == "z"
    x = reshape(x,n1*n2,n3) #there are n3 slices of dimension n1*n2
  end

  #project every column of matrix x (a slice of the original tensor)
  if orth == true
     x .= A*(A'*x)
  else
    x .= A*((A'*A)\(A'*x))
  end

  #permute back and vectorize
  if mode[2] == "x"
    x = reshape(x,n2,n3,n1)
    x = permutedims(x,[3;1;2]);
  elseif mode[2] == "y"
    x = reshape(x,n1,n3,n2)
    x = permutedims(x,[1;3;2]);
  elseif mode[2] == "z"
    x = reshape(x,n1,n2,n3) #there are n3 slices of dimension n1*n2
  end

else
  error("for 3D models, the mode of application for project_subspace! needs to be (slice,x) or (slice,y) or (slice,z)")

end
x = vec(x)
end
