export project_l1_Duchi!

"""
    project_l1_Duchi!(v::Union{Vector{TF},Vector{Complex{TF}}}, b::TF)
  
Projects ector onto L1 ball of specified radius.

w = ProjectOntoL1Ball(v, b) returns the vector w which is the solution
  to the following constrained minimization problem:

   min   ||w - v||_2
   s.t.  ||w||_1 <= b.

  That is, performs Euclidean projection of v to the 1-norm ball of radius
  b.

Author: John Duchi (jduchi@cs.berkeley.edu)
Translated (with some modification) to Julia 1.1 by Bas Peters
"""

function project_l1_Duchi!(v::Union{Vector{TF},Vector{Complex{TF}}}, b::TF) where {TF<:Real}
  b <= TF(0) && error("Radius of L1 ball is negative")
  norm(v, 1) <= b && return v

  lv = length(v)
  u  = similar(v)
  sv = Vector{TF}(undef, lv)

  # use RadixSort for Float32 (short keywords)
  copyto!(u, v)
  u .= abs.(u)
  u  = convert(Vector{TF},u)
  if TF==Float32
    sort!(u, rev=true, alg=RadixSort) 
  else
    u = sort!(u, rev=true, alg=QuickSort)
  end
  cumsum!(sv, u)

  # Thresholding level
  rho = 0
  while u[rho+1] > ((sv[rho+1] - b)/(rho+1)) && (rho+1) < lv
    rho += 1
  end
  rho = max(1, rho)
  theta = max.(TF(0) , (sv[rho] .- b) ./ rho)::TF

  # Projection as soft thresholding
  v .= sign.(v) .* max.(abs.(v) .- theta, TF(0))

  return v
end
