export project_rank!

function project_rank!(
                      x :: Array{TF,2},
                      r :: TI,
                      mode :: String #not an option for matrices
                      ) where {TF<:Real,TI<:Integer}
"""
Project the matrix X onto the set of rank-r ( need r<min(n1,n2) ) matrices
outputs a vector. Uses singular value decomposition
"""
  F  = svd(x)
  x .= F.U[:,1:r] * diagm(F.S[1:r])* F.Vt[1:r,:]
  x  = vec(x)
end

function project_rank!(
                      x :: Array{TF,3},
                      r :: TI,
                      mode ::String
                      ) where {TF<:Real,TI<:Integer}
"""
Project each slice (x,y or z) of the tensor (3D model) onto the set of rank-r ( need r<min(n1,n2) ) matrices
outputs a vector. Uses singular value decomposition
"""

if mode == "x" #project y-z slices, so one slice per gridpoint in the x direction
  Threads.@threads for i=1:size(x,1)
    F = svd(view(x,i,:,:))
    @inbounds x[i,:,:] .= F.U[:,1:r] * diagm(F.S[1:r])* F.Vt[1:r,:]
  end
elseif mode == "y"
  Threads.@threads for i=1:size(x,2)
    F = svd(view(x,:,i,:))
    @inbounds x[:,i,:] .= F.U[:,1:r] * diagm(F.S[1:r])* F.Vt[1:r,:]
  end
elseif mode == "z"
  Threads.@threads for i=1:size(x,3)
    F=svd(view(x,:,:,i))
    @inbounds x[:,:,i] .= F.U[:,1:r] * diagm(F.S[1:r])* F.Vt[1:r,:]
  end
end #end if

  x=vec(x)
end
