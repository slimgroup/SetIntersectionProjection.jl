export project_nuclear!

function project_nuclear!(
                          x     ::Array{TF,2},
                          sigma ::TF,
                          mode  #not an option for matrix inputs
                          ) where {TF<:Real}
"""
Project the matrix onto the set of matrices with nuclear norm less then or equal to sigma
outputs a vector. Uses singular value decomposition
"""

  #slow reference, like matlab
  #(U,S,V) = svd(x)
  #x = U[:,1:r]*diagm(S)*(V[:,1:r])'

  #F=svd!(x)
  F = svd(x) #obtain svd
  project_l1_Duchi!(F.S, sigma) # project singular values onto the l1-ball of size sigma

  #copy!(x,F.U * diagm(F.S) * F.Vt) #reconstruct
  x .= F.U * diagm(0 => F.S) * F.Vt #reconstruct
  x = vec(x)
  #return vec(x)
end

function project_nuclear!(
                          x     ::Array{TF,3},
                          sigma ::TF,
                          mode  ::Tuple{String,String}
                          ) where {TF<:Real}
"""
Project each slice of the tensor (x,y or z) onto the set of matrices with nuclear norm less then or equal to sigma
outputs a vector. Uses singular value decomposition
"""
if mode[1] == "slice"
  if mode[2] == "x" #project y-z slices, so one slice per gridpoint in the x direction
    Threads.@threads for i=1:size(x,1)
      F=svd(view(x,i,:,:))
      project_l1_Duchi!(F.S, sigma) # project singular values onto the l1-ball of size sigma
      @inbounds x[i,:,:] .= F.U * diagm(0 => F.S) * F.Vt #reconstruct
    end
  elseif mode[2] == "y"
    Threads.@threads for i=1:size(x,2)
      F=svd(view(x,:,i,:))
      project_l1_Duchi!(F.S, sigma) # project singular values onto the l1-ball of size sigma
      @inbounds x[:,i,:].= F.U * diagm(0 => F.S) * F.Vt
    end
  elseif mode[2] == "z"
    Threads.@threads for i=1:size(x,3)
      F=svd(view(x,:,:,i))
      project_l1_Duchi!(F.S, sigma) # project singular values onto the l1-ball of size sigma
      @inbounds x[:,:,i].= F.U * diagm(0 => F.S) * F.Vt
    end
  end #end if mode[2]
else
 error("mode[1] for nuclear norm projections can only be: slice")
end #if mode[1]

x=vec(x)
end
