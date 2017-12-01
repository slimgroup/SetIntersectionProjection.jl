export project_nuclear!

function project_nuclear!{TF<:Real}(
                        x     ::Array{TF,2},
                        sigma ::TF
                        )
"""
Project the matrix onto the set of matrices with nuclear norm less then or equal to sigma
outputs a vector. Uses singular value decomposition
"""

  #slow reference, like matlab
  #(U,S,V) = svd(x)
  #x = U[:,1:r]*diagm(S)*(V[:,1:r])'

  #F=svdfact!(x)
  F = svdfact(x) #obtain svd
  project_l1_Duchi!(F.S, sigma) # project singular values onto the l1-ball of size sigma

    #copy!(x,F.U * diagm(F.S) * F.Vt) #reconstruct
    x.=F.U * diagm(F.S) * F.Vt #reconstruct
    x=vec(x)
    #return vec(x)
end

function project_nuclear!{TF<:Real}(
                        x     ::Array{TF,3},
                        sigma ::TF,
                        mode ::String
                        )
"""
Project each slice of the tensor (x,y or z) onto the set of matrices with nuclear norm less then or equal to sigma
outputs a vector. Uses singular value decomposition
"""
if mode == "x" #project y-z slices, so one slice per gridpoint in the x direction
  Threads.@threads for i=1:size(x,1)
    F=svdfact(view(x,i,:,:))
    project_l1_Duchi!(F.S, sigma) # project singular values onto the l1-ball of size sigma
    @inbounds x[i,:,:].= F.U[:,1:r] * diagm(F.S[1:r])* F.Vt[1:r,:] #reconstruct
  end
elseif mode == "y"
  Threads.@threads for i=1:size(x,2)
    F=svdfact(view(x,:,i,:))
    project_l1_Duchi!(F.S, sigma) # project singular values onto the l1-ball of size sigma
    @inbounds x[:,i,:].= F.U[:,1:r] * diagm(F.S[1:r])* F.Vt[1:r,:]
  end
elseif mode == "z"
  Threads.@threads for i=1:size(x,3)
    F=svdfact(view(x,:,:,i))
    project_l1_Duchi!(F.S, sigma) # project singular values onto the l1-ball of size sigma
    @inbounds x[:,:,i].= F.U[:,1:r] * diagm(F.S[1:r])* F.Vt[1:r,:]
  end
end #end if
x=vec(x)
end
