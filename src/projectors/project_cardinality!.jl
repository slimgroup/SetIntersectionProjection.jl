export project_cardinality!

function project_cardinality!(
                             x::Union{Vector{TF},Vector{Complex{TF}}},
                             k::TI,
                             ) where {TF<:Real,TI<:Integer}
  """
  Project the vector x onto the set of vectors with cardinality (l0 'norm') less then or equal to k.
  project m onto {m | card(m) <= k} : x = argmin_x 1/2 ||x-m||2^2 s.t. card(x)<=k
  """

#sort_ind = sortperm( a, by=abs, rev=true)
#val_max=x[sort_ind[1:k]]
#x=zeros(length(x))
#x[sort_ind[1:k]]=val_max

#alternative
sort_ind = sortperm( x, by=abs, rev=true)
x[sort_ind[k+1:end]] .= TF(0.0)
  return x
end

function project_cardinality!(
                             x    ::Array{TF,2},  #matrix, project each row or colum
                             k    ::TI,               #maximum cardinality
                             mode ::Tuple{String,String},    #
                             return_vec=true ::Bool
                             ) where {TF<:Real,TI<:Integer}
  """
  Project each of the columns/rows of the model in matrix form onto the set of vectors with cardinality (l0 'norm') less then or equal to k.
  project m onto {m | card(m)<= k} : x = argmin_x 1/2 ||x-m||2^2 s.t. card(x)<=k
  """

#sort_ind = sortperm( a, by=abs, rev=true)
#val_max=x[sort_ind[1:k]]
#x=zeros(length(x))
#x[sort_ind[1:k]]=val_max

#alternative
if mode[1] == "fiber"
  if mode[2] == "x"
  Threads.@threads for i=1:size(x,2)
      sort_ind = sortperm( view(x,:,i), by=abs, rev=true)
  @inbounds x[sort_ind[k+1:end],i] .= TF(0.0)
    end
  elseif mode[2] == "z"
  Threads.@threads for i=1:size(x,1)
      sort_ind = sortperm( view(x,i,:), by=abs, rev=true)
  @inbounds x[i,sort_ind[k+1:end]] .= TF(0.0)
    end
  end
else
  error("for 2D models, the mode of application for project_cardinality! needs to be (fiber,x) or (fiber,y). Or, provide the model as a vector")
end

if return_vec==true
  return vec(x)
else
  return x
end

end

function project_cardinality!(
                             x    ::Array{TF,3},
                             k    ::TI,
                             mode ::Tuple{String,String},
                             return_vec=true ::Bool
                             ) where {TF<:Real,TI<:Integer}
  """
  Project each of the x or y or z fibers of the model in tensor form onto the set of vectors with cardinality (l0 'norm') less then or equal to k.
  project m onto {m | card(m)<= k} : x = argmin_x 1/2 ||x-m||2^2 s.t. card(x)<=k
  """

(n1,n2,n3) = deepcopy(size(x))
#sort_ind = sortperm( a, by=abs, rev=true)
#val_max=x[sort_ind[1:k]]
#x=zeros(length(x))
#x[sort_ind[1:k]]=val_max

#alternative

#Fiber based projection for 3D tensor
if mode[1] == "fiber"
  if mode[2] == "x"
    for i=1:n2
    Threads.@threads for j=1:n3
          #sort_ind = sortperm( x[:,i], by=abs, rev=true)
          sort_ind = sortperm( view(x,:,i,j), by=abs, rev=true)
  @inbounds x[sort_ind[k+1:end],i,j] .= TF(0.0)
        end
    end
  elseif mode[2] == "z"
    Threads.@threads for i=1:n1
    for j=1:n2
        #sort_ind = sortperm( x[:,i], by=abs, rev=true)
        sort_ind = sortperm( view(x,i,j,:), by=abs, rev=true)
  @inbounds x[i,j,sort_ind[k+1:end]] .= TF(0.0)
      end
    end
  elseif mode[2] == "y"
    for i=1:n1
    Threads.@threads for j=1:n3
        #sort_ind = sortperm( x[:,i], by=abs, rev=true)
        sort_ind = sortperm( view(x,i,:,j), by=abs, rev=true)
  @inbounds x[i,sort_ind[k+1:end],j] .= TF(0.0)
      end
    end
  end

elseif mode[1] == "slice" #Slice based projection for 3D tensor
  #code currently does not mutate the input for slice projections for 3D tensors...
  #permute and reshape
  if mode[2] == "x"
    x = permutedims(x,[2;3;1])
    x = reshape(x,n2*n3,n1)
  elseif mode[2] == "y"
    x = permutedims(x,[1;3;2])
    x = reshape(x,n1*n3,n2)
  elseif mode[2] == "z"
    x = reshape(x,n1*n2,n3) #there are n3 slices of dimension n1*n2
  end

  #project, same for all modes because we permuted and reshaped already
  Threads.@threads for i=1:size(x,2)
    sort_ind = sortperm( view(x,:,i), by=abs, rev=true)
    @inbounds x[sort_ind[k+1:end],i] .= TF(0.0)
  end

  #reverse reshape and permute back
  if mode[2] == "x"
    x = reshape(x,n2,n3,n1)
    x = permutedims(x,[3;1;2]);
  elseif mode[2] == "y"
    x = reshape(x,n1,n3,n2)
    x = permutedims(x,[1;3;2]);
  elseif mode[2] == "z"
    x = reshape(x,n1,n2,n3) #there are n3 slices of dimension n1*n2
  end
end #if slice/fiber mode

if return_vec==true
  x = vec(x)
end
return x
end
