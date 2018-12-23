export project_cardinality!

function project_cardinality!{TF<:Real,TI<:Integer}(
                             x::Union{Vector{TF},Vector{Complex{TF}}},
                             k::TI,
                             )
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
x[sort_ind[k+1:end]]=0.0;
  return x
end

function project_cardinality!{TF<:Real,TI<:Integer}(
                             x    ::Array{TF,2},  #matrix, project each row or colum
                             k    ::TI,               #maximum cardinality
                             mode ::Tuple{String,String},    #
                             return_vec=true ::Bool
                             )
  """
  Project each of the columns/rows of the model in matrix form onto the set of vectors with cardinality (l0 'norm') less then or equal to k.
  project m onto {m | card(m)<= k} : x = argmin_x 1/2 ||x-m||2^2 s.t. card(x)<=k
  """

#sort_ind = sortperm( a, by=abs, rev=true)
#val_max=x[sort_ind[1:k]]
#x=zeros(length(x))
#x[sort_ind[1:k]]=val_max

#alternative
if mode == "x"
Threads.@threads for i=1:size(x,2)
    #sort_ind = sortperm( x[:,i], by=abs, rev=true)
    sort_ind = sortperm( view(x,:,i), by=abs, rev=true)
@inbounds x[sort_ind[k+1:end],i]=0.0;
  end
elseif mode == "y"
Threads.@threads for i=1:size(x,1)
    #sort_ind = sortperm( x[:,i], by=abs, rev=true)
    sort_ind = sortperm( view(x,i,:), by=abs, rev=true)
@inbounds x[i,sort_ind[k+1:end]]=0.0;
  end
end

if return_vec==true
  return vec(x)
else
  return x
end

end

function project_cardinality!{TF<:Real,TI<:Integer}(
                             x    ::Array{TF,3},
                             k    ::TI,
                             mode ::Tuple{String,String},
                             return_vec=true ::Bool
                             )
  """
  Project each of the x or y or z fibers of the model in tensor form onto the set of vectors with cardinality (l0 'norm') less then or equal to k.
  project m onto {m | card(m)<= k} : x = argmin_x 1/2 ||x-m||2^2 s.t. card(x)<=k
  """

(n1,n2,n3)=size(x)
#sort_ind = sortperm( a, by=abs, rev=true)
#val_max=x[sort_ind[1:k]]
#x=zeros(length(x))
#x[sort_ind[1:k]]=val_max

#alternative

#Fiber based projection
if mode[1] == "fiber"
  if mode[2] == "x"
    for i=1:n2
    Threads.@threads for j=1:n3
          #sort_ind = sortperm( x[:,i], by=abs, rev=true)
          sort_ind = sortperm( view(x,:,i,j), by=abs, rev=true)
  @inbounds x[sort_ind[k+1:end],i,j]=0.0;
        end
    end
  elseif mode[2] == "z"
    Threads.@threads for i=1:n1
    for j=1:n2
        #sort_ind = sortperm( x[:,i], by=abs, rev=true)
        sort_ind = sortperm( view(x,i,j,:), by=abs, rev=true)
  @inbounds x[i,j,sort_ind[k+1:end]]=0.0;
      end
    end
  elseif mode[2] == "y"
    for i=1:n1
    Threads.@threads for j=1:n3
        #sort_ind = sortperm( x[:,i], by=abs, rev=true)
        sort_ind = sortperm( view(x,i,:,j), by=abs, rev=true)
  @inbounds x[i,sort_ind[k+1:end],j]=0.0;
      end
    end
  end

elseif mode[1] == "slice" #Slice based projection
  if mode[2] == "x"
    x = reshape(x,n1,n2*n3)
    Threads.@threads for i=1:n1
      sort_ind = sortperm( x[i,:], by=abs, rev=true)
      @inbounds x[i,sort_ind[k+1:end]]=0.0;
    end
    x = reshape(x,n1,n2,n3)
  elseif mode[2] == "z"
    x = reshape(x,n1*n2,n3)
    Threads.@threads for i=1:n3
      sort_ind = sortperm( x[:,i], by=abs, rev=true)
      @inbounds x[sort_ind[k+1:end],i]=0.0;
    end
    x = reshape(x,n1,n2,n3)
  elseif mode[2] == "y"
    permutedims(x,[2;1;3]);
    x = reshape(x,n2,n1*n3)
    Threads.@threads for i=1:size(x,1)
      sort_ind = sortperm( x[i,:], by=abs, rev=true)
      @inbounds x[i,sort_ind[k+1:end]]=0.0;
    end
    x = reshape(x,n2,n1,n3);
    permutedims(x,[2;1;3]);
  end #end if

end #if slice/fiber mode

if return_vec==true
  return vec(x)
else
  return x
end

end
