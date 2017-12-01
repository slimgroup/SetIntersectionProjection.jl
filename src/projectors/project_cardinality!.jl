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
                             mode ::String,           #"x" or "z"
                             return_vec=true ::Bool
                             )
"""
Project each of the columns of the model in matrix form onto the set of vectors with cardinality (l0 'norm') less then or equal to k.
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
elseif mode == "z"
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
                             mode ::String,
                             return_vec=true ::Bool
                             )
"""
Project each of the x or y or z fibers of the model in tensor form onto the set of vectors with cardinality (l0 'norm') less then or equal to k.
project m onto {m | card(m)<= k} : x = argmin_x 1/2 ||x-m||2^2 s.t. card(x)<=k
"""

#sort_ind = sortperm( a, by=abs, rev=true)
#val_max=x[sort_ind[1:k]]
#x=zeros(length(x))
#x[sort_ind[1:k]]=val_max

#alternative
if mode == "x"
  for i=1:size(x,2)
  Threads.@threads for j=1:size(x,3)
        #sort_ind = sortperm( x[:,i], by=abs, rev=true)
        sort_ind = sortperm( view(x,:,i,j), by=abs, rev=true)
@inbounds x[sort_ind[k+1:end],i,j]=0.0;
      end
  end
elseif mode == "z"
  for i=1:size(x,1)
  Threads.@threads for j=1:size(x,2)
      #sort_ind = sortperm( x[:,i], by=abs, rev=true)
      sort_ind = sortperm( view(x,i,j,:), by=abs, rev=true)
@inbounds x[i,j,sort_ind[k+1:end]]=0.0;
    end
  end
elseif mode == "y"
  for i=1:size(x,1)
  Threads.@threads for j=1:size(x,3)
      #sort_ind = sortperm( x[:,i], by=abs, rev=true)
      sort_ind = sortperm( view(x,i,:,j), by=abs, rev=true)
@inbounds x[i,sort_ind[k+1:end],j]=0.0;
    end
  end
end

if return_vec==true
  return vec(x)
else
  return x
end

end
