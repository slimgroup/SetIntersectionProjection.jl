export project_cardinality_parallel

function project_cardinality_parallel(x::Vector,
                             k ::Int64,
                             n_proc::Int64
                             )
"""
Project the vector x onto the set of vectors with cardinality (l0 'norm') less then or equal to k.
project m onto {m | card(m)<= k} : x = argmin_x 1/2 ||x-m||2^2 s.t. card(x)<=k
A parallel version
"""

#sort_ind = sortperm( a, by=abs, rev=true)
#val_max=x[sort_ind[1:k]]
#x=zeros(length(x))
#x[sort_ind[1:k]]=val_max

ns = convert(Int64,floor((length(x)/n_proc)))
sort_ind=Vector{Vector{Int64}}(n_proc)
sort_ind[1] = sortperm( x[1:ns], by=abs, rev=true)
sort_ind[2] = sortperm( x[ns+1:end], by=abs, rev=true)

val_max=Vector{Float64}(2*k)
val_max[1:k]=x[sort_ind[1][1:k]]
val_max[1+k:k*2]=x[sort_ind[2][1:k]]

sort_final_ind = sortperm(val_max,by=abs,rev=true  )
#sort_ind[1]=sort_ind[1][1:k]
#sort_ind[2]=sort_ind[2][1:k]
#sort_ind_total=[sort_ind[1];sort_ind[2]]
x[sort_ind[1][k+1:end]]=0.0;
x[sort_ind[2][k+1:end]+ns]=0;.0

for i=(k+1):length(sort_final_ind)
  print(i)
  if sort_final_ind[i]<=k
    x[sort_ind[1][sort_final_ind[i]]]=0.0
  else
    x[sort_ind[2][sort_final_ind[i]-k]]=0.0
  end
end

  return x
end
