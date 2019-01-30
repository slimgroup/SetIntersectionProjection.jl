export project_histogram_relaxed!

"""
Computes the projection of x onto a (relaxed) histogram.
P*LB <= P*x <= P*UB
x is a vector and the bounds are vectors as well. P is the orthogonal permutation matrix that
sorts x by magnitude. The LB and UB are vectors that are the sorted magnitude of the bounds
"""
function project_histogram_relaxed!(x::Vector{TF},LB::Vector{TF},UB::Vector{TF}) where {TF<:Real}

sort_ind = sortperm( x, by=abs)
x.=x[sort_ind]
#we use sorting from small to large. LB and UB also need to be sorted from small to large!
@inbounds Threads.@threads for j in (1:length(x))
  @inbounds  x[j] = min(x[j],UB[j])
  @inbounds  x[j] = max(LB[j],x[j])
end
sort_ind_sort_ind = sortperm( sort_ind)
x.=x[sort_ind_sort_ind] #rearrange back to original order

# This version needs two sorts.
# We can also apply the bounds in a loop over the x[sort_ind] coeffs
# without sorting x and sorting it back, but it requires memory access
# to x, LB and UB in non linear order.
return x
end
