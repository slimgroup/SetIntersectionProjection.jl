export a_is_b_min_c_MT!

function a_is_b_min_c_MT!(a::Vector{TF},b::Vector{TF},c::Vector{TF}) where {TF<:Real}
"""
in-place multithreaded a = b - c
"""
    Threads.@threads for j=1:length(a)
      @inbounds a[j] = b[j] - c[j]
    end
end
