export a_is_b_min_c_MT!

function a_is_b_min_c_MT!{TF<:Real}(a::Vector{TF},b::Vector{TF},c::Vector{TF})
    Threads.@threads for j=1:length(a)
      @inbounds a[j] = b[j] - c[j]
    end
end
