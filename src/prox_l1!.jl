export prox_l1!

"""
simple scirpt to compute the l1 proximal map.
our rho is 1./rho compared to other definitions of the proximal map (just a convention)
"""

function prox_l1!(x::Vector{TF},rho::TF) where {TF<:Real}
  x .= sign.(x) .* max.(TF(0.0), abs.(x) .- (1 ./ rho))
end
