export prox_l2s!
function prox_l2s!{TF<:Real}(
  x   ::Vector{TF},
  rho ::TF,
  m   ::Vector{TF}
  )

  x.= (x.*rho.+m)./(rho.+1);
  return x
end
