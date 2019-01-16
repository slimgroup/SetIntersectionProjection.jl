export project_annulus!

function project_annulus!(
                     x     ::Union{Vector{TF},Vector{Complex{TF}}},
                     sigma_min ::TF,         #minimum 2-norm ||x||_2,
                     sigma_max ::TF          #maximum 2-norm ||x||_2
                     ) where {TF<:Real}

nl2 = norm(x,2)
if sigma_min <= nl2 <= sigma_max
  return x
elseif nl2 > sigma_max
  Base.LinAlg.scale!(x,sigma_max/nl2)
elseif nl2 < sigma_min && nl2>0
  Base.LinAlg.scale!(x,sigma_min/nl2)
elseif nl2 < sigma_min && nl2==0
  copy!(x,ones(TF,length(x)).*(sigma_min./sqrt(length(x))))
end

return x
end
