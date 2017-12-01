export project_l2!

function project_l2!{TF<:Real}(
                     x     ::Union{Vector{TF},Vector{Complex{TF}}},
                     sigma ::TF          #maximum 2-norm ||x||_2
                     )

nl2 = norm(x,2)
if nl2 <= sigma
  return x
else
  Base.LinAlg.scale!(x,sigma/nl2)
end

return x
end
