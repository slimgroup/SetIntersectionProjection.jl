export project_subspace!

function project_subspace!{TF<:Real}(
                          input::Vector{TF},
                          A,
                          orth::Bool
                          )
  if orth == true
     input .= A*(A'*input)
  else
    input .= A*((A'*A)\(A'*input))
  end

end
