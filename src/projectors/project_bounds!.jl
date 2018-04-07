export project_bounds!

function project_bounds!{TF<:Real}(x::Vector{TF},LB::TF,UB::TF)
  """
  Computes the projection of x onto the set of constraints LB <= x <= UB
  x is a vector but the bounds are scalars. Uses some manual tricks like @inbounds and multi threading, attempt to be faster than native julia: x. = max.(LB,min.(x,UB))
  """
  Threads.@threads for j in (1:length(x))
    @inbounds x[j] = max(LB,min(x[j],UB))
  end
    return x
end

function project_bounds!{TF<:Real}(x::Vector{TF},LB::Vector{TF},UB::Vector{TF})
  """
  Computes the projection of x onto the set of constraints LB <= x <= UB
  x is a vector and the bounds are vectors as well. Uses some manual tricks like @inbounds and multi threading, attempt to be faster than native julia: x. = max.(LB,min.(x,UB))
  """
@inbounds Threads.@threads for j in (1:length(x))
    #x[i] = max.(LB,min.(x[i],UB))
  @inbounds  x[j] = min(x[j],UB[j])
  @inbounds  x[j] = max(LB[j],x[j])
  end
    return x
end

function project_bounds!{TF<:Real}(x::Array{TF,2},LB::Vector{TF},UB::Vector{TF},mode::String)
  """
  Computes the projection of x onto the set of constraints LB <= x <= UB
  x is a matrix and the bounds are per row or column.
  """

  if mode == "x"
    Threads.@threads for i=1:size(x,2)
      @inbounds x[:,i].=min.(max.(x[:,i],LB),UB)
    end
  elseif mode == "y"
    Threads.@threads for i=1:size(x,1)
      @inbounds x[i,:].=min.(max.(x[i,:],LB),UB)
    end
  end
return x
end

function project_bounds!{TF<:Real}(x::Array{TF,3},LB::Vector{TF},UB::Vector{TF},mode::String)
  """
  Computes the projection of x onto the set of constraints LB <= x <= UB
  x is a 3D array and the bounds are per fiber (x/y/z).
  """

  if mode == "x"
    Threads.@threads for i=1:size(x,2)
      for j=1:size(x,3)
        @inbounds x[:,i,j].=min.(max.(x[:,i,j],LB),UB)
      end
    end
  elseif mode == "z"
    Threads.@threads for i=1:size(x,1)
      for j=1:size(x,3)
        @inbounds x[i,:,j].=min.(max.(x[i,:,j],LB),UB)
      end
    end
  elseif mode == "z"
    Threads.@threads for i=1:size(x,1)
      for j=1:size(x,2)
        @inbounds x[i,j,:].=min.(max.(x[i,j,:],LB),UB)
      end
    end

  end
return x
end
