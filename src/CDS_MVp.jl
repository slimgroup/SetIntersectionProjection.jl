export CDS_MVp, CDS_MVp2, CDS_MVp3, CDS_MVp4

"""
compute single-thread matrix vector product with vector x, output is vector y: y=A*x
MVP is in the compressed diagonal format.
R is a tall matrix N by ndiagonals, corresponding to a square matrix A
offsets indicate offset of diagonal compared to the main diagonal in A (which is 0)
"""
function CDS_MVp(
                        N      ::Integer,
                        ndiags ::Integer,
                        R      ::Array{TF,2},
                        offset ::Vector{TI},
                        x      ::Vector{TF},
                        y      ::Vector{TF}) where {TF<:Real,TI<:Integer}

  for i = 1 : ndiags
      d  = offset[i]
      r0 = max(1, 1-d)
      r1 = min(N, N-d)
      c0 = max(1, 1+d)
       for r = r0 : r1
         c = r - r0 + c0 #original
       @inbounds y[r] = y[r] + R[r,i] * x[c]#original
      end
  end
  return y
end

# #the versions below is insignificantly faster than the original version above
# function CDS_MVp(
#   N::Integer,
#   ndiags::Integer,
#   R::Array{TF,2},
#   offset::Vector{TI},
#   x::Vector{TF},
#   y::Vector{TF}) where {TF<:Real,TI<:Integer}

#   for i = 1 : ndiags
#     d = offset[i]
#     r0 = max(1, 1-d)
#     r1 = min(N, N-d)
#     c0 = max(1, 1+d)
#     #r0_plus_c0 = r0 + c0
#     c = deepcopy(c0)
#     for r = r0 : r1
#       @inbounds y[r] = y[r] + R[r,i] * x[c]#original
#       c += 1
#     end
#   end
# return y
# end

# function CDS_MVp(
#   N      ::Integer,
#   ndiags ::Integer,
#   R      ::Array{TF,2},
#   offset ::Vector{TI},
#   x      ::Vector{TF},
#   y      ::Vector{TF}) where {TF<:Real,TI<:Integer}

#   for i = 1 : ndiags
#     d  = offset[i]
#     r0 = max(1, 1-d)
#     r1 = min(N, N-d)
#     c0 = max(1, 1+d)
#     # for r = r0 : r1
#     #   c = r - r0 + c0 #original
#     #   @inbounds y[r] = y[r] + R[r,i] * x[c]#original
#     # end
#     @inbounds y[r0 : r1] .= y[r0 : r1] .+ R[r0:r1,i].*x[r0 - r0 + c0:r1 - r0 + c0]
#   end

#   return y
# end

