function CDS_MVp_MT{TF<:Real,TI<:Integer}(
                        N::Integer,
                        ndiags::Integer,
                        R::Array{TF,2},
                        offset::Vector{TI},
                        x::Vector{TF},
                        y::Vector{TF})
#R is a tall matrix N by ndiagonals, corresponding to a square matrix A
  for i = 1 : ndiags
      d = offset[i]
      r0 = max(1, 1-d)
      r1 = min(N, N-d)
      c0 = max(1, 1+d)
      CDS_MVp_MT_subfunc(R,x,y,r0,c0,r1,i)
  end
  return y
end
