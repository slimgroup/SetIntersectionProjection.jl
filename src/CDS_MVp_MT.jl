export CDS_MVp_MT

"""
compute multi-threaded matrix vector product with vector x, output is vector y: y=A*x
MVP is in the compressed diagonal format.
R is a tall matrix N by ndiagonals, corresponding to a square matrix A
offsets indicate offset of diagonal compared to the main diagonal in A (which is 0)
"""
function CDS_MVp_MT(
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
      CDS_MVp_MT_subfunc(R,x,y,r0,c0,r1,i)
  end
  return y
end
