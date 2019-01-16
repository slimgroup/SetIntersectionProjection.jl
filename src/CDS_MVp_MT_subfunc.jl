export CDS_MVp_MT_subfunc
function CDS_MVp_MT_subfunc(
        R::Array{TF,2},
        x::Vector{TF},
        y::Vector{TF},
        r0::Int,
        c0::Int,
        r1::Int,
        i::Int) where {TF<:Real}

"""
wrapper around Julia threads to compute a multi-threaded matrix vector product in CDS formulated
This is a subfunction of CDS_MVp_MT.jl
"""

      #s=rind-1
  @Threads.threads for r = r0 : r1
      c = r - r0 + c0 #original
      @inbounds y[r] = y[r] + R[r,i] * x[c]#original
    end
  return y
end
