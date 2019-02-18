export project_l1_Duchi!
function project_l1_Duchi!(v::Union{Vector{TF},Vector{Complex{TF}}}, b::TF) where {TF<:Real}
# % PROJECTONTOL1BALL Projects point onto L1 ball of specified radius.
# %
# % w = ProjectOntoL1Ball(v, b) returns the vector w which is the solution
# %   to the following constrained minimization problem:
# %
# %    min   ||w - v||_2
# %    s.t.  ||w||_1 <= b.
# %
# %   That is, performs Euclidean projection of v to the 1-norm ball of radius
# %   b.
# %
# % Author: John Duchi (jduchi@cs.berkeley.edu)
# Translated (with some modification) to Julia 1.1 by Bas Peters
if (b <= TF(0))
  error("Radius of L1 ball is negative");
end
if norm(v, 1) <= b
  return v
end
lv=length(v)
u = Vector{TF}(undef,lv)
sv= Vector{TF}(undef,lv)

#u = sort(abs(v),'descend');
#u = sort(abs.(v), rev=true) #faster than the line below
#u = sort(v, by=abs , rev=true)

#use RadixSort for Float32 (short keywords)
u=copy(v)
if TF==Float32
 u = sort!(abs.(u), rev=true,alg=RadixSort)
else
 u = sort!(abs.(u), rev=true,alg=QuickSort)
end

#sv = cumsum(u);
sv = cumsum(u)

#rho = find(u > (sv - b) ./ (1:length(u))', 1, 'last');
#tmp = (u .> ((sv.-b)./ LinSpace(1,lv,lv)))#::BitVector

rho = max(1,min(lv,findlast(u .> ((sv.-b)./ (1.0:1.0:lv)))))
#convert(TF,rho) why was this here...
#rho = findlast(tmp)::Int64

#theta = max(0, (sv(rho) - b) / rho);
theta = max.(TF(0) , (sv[rho] .- b) ./ rho)::TF

#w = sign(v) .* max(abs(v) - theta, 0);
v .= sign.(v) .* max.(abs.(v) .- theta, TF(0))

return v
end #end project_l1_Duchi
#
# function project_l1_Duchi!{TF<:Real}(v::Union{Array{TF,3},Array{Complex{TF},3}}, b::TF,mode::String)
#
# if (b < TF(0))
#   error("Radius of L1 ball is negative");
# end
# (n1,n2,n3)=size(v)
#
# ##Slice based projection
# if mode == "x_slice"
#   u = Vector{TF}(n2*n3)
#   sv= Vector{TF}(n2*n3)
#   v = reshape(v,n1,n2*n3)
#   Threads.@threads for i=1:n1
#     u = sort(abs.(v[i,:]), rev=true)
#     sv = cumsum(u)
#     rho = findlast(u .> ((sv.-b)./ (1.0:1.0:(n2*n3))))
#     theta = max.(TF(0) , (sv[rho] .- b) ./ rho)::TF
#     @inbounds v[i,:] .= sign.(v[i,:]) .* max.(abs.(v[:,i]).-theta, TF(0))
#   end
#   v = reshape(v,n1,n2,n3)
# elseif mode == "z_slice"
#   u = Vector{TF}(n1*n2)
#   sv= Vector{TF}(n1*n2)
#   v = reshape(v,n1*n2,n3)
#   Threads.@threads for i=1:n3
#     u = sort(abs.(v[:,i]), rev=true)
#     sv = cumsum(u)
#     rho = findlast(u .> ((sv.-b)./ (1.0:1.0:(n1*n2))))
#     theta = max.(TF(0) , (sv[rho] .- b) ./ rho)::TF
#     @inbounds v[:,i] .= sign.(v[:,i]) .* max.(abs.(v[:,i]).-theta, TF(0))
#   end
#   v = reshape(v,n1,n2,n3)
# elseif mode == "y_slice"
#   permutedims(v,[2;1;3]);
#   v = reshape(v,n2,n1*n3)
#   u = Vector{TF}(n1*n3)
#   sv= Vector{TF}(n1*n3)
#   Threads.@threads for i=1:size(v,1)
#     u = sort(abs.(v[i,:]), rev=true)
#     sv = cumsum(u)
#     rho = findlast(u .> ((sv.-b)./ (1.0:1.0:(n1*n3))))
#     theta = max.(TF(0) , (sv[rho] .- b) ./ rho)::TF
#     @inbounds v[i,:] .= sign.(v[i,:]) .* max.(abs.(v[:,i]).-theta, TF(0))
#   end
#   v = reshape(v,n2,n1,n3);
#   permutedims(v,[2;1;3]);
# else
#   error("provided incorred mode for l1_projection")
# end #end if
# v=vec(v)
#
# return v
# end #end project_l1_Duchi
