export project_l1_Duchi!
function project_l1_Duchi!{TF<:Real}(v::Union{Vector{TF},Vector{Complex{TF}}}, b::TF)
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
# Translated to Julia 0.6 by Bas Peters

if (b < TF(0))
  error("Radius of L1 ball is negative");
end
if norm(v, 1) <= b
  return v
end
const lv=length(v)
u = Vector{TF}(lv)
sv= Vector{TF}(lv)

#u = sort(abs(v),'descend');
u = sort(abs.(v), rev=true)
#u = sort(v, by=abs , rev=true)

#sv = cumsum(u);
sv = cumsum(u)

#rho = find(u > (sv - b) ./ (1:length(u))', 1, 'last');
#tmp = (u .> ((sv.-b)./ LinSpace(1,lv,lv)))#::BitVector

const rho = findlast(u .> ((sv.-b)./ (1.0:1.0:lv)))
#convert(TF,rho) why was this here...
#rho = findlast(tmp)::Int64

#theta = max(0, (sv(rho) - b) / rho);
const theta = max.(TF(0) , (sv[rho] .- b) ./ rho)::TF

#w = sign(v) .* max(abs(v) - theta, 0);
v .= sign.(v) .* max.(abs.(v).-theta, TF(0))

return v
end #end project_l1_Duchi
