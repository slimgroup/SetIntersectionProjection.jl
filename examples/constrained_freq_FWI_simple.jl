#simple frequency domain FWI to illustrate PARSDMM
#include("/home/slim/bpeters/.julia/v0.6/WAVEFORM/src/Waveform.jl")
using Waveform

using JOLI
using PyPlot
#using OptimPackNextGen
using opesciSLIM.SLIM_optim
using Waveform

@everywhere using SetIntersectionProjection

include("SPGslim.jl")

#data directory for loading and writing results
data_dir = "/data/slim/bpeters/SetIntersection_data_results"


# Set up the model geometry
# In 2D, we organize our coordinates as (z,x) for legacy + visualization purposes

# Number of gridpoints in z-, x-
n = [151;101];

# Spacing of gridpoints in z-, x-
d = 10.0*[1;1];

# Origin point of domain in z-, x- (typically (0.,0.))
o = 0.0*[1;1];

# Constant velocity background
vel_background = 2500;

# Source wavelet time shift (in seconds)
t0 = 0.0;

# Peak frequency of Ricker Wavelet (in Hz)
f0 = 10.0;

# Units of the model
unit = "m/s";

# Model domain range
L = n.*d;

# x-coordinates of the sources
xsrc = 0.0:50.0:L[2];

# y-coordinates of the sources (irrelevant in 2D)
ysrc = [0.0];

# z-coordinates of the sources
zsrc = [10.0];

# x-coordinates of the receivers
xrec = [950.0];

# y-coordinates of the receivers (irrelevant in 2D)
yrec = [0.0];

# z-coordinates of the receivers (this is a transmission experiment)
#zrec = 0.0:50.0:L[2];
zrec = 0.0:50.0:L[1];

# Frequencies to compute
freqs = [3.0 ; 5.0 ; 8.0]#[5.0;10.0;15.0];

# Maximum wavelength
λ = vel_background/maximum(freqs);

# Model type
model = Model{Int64,Float64}(n,d,o,t0,f0,unit,freqs,xsrc,ysrc,zsrc,xrec,yrec,zrec);
(z,x) = odn_to_grid(o,d,n);
nsrc,nrec,nfreq = length(xsrc),length(xrec),length(freqs)

# Set up computational parameters
## Size/spacing of computational domain
comp_n = n;
comp_d = d;
comp_o = o;

# Number of PML points
npml = round(Int,λ/minimum(comp_d))*[1 1; 1 1];

# PDE scheme
scheme = Waveform.helm2d_chen9p;

# Whether to remove elements in the PML when restricting to the model domain (if false, will use stacking)
cut_pml = true;

# If true, use an implicit matrix representation (false for 2D, the matrices are small enough to form explicitly)
implicit_matrix = false;

# Binary mask of the sources and frequencies to compute
srcfreqmask = trues(nsrc,nfreq);

# Misfit function for the objective
misfit = least_squares;

# Linear solver options
lsopts = LinSolveOpts(solver=:lufact);

# Computational options type
opts = PDEopts{Int64,Float64}(scheme,comp_n,comp_d,comp_o,cut_pml,implicit_matrix,npml,misfit,srcfreqmask,lsopts);

# Create velocity model
v0 = vel_background*ones(n...);
v = copy(v0);
#v[div(n[1],3):2*div(n[1],3),div(n[2],3):2*div(n[2],3)] = 1.25*vel_background;
v[61:80,50:70]=2400
v = vec(v);
v0 = vec(v0);

function plot_velocity(vs,title_str,model,keyword,rec_x,rec_z,src_x,src_z)
    fig = figure()
    ax1 = axes()
    vplot = imshow(reshape(vs,model.n...),vmin=minimum(v)-50,vmax=maximum(v)+50,axes=ax1,cmap="jet",extent=[0.0 , model.d[2]*model.n[2] , model.d[1]*model.n[1] , 0.0])
    title(title_str)
    xlabel("x [m]")
    ylabel("z [m]")
    zt = 0:20:n[1]
    xt = 0:20:n[2]
    #xticks(xt,round.(Int,xt*model.d[2]))
    #yticks(zt,round.(Int,zt*model.d[1]))
    colorbar()
    figure;plot(rec_x,rec_z,linewidth=1.0, marker="o",linestyle="")
    figure;plot(src_x,src_z,linewidth=1.0, marker="x",linestyle="","k")
    savefig(joinpath(data_dir,string("CFWI_simple_freq_m_est_",keyword,".eps")),bbox_inches="tight",dpi=300)
    savefig(joinpath(data_dir,string("CFWI_simple_freq_m_est_",keyword,".png")),bbox_inches="tight")
    return nothing
end
plot_velocity(v,"a) True velocity",model,"true",repmat(xrec,length(zrec),1),zrec,xsrc,repmat(zsrc,length(xsrc),1))
plot_velocity(v0,"b) Initial velocity",model,"initial",repmat(xrec,length(zrec),1),zrec,xsrc,repmat(zsrc,length(xsrc),1))

# Source weight matrix
Q = eye(nsrc);

# Generate data
D = forw_model(v,Q,model,opts);

imshow(real(D[:,1:nsrc]),aspect="auto");
title("Frequency slice at $(model.freq[1]) Hz")

# Frequency continuation
max_func_evals = 10

# Frequency continuation #original
# size_freq_batch = 2
# overlap = 1
# freq_partition = partition(nfreq,size_freq_batch,overlap)
# vest = v0
# opts1 = deepcopy(opts);
# proj! = (xproj,x)->project_bounds!(x,minimum(v),maximum(v),xproj);
#
# for j in 1:size(freq_partition,2)
#     fbatch = freq_partition[:,j]
#     srcfreqmask = falses(nsrc,nfreq)
#     srcfreqmask[:,fbatch] = true
#     opts1.srcfreqmask = srcfreqmask
#     obj! = construct_pde_misfit(v,Q,D,model,opts1,batch_mode=false)
#     vest = spg(obj!,proj!,vest,3,maxfc=max_func_evals)
# end

#Frequency continuation my own
size_freq_batch = 2
overlap = 1
freq_partition = partition(nfreq,size_freq_batch,overlap)
vest = v0
opts1 = deepcopy(opts);
# proj! = (xproj,x)->project_bounds!(x,minimum(v)-50,maximum(v)+50,xproj);
#
#
# for j in 1:size(freq_partition,2)
#     fbatch = freq_partition[:,j]
#     srcfreqmask = falses(nsrc,nfreq)
#     srcfreqmask[:,fbatch] = true
#     opts1.srcfreqmask = srcfreqmask
#     obj! = construct_pde_misfit(v,Q,D,model,opts1,batch_mode=false)
#     vest = spg(obj!,proj!,vest,3,maxfc=max_func_evals)
# end
#
# plot_velocity(vest,"Inverted velocity - freq continuation",model)

# with constraints:
options=PARSDMM_options()
options.FL=Float64
options=default_PARSDMM_options(options,options.FL)
options.adjust_gamma           = true
options.adjust_rho             = true
options.adjust_feasibility_rho = true
options.Blas_active            = true
options.maxit                  = 1000
options.feas_tol= 0.001
options.obj_tol=0.001
options.evol_rel_tol = 0.00001

options.rho_ini=[1.0f0]

set_zero_subnormals(true)
BLAS.set_num_threads(2)
FFTW.set_num_threads(2)
options.parallel=false
options.linear_inv_prob_flag = false
options.zero_ini_guess=true

type compgrid
  d :: Tuple
  n :: Tuple
end
comp_grid=compgrid((model.d[1],model.d[2]),(model.n[1],model.n[2]))
#function run_CFWI(freq_partition,nsrc,nfreq,v,Q,D,model,opts1,proj!,vest,max_func_evals)

options_SPG = spg_options(verbose=3, maxIter=max_func_evals, memory=5,suffDec=1f-6)
options_SPG.progTol=1e-6
options_SPG.testOpt=false
options_SPG.interp=2

function run_CFWI(freq_partition,nsrc,nfreq,v,Q,D,model,opts1,options_SPG,proj!,vest,max_func_evals)
  for j in 1:size(freq_partition,2)
      fbatch = freq_partition[:,j]
      srcfreqmask = falses(nsrc,nfreq)
      srcfreqmask[:,fbatch] = true
      opts1.srcfreqmask = srcfreqmask
      obj! = construct_pde_misfit(v,Q,D,model,opts1,batch_mode=false)
      function objective_function(in,obj!)
        f=0.0
        g=zeros(length(in))
        f=obj!(in,g);
        return f,g
      end
      f_obj = in -> objective_function(in,obj!)
      vest, fsave, funEvals= minConf_SPG(f_obj, vec(vest), proj!, options_SPG)
      #vest = spg(obj!,proj!,vest,3,maxfc=max_func_evals,eps1 = 1e-8,eps2 = 1e-8)
  end
  return vest
end

CFWI(ini_model,projector) = run_CFWI(freq_partition,nsrc,nfreq,v,Q,D,model,opts1,options_SPG,projector,ini_model,max_func_evals)

constraint_strategy_list=[1 2 3 4 5 6 7 8 9]# 2 3 4 5 6]
for j in constraint_strategy_list
  if j==1
    keyword="bounds_only"
    title_str="c) bounds only"

    constraint=Dict()
    constraint["use_bounds"]=true
    constraint["m_min"] = minimum(v)-50.0
    constraint["m_max"] = maximum(v)+50.0

  # various types of cardinality and rank
elseif j==2
		keyword="cardmat_cardcol_rank_bounds"
    title_str="h) fiber and matrix grad. card. & rank & bounds"

		constraint=Dict()
		constraint["use_bounds"]=true
    constraint["m_min"] = minimum(v)-50.0
    constraint["m_max"] = maximum(v)+50.0

		#cardinality on derivatives (column and row wise)
		constraint["use_TD_card_fiber_x"]			= true
		constraint["card_fiber_x"] 						= 2
	  constraint["TD_card_fiber_x_operator"]= "D_x"

		constraint["use_TD_card_fiber_z"]=true
		constraint["card_fiber_z"]=2
		constraint["TD_card_fiber_z_operator"]="D_z"

		#cardinality on derivatives (matrix based)
		constraint["use_TD_card_1"]=true
		constraint["card_1"]=round(Integer,3*0.33*n[2])
		constraint["TD_card_operator_1"]="D_x"

		constraint["use_TD_card_2"]=true
		constraint["card_2"]=round(Integer,3*0.33*n[1])
		constraint["TD_card_operator_2"]="D_z"

		#rank constraint
		constraint["use_TD_rank_1"]=true
	  constraint["TD_max_rank_1"]=3
    constraint["TD_rank_operator_1"]="identity"
    #true tv and bounds
  elseif j==3
      keyword="trueTV_bounds"
      title_str="d) true TV & bounds"

      constraint=Dict()
      constraint["use_bounds"]=true
      constraint["m_min"] = minimum(v)-50.0
      constraint["m_max"] = maximum(v)+50.0

      constraint["use_TD_l1_1"]=true
      constraint["TD_l1_operator_1"]="TV"
      (TV,dummy1,dummy2,dummy3)=get_TD_operator(comp_grid,"TV",options.FL)
      constraint["TD_l1_sigma_1"]=norm(TV*vec(v),1)

    elseif j==4
      keyword="rank_bounds"
      title_str="rank & bounds"

      constraint=Dict()
      constraint["use_bounds"]=true
      constraint["m_min"] = minimum(v)-50.0
      constraint["m_max"] = maximum(v)+50.0

      #rank constraint
  		constraint["use_TD_rank_1"]=true
  	  constraint["TD_max_rank_1"]=3
      constraint["TD_rank_operator_1"]="identity"

      # various types of cardinality
    elseif j==5
    			keyword="cardmat_cardcol_bounds"
          title_str="g) fiber and matrix grad. card. & bounds"

    			constraint=Dict()
    			constraint["use_bounds"]=true
          constraint["m_min"] = minimum(v)-50.0
          constraint["m_max"] = maximum(v)+50.0

    			#cardinality on derivatives (column and row wise)
    			constraint["use_TD_card_fiber_x"]			= true
    			constraint["card_fiber_x"] 						= 2
    		  constraint["TD_card_fiber_x_operator"]= "D_x"

    			constraint["use_TD_card_fiber_z"]=true
    			constraint["card_fiber_z"]=2
    			constraint["TD_card_fiber_z_operator"]="D_z"

    			#cardinality on derivatives (matrix based)
    			constraint["use_TD_card_1"]=true
    			constraint["card_1"]=round(Integer,3*0.33*n[2])
    			constraint["TD_card_operator_1"]="D_x"

    			constraint["use_TD_card_2"]=true
    			constraint["card_2"]=round(Integer,3*0.33*n[1])
    			constraint["TD_card_operator_2"]="D_z"

        elseif j==6
    				keyword="cardmat_bounds"
            title_str="f) matrix grad. card. & bounds"

    				constraint=Dict()
    				constraint["use_bounds"]=true
            constraint["m_min"] = minimum(v)-50.0
            constraint["m_max"] = maximum(v)+50.0

    				#cardinality on derivatives (matrix based)
    				constraint["use_TD_card_1"]=true
    				constraint["card_1"]=round(Integer,3*0.33*n[2])
    				constraint["TD_card_operator_1"]="D_x"

    				constraint["use_TD_card_2"]=true
    				constraint["card_2"]=round(Integer,3*0.33*n[1])
    				constraint["TD_card_operator_2"]="D_z"

            #rank constraint
        		constraint["use_TD_rank_1"]=false
        	  constraint["TD_max_rank_1"]=3
            constraint["TD_rank_operator_1"]="identity"

          elseif j==7
    					keyword="cardcol_bounds"
              title_str="e) fiber grad. card. & bounds"

    					constraint=Dict()
    					constraint["use_bounds"]=true
              constraint["m_min"] = minimum(v)-50.0
              constraint["m_max"] = maximum(v)+50.0

    					#cardinality on derivatives (column and row wise)
    					constraint["use_TD_card_fiber_x"]			= true
    					constraint["card_fiber_x"] 						= 2
    					constraint["TD_card_fiber_x_operator"]= "D_x"

    					constraint["use_TD_card_fiber_z"]=true
    					constraint["card_fiber_z"]=2
    					constraint["TD_card_fiber_z_operator"]="D_z"

              #rank constraint
          		constraint["use_TD_rank_1"]=false
          	  constraint["TD_max_rank_1"]=3
              constraint["TD_rank_operator_1"]="identity"

            elseif j==8
              keyword="rank_TVrank_bounds"
              title_str="rank, grad. rank, bounds"

              constraint=Dict()
              constraint["use_bounds"]=true
              constraint["m_min"] = minimum(v)-50.0
              constraint["m_max"] = maximum(v)+50.0

              #rank constraint
          		constraint["use_TD_rank_1"]=true
          	  constraint["TD_max_rank_1"]=3
              constraint["TD_rank_operator_1"]="identity"

              constraint["use_TD_rank_2"]=true;
              constraint["TD_max_rank_2"]=3
              constraint["TD_rank_operator_2"]="D_x"

              constraint["use_TD_rank_3"]=true;
              constraint["TD_max_rank_3"]=3
              constraint["TD_rank_operator_3"]="D_z"

            elseif j==9
              keyword="cardmat_cardcol_cardDxz_rank_bounds"
              title_str="i) fiber and matrix grad. card. & Dxz card. & rank & bounds"

          		constraint=Dict()
          		constraint["use_bounds"]=true
              constraint["m_min"] = minimum(v)-50.0
              constraint["m_max"] = maximum(v)+50.0

          		#cardinality on derivatives (column and row wise)
          		constraint["use_TD_card_fiber_x"]			= true
          		constraint["card_fiber_x"] 						= 2
          	  constraint["TD_card_fiber_x_operator"]= "D_x"

          		constraint["use_TD_card_fiber_z"]=true
          		constraint["card_fiber_z"]=2
          		constraint["TD_card_fiber_z_operator"]="D_z"

          		#cardinality on derivatives (matrix based)
          		constraint["use_TD_card_1"]=true
          		constraint["card_1"]=round(Integer,3*0.33*n[2])
          		constraint["TD_card_operator_1"]="D_x"

          		constraint["use_TD_card_2"]=true
          		constraint["card_2"]=round(Integer,3*0.33*n[1])
          		constraint["TD_card_operator_2"]="D_z"

              constraint["use_TD_card_3"]=true
              constraint["card_3"]=4
              constraint["TD_card_operator_3"]="D_xz"

          		#rank constraint
          		constraint["use_TD_rank_1"]=true
          	  constraint["TD_max_rank_1"]=3
              constraint["TD_rank_operator_1"]="identity"

              #2 cycles: 1: cardinality and bounds. 2: add rank
            # elseif j==8
            # 		keyword="cardmat_cardcol_rank_bounds_2_cycle1"
            # 		keyword2="cardmat_cardcol_rank_bounds_2_cycle2"
            #
            # 		constraint=Dict()
            # 		constraint["use_bounds"]=true
            # 		constraint["m_min"] = (1f0./vec(v_max)).^2
            # 		constraint["m_max"] = (1f0./vec(v_min)).^2
            #
            # 		#cardinality on derivatives (column and row wise)
            # 		constraint["use_TD_card_fiber_x"]			= true
            # 		constraint["card_fiber_x"] 						= 2
            # 	  constraint["TD_card_fiber_x_operator"]= "D_x"
            #
            # 		constraint["use_TD_card_fiber_z"]=true
            # 		constraint["card_fiber_z"]=2
            # 		constraint["TD_card_fiber_z_operator"]="D_z"
            #
            # 		#cardinality on derivatives (matrix based)
            # 		constraint["use_TD_card_1"]=true
            # 		constraint["card_1"]=round(Integer,3*0.33*n[1])
            # 		constraint["TD_card_operator_1"]="D_x"
            #
            # 		constraint["use_TD_card_2"]=true
            # 		constraint["card_2"]=round(Integer,3*0.33*n[2])
            # 		constraint["TD_card_operator_2"]="D_z"
            #
            # 		#rank constraint
            # 		constraint["use_rank"]=false
            # 	  constraint["max_rank"]=3

end

  #set up constraints, precompute some things and define projector
  (P_sub,TD_OP,TD_Prop) = setup_constraints(constraint,comp_grid,options.FL)
  (TD_OP,AtA,l,y) = PARSDMM_precompute_distribute(TD_OP,TD_Prop,comp_grid,options)
  proj_intersection = x-> PARSDMM(x,AtA,TD_OP,TD_Prop,P_sub,comp_grid,options)
  # function prj!{T}(dst::T, src::T)
  #   (dst[:],dummy1,dummy2,dymmy3) = proj_intersection(src)
  #   return dst
  # end
   function prj!(input)
     (x,dummy1,dummy2,dymmy3) = proj_intersection(input)
     return x
   end
  v_ini=deepcopy(v0)
  x=CFWI(v_ini,prj!)
  plot_velocity(x,title_str,model,keyword,repmat(xrec,length(zrec),1),zrec,xsrc,repmat(zsrc,length(xsrc),1))

end
