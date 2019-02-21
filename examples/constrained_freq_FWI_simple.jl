#simple frequency domain FWI to illustrate PARSDMM
#include("/home/slim/bpeters/.julia/v0.6/WAVEFORM/src/Waveform.jl")
using Waveform

#This example used the Waveform package, which is in julia 0.6 and uses the v06 branch of JOLI (https://github.com/slimgroup/JOLI.jl)
#untill everyting is updated to current julia version, you need to got to (on linux) /.julia/v0.6/JOLI and use: git checkout v06

using JOLI
ENV["MPLBACKEND"]="qt5agg"
using PyPlot
#using OptimPackNextGen
#using opesciSLIM.SLIM_optim
#using Waveform

@everywhere using SetIntersectionProjection

#include("SPGslim.jl")
include(string(Pkg.dir("JUDI"),"/src/Optimization/SPGSlim.jl"))
include(string(Pkg.dir("JUDI"),"/src/Optimization/OptimizationFunctions.jl"))

#data directory for loading and writing results
#save_dir = "/data/slim/bpeters/SetIntersection_data_results"
save_dir = pwd()#"/Volumes/Users/bpeters/Downloads"

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
freqs = [3.0 ; 5.0 ; 7.0]# ; 8.0]#[5.0;10.0;15.0];

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
    #ax1 = axes()
    #vplot = imshow(reshape(vs,model.n...),vmin=minimum(v)-50,vmax=maximum(v)+50,axes=ax1,cmap="jet",extent=[0.0 , model.d[2]*model.n[2] , model.d[1]*model.n[1] , 0.0])
    vplot = imshow(reshape(vs,model.n...),vmin=minimum(v)-50,vmax=maximum(v)+50,cmap="jet",extent=[0.0 , model.d[2]*model.n[2] , model.d[1]*model.n[1] , 0.0])
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
    savefig(joinpath(save_dir,string("CFWI_simple_freq_m_est_",keyword,".png")),bbox_inches="tight",dpi=300)
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
size_freq_batch = 3
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
options.feasibility_only = false
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

constraint_strategy_list=[1 2 3 4 5]#
for j in constraint_strategy_list
  if j==1
    keyword="bounds_only"
    title_str="c) bounds only"

    constraint = Vector{SetIntersectionProjection.set_definitions}()

    #bounds:
    m_min     = minimum(v)-50.0
    m_max     = maximum(v)+50.0
    set_type  = "bounds"
    TD_OP     = "identity"
    app_mode  = ("matrix","")
    custom_TD_OP = ([],false)
    push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

elseif j==2 #various types of cardinality and rank
		keyword="cardmat_cardcol_rank_bounds"
    title_str="c) fiber, matrix grad. card. & rank & bounds"

    constraint = Vector{SetIntersectionProjection.set_definitions}()

    #bounds:
    m_min     = minimum(v)-50.0
    m_max     = maximum(v)+50.0
    set_type  = "bounds"
    TD_OP     = "identity"
    app_mode  = ("matrix","")
    custom_TD_OP = ([],false)
    push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))


		#cardinality on derivatives (column and row wise)
    m_min     = 0
    m_max     = 2
    set_type  = "cardinality"
    TD_OP     = "D_x"
    app_mode  = ("fiber","x")
    custom_TD_OP = ([],false)
    push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))
		# constraint["use_TD_card_fiber_x"]			= true
		# constraint["card_fiber_x"] 						= 2
	  # constraint["TD_card_fiber_x_operator"]= "D_x"

    m_min     = 0
    m_max     = 2
    set_type  = "cardinality"
    TD_OP     = "D_z"
    app_mode  = ("fiber","z")
    custom_TD_OP = ([],false)
    push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

		#cardinality on derivatives (matrix based)
    m_min     = 0
    m_max     = round(Integer,3*0.33*n[2])
    set_type  = "cardinality"
    TD_OP     = "D_x"
    app_mode  = ("matrix","")
    custom_TD_OP = ([],false)
    push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))
		# constraint["use_TD_card_1"]=true
		# constraint["card_1"]=round(Integer,3*0.33*n[2])
		# constraint["TD_card_operator_1"]="D_x"

    m_min     = 0
    m_max     = round(Integer,3*0.33*n[1])
    set_type  = "cardinality"
    TD_OP     = "D_z"
    app_mode  = ("matrix","")
    custom_TD_OP = ([],false)
    push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))
		# constraint["use_TD_card_2"]=true
		# constraint["card_2"]=round(Integer,3*0.33*n[1])
		# constraint["TD_card_operator_2"]="D_z"

		#rank constraint
    m_min     = 0
    m_max     = 3
    set_type  = "rank"
    TD_OP     = "identity"
    app_mode  = ("matrix","")
    custom_TD_OP = ([],false)
    push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))
		# constraint["use_TD_rank_1"]=true
	  # constraint["TD_max_rank_1"]=3
    # constraint["TD_rank_operator_1"]="identity"


  elseif j==3 #true tv and bounds
      keyword="trueTV_bounds"
      title_str="d) true TV & bounds"

      constraint = Vector{SetIntersectionProjection.set_definitions}()

      #bounds:
      m_min     = minimum(v)-50.0
      m_max     = maximum(v)+50.0
      set_type  = "bounds"
      TD_OP     = "identity"
      app_mode  = ("matrix","")
      custom_TD_OP = ([],false)
      push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

      #TV
      (TV,dummy1,dummy2,dummy3)=get_TD_operator(comp_grid,"TV",options.FL)
      m_min     = 0.0
      m_max     = norm(TV*vec(v),1)
      set_type  = "l1"
      TD_OP     = "TV"
      app_mode  = ("matrix","")
      custom_TD_OP = ([],false)
      push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

    elseif j==4 # various types of cardinality
    			keyword="cardmat_cardcol_bounds"
          title_str="b) fiber, matrix grad. card. & bounds"

    			constraint = Vector{SetIntersectionProjection.set_definitions}()

          #bounds:
          m_min     = minimum(v)-50.0
          m_max     = maximum(v)+50.0
          set_type  = "bounds"
          TD_OP     = "identity"
          app_mode  = ("matrix","")
          custom_TD_OP = ([],false)
          push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

    			#cardinality on derivatives (column and row wise)
          m_min     = 0
          m_max     = 2
          set_type  = "cardinality"
          TD_OP     = "D_x"
          app_mode  = ("fiber","x")
          custom_TD_OP = ([],false)
          push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

          m_min     = 0
          m_max     = 2
          set_type  = "cardinality"
          TD_OP     = "D_z"
          app_mode  = ("fiber","z")
          custom_TD_OP = ([],false)
          push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

    			#cardinality on derivatives (matrix based)
          m_min     = 0
          m_max     = round(Integer,3*0.33*n[2])
          set_type  = "cardinality"
          TD_OP     = "D_x"
          app_mode  = ("matrix","")
          custom_TD_OP = ([],false)
          push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

          m_min     = 0
          m_max     = round(Integer,3*0.33*n[1])
          set_type  = "cardinality"
          TD_OP     = "D_z"
          app_mode  = ("matrix","")
          custom_TD_OP = ([],false)
          push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

        elseif j==5
    				keyword="cardmat_bounds"
            title_str="a) matrix grad. card. & bounds & rank"

    				constraint = Vector{SetIntersectionProjection.set_definitions}()

            #bounds:
            m_min     = minimum(v)-50.0
            m_max     = maximum(v)+50.0
            set_type  = "bounds"
            TD_OP     = "identity"
            app_mode  = ("matrix","")
            custom_TD_OP = ([],false)
            push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

    				#cardinality on derivatives (matrix based)
            m_min     = 0
            m_max     = round(Integer,3*0.33*n[2])
            set_type  = "cardinality"
            TD_OP     = "D_x"
            app_mode  = ("matrix","")
            custom_TD_OP = ([],false)
            push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

            m_min     = 0
            m_max     = round(Integer,3*0.33*n[1])
            set_type  = "cardinality"
            TD_OP     = "D_z"
            app_mode  = ("matrix","")
            custom_TD_OP = ([],false)
            push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

            #rank constraint
            m_min     = 0
            m_max     = 3
            set_type  = "rank"
            TD_OP     = "identity"
            app_mode  = ("matrix","")
            custom_TD_OP = ([],false)
            push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))


end #end if block for constraint setup choices

  #set up constraints, precompute some things and define projector
  (P_sub,TD_OP,set_Prop) = setup_constraints(constraint,comp_grid,options.FL)
  (TD_OP,AtA,l,y)        = PARSDMM_precompute_distribute(TD_OP,set_Prop,comp_grid,options)
  options.rho_ini        = ones(length(TD_OP))*10.0
  # for i=1:length(options.rho_ini)
  #   if set_Prop.ncvx[i]==true
  #     options.rho_ini[i]=1.0
  #   end
  # end
  proj_intersection = x-> PARSDMM(x,AtA,TD_OP,set_Prop,P_sub,comp_grid,options)
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

end #end loop over FWI with various constraint combinations
