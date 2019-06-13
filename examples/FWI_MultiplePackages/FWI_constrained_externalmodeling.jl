#This is an example of how the SetIntersectionProjection package
#can work together with another package written in an older Julia version

#We use SetIntersectionProjection in Julia 1.1 and WAVEFORM in Julia 0.6
#The idea is as follows: 1) set up experiment in julia 1.1; 2) save velocity model
#and some other info ; 3) call a Julia 0.6 script that loads the model and runs one or a few FWI iterations and saves model again;
# 4) load saved model in Julia 1.1 and project onto intersection; 5) save model....

#This is not mathematically equivalent to running (spectral) projected gradient. It is
#intended to be a practical work-around to deal with incompatible software packages/versions.
#The results obtained using this script are not entirely the same as in the corresponding paper
#because that was generated using spectral projected gradient in julia 0.6

using LinearAlgebra
using JLD
ENV["MPLBACKEND"]="qt5agg"
using PyPlot
#using Waveform

using SetIntersectionProjection

save_dir = pwd()

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
xsrc = 0.0:50.0:L[2];xsrc=convert(Vector{Float64},xsrc)

# y-coordinates of the sources (irrelevant in 2D)
ysrc = [0.0];ysrc=convert(Vector{Float64},ysrc)

# z-coordinates of the sources
zsrc = [10.0];zsrc=convert(Vector{Float64},zsrc)

# x-coordinates of the receivers
xrec = [950.0];xrec=convert(Vector{Float64},xrec)

# y-coordinates of the receivers (irrelevant in 2D)
yrec = [0.0];yrec=convert(Vector{Float64},yrec)

# z-coordinates of the receivers (this is a transmission experiment)
#zrec = 0.0:50.0:L[2];
zrec = 0.0:50.0:L[1];zrec=convert(Vector{Float64},zrec)

# Frequencies to compute
freqs = [3.0 ; 5.0 ; 7.0]# ; 8.0]#[5.0;10.0;15.0];

# Maximum wavelength
λ = vel_background/maximum(freqs);

nsrc,nrec,nfreq = length(xsrc),length(xrec),length(freqs)

# Set up computational parameters
## Size/spacing of computational domain
comp_n = n;
comp_d = d;
comp_o = o;

# Number of PML points
npml = round(Int,λ/minimum(comp_d))*[1 1; 1 1];

# Create velocity model
v0 = vel_background*ones(n...);
v = copy(v0);
#v[div(n[1],3):2*div(n[1],3),div(n[2],3):2*div(n[2],3)] = 1.25*vel_background;
v[61:80,50:70] .= 2400
v = vec(v);
v0 = vec(v0);

function plot_velocity(vs,title_str,n,d,keyword,rec_x,rec_z,src_x,src_z)
    fig = figure(figsize=(4.2, 3.2))
    FS  = 8
    LS  = 6
    TML = 2
    PD  = 2
    #ax1 = axes()
    #vplot = imshow(reshape(vs,model.n...),vmin=minimum(v)-50,vmax=maximum(v)+50,axes=ax1,cmap="jet",extent=[0.0 , model.d[2]*model.n[2] , model.d[1]*model.n[1] , 0.0])
    vplot = imshow(reshape(vs,n[1],n[2]),vmin=minimum(v)-50,vmax=maximum(v)+50,cmap="jet",extent=[0.0 , d[2]*n[2] , d[1]*n[1] , 0.0])
    tick_params(labelsize=LS,length=TML,pad=PD)
    title(title_str,FontSize=FS)
    xlabel("x [m]",FontSize=FS)
    ylabel("z [m]",FontSize=FS)
    zt = 0:20:n[1]
    xt = 0:20:n[2]
    #xticks(xt,round.(Int,xt*model.d[2]))
    #yticks(zt,round.(Int,zt*model.d[1]))
    cbar = colorbar()
    cbar[:ax][:tick_params](labelsize=LS)
    figure;plot(rec_x,rec_z,linewidth=1.0, marker="o",linestyle="",markersize=2)
    figure;plot(src_x,src_z,linewidth=1.0, marker="x",linestyle="","k",markersize=2)
    savefig(joinpath(save_dir,string("CFWI_simple_freq_m_est_",keyword,".eps")),bbox_inches="tight",dpi=300)
    return nothing
end
plot_velocity(v,"a) True velocity",n,d,"true",repeat(xrec,length(zrec)),zrec,xsrc,repeat(zsrc,length(xsrc)))
plot_velocity(v0,"b) Initial velocity",n,d,"initial",repeat(xrec,length(zrec)),zrec,xsrc,repeat(zsrc,length(xsrc)))

# Source weight matrix
Q = Matrix{Float64}(I,nsrc, nsrc)

# Frequency continuation
max_func_evals = 2

vest = v0
#opts1 = deepcopy(opts);

bound_min = minimum(v)-50.0
bound_max = maximum(v)+50.0

v_true    = v
vel_model = deepcopy(v0)
vel_model = save("vel_model.jld","vel_model",vel_model)

#Save info for FWI
FWI_info = save("FWI_info.jld","bound_min",bound_min,"bound_max",bound_max,"nsrc",nsrc,"nfreq",nfreq,"Q",Q,"max_func_evals",max_func_evals,"v_true",v_true,"n",n,"d",d,"o",o,"t0",t0,"f0",f0,"unit",unit,"freqs",freqs,"xsrc",xsrc,"ysrc",ysrc,"zsrc",zsrc,"xrec",xrec,"yrec",yrec,"zrec",zrec,"npml",npml)


# Setup constraints:
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
#FFTW.set_num_threads(2)
options.parallel=false
options.feasibility_only = false
options.zero_ini_guess=true

mutable struct compgrid
  d :: Tuple
  n :: Tuple
end

comp_grid=compgrid((d[1],d[2]),(n[1],n[2]))
constraint_strategy_list=[1 2 3 4 5]#
for j in constraint_strategy_list

  #reset initial model
  vel_model = deepcopy(v0)
  vel_model = save("vel_model.jld","vel_model",vel_model)

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
    title_str="c)"# fiber, matrix grad. card. & rank & bounds"

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

		#rank constraint
    m_min     = 0
    m_max     = 3
    set_type  = "rank"
    TD_OP     = "identity"
    app_mode  = ("matrix","")
    custom_TD_OP = ([],false)
    push!(constraint, set_definitions(set_type,TD_OP,m_min,m_max,app_mode,custom_TD_OP))

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
          title_str="b)"#" fiber, matrix grad. card. & bounds"

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
            title_str="a)"# matrix grad. card. & bounds & rank"

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

  for i=1:8
    #1) run software package in different julia version for a few iterations to update model
    run(`/Applications/Julia-0.6.app/Contents/Resources/julia/bin/julia GetUPDModel.jl`)

    #2) load updated model
    tmp = load("vel_model.jld")
    vel_model = tmp["vel_model"]

    #3) project new model
    vel_model = prj!(vec(vel_model))

    #4) save projected model
    save("vel_model.jld","vel_model",vel_model)
  end

  #load final model
  tmp = load("vel_model.jld")
  vel_model = tmp["vel_model"]
  plot_velocity(vel_model,title_str,n,d,keyword,repeat(xrec,length(zrec)),zrec,xsrc,repeat(zsrc,length(xsrc)))

end #end loop over FWI with various constraint combinations
