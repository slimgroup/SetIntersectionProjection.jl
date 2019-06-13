#input: model
#run WAVEFORM package for seismic full-waveform inversion for 1 or a few iterations
#output: updated model

#This package only runs in Julia 0.6
#This example used the Waveform package, which is in julia 0.6 and uses the v06 branch of JOLI (https://github.com/slimgroup/JOLI.jl)
#untill everyting is updated to current julia version, you need to got to (on linux) /.julia/v0.6/JOLI and use: git checkout v06

#this example also uses two scripts from the JUDI package (see below)

using Waveform
using JOLI
using JLD

include(string(Pkg.dir("JUDI"),"/src/Optimization/SPGSlim.jl"))
include(string(Pkg.dir("JUDI"),"/src/Optimization/OptimizationFunctions.jl"))

#load model
tmp = load("vel_model.jld")
vel_model = tmp["vel_model"]

#load other information for waveform inversion (frequencies, sources, options, bound constraints)
FWI_info = load("FWI_info.jld")
#unpack
bound_min       = FWI_info["bound_min"]
bound_max       = FWI_info["bound_max"]
nsrc            = FWI_info["nsrc"]
nfreq           = FWI_info["nfreq"]
Q               = FWI_info["Q"]
max_func_evals  = FWI_info["max_func_evals"]
v_true          = FWI_info["v_true"]
Q               = FWI_info["Q"]
n               = FWI_info["n"]
d               = FWI_info["d"]
o               = FWI_info["o"]
t0              = FWI_info["t0"]
f0              = FWI_info["f0"]
unit            = FWI_info["unit"]
freqs           = FWI_info["freqs"]
xsrc            = FWI_info["xsrc"]
ysrc            = FWI_info["ysrc"]
zsrc            = FWI_info["zsrc"]
xrec            = FWI_info["xrec"]
yrec            = FWI_info["yrec"]
zrec            = FWI_info["zrec"]
npml            = FWI_info["npml"]

model = Model{Int64,Float64}(n,d,o,t0,f0,unit,freqs,xsrc,ysrc,zsrc,xrec,yrec,zrec);
(z,x) = odn_to_grid(o,d,n);
## Size/spacing of computational domain
comp_n = n;
comp_d = d;
comp_o = o;

# Linear solver options
lsopts = LinSolveOpts(solver=:lufact);

# Misfit function for the objective
misfit = least_squares;

# PDE scheme
scheme = Waveform.helm2d_chen9p;

# Whether to remove elements in the PML when restricting to the model domain (if false, will use stacking)
cut_pml = true;

# If true, use an implicit matrix representation (false for 2D, the matrices are small enough to form explicitly)
implicit_matrix = false;

# Binary mask of the sources and frequencies to compute
srcfreqmask = trues(nsrc,nfreq);

# Computational options type
opts = PDEopts{Int64,Float64}(scheme,comp_n,comp_d,comp_o,cut_pml,implicit_matrix,npml,misfit,srcfreqmask,lsopts);

#frequency partition
size_freq_batch=length(freqs)
overlap = 1
freq_partition = partition(nfreq,size_freq_batch,overlap)

#if data was not generated previously, generate it here
if isfile("D.jld")==true
  tmp = load("D.jld")
  D = tmp["D"]
else
  D = forw_model(v_true,Q,model,opts)
  save("D.jld","D",D)#save data
end

#proj! = (xproj,x)->project_bounds!(x,bound_min,bound_max,xproj);
#proj! = in -> 1.0.*in
proj! = in -> min.(max.(in,bound_min),bound_max)

options_SPG = spg_options(verbose=3, maxIter=max_func_evals, memory=3,suffDec=1f-6)
options_SPG.progTol=1e-6
options_SPG.testOpt=false
options_SPG.interp=2

for j in 1:size(freq_partition,2)
    fbatch = freq_partition[:,j]
    srcfreqmask = falses(nsrc,nfreq)
    srcfreqmask[:,fbatch] = true
    opts.srcfreqmask = srcfreqmask
    obj! = construct_pde_misfit(vel_model,Q,D,model,opts,batch_mode=false)
    function objective_function(in,obj!)
      f=0.0
      g=zeros(length(in))
      f=obj!(in,g);
      return f,g
    end
    f_obj = in -> objective_function(in,obj!)
    #vel_model = spg(obj!,proj!,vel_model,3,maxfc=max_func_evals)
    vel_model, fsave, funEvals= minConf_SPG(f_obj, vec(vel_model), proj!, options_SPG)
end

#save new updated model
save("vel_model.jld","vel_model",vel_model)
