using DrWatson
quickactivate(joinpath(@__DIR__,".."))
include(srcdir("intro.jl"))
using CSV, AugmentedGaussianProcesses
using PyCall
gpflow = pyimport("gpflow")
tf = pyimport("tensorflow")
defaultconvdict = Dict(:time_max=>1e4,:conv_max=>200,:file_name=>"covtype",
                        :nInducing=>50,:nMinibatch=>10,:likelihood=>:Logistic,
                        :doCAVI=>true,:doGD=>!true,:doNGD=>!true)

convdictlist = Dict(:time_max=>1e4,:conv_max=>100,:file_name=>"heart",
                        :nInducing=>[10,20,50],:nMinibatch=>[50,100,200],:likelihood=>:Logistic, :doCAVI=>true,:doGD=>true,:doNGD=>true)
convdictlist = dict_list(convdictlist)
tmpfolder = projectdir("tmp","conv_exp",string(convdictlist[1][:file_name]),string(convdictlist[1][:likelihood]))
# map(convergence_exp,convdictlist)
##
files = tmpsave(convdictlist,tmpfolder)
try
    script = scriptdir("_conv_exp.jl")
    for r in files
        @show fullpath = "$(tmpfolder*r)"
        submit = `julia $script $(fullpath)`
        run(submit)
    end
catch e
    rm(tmpfolder,recursive=true)
    rethrow(e)
end
rm(tmpfolder,recursive=true)

# convergence_exp()
