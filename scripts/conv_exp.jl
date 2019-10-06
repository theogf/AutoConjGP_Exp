using DrWatson
quickactivate(joinpath(@__DIR__,".."))
include(srcdir("intro.jl"))

defaultconvdict = Dict(:time_max=>1e4,:conv_max=>200,:file_name=>"covtype",
                        :nInducing=>50,:nMinibatch=>10,:likelihood=>:Logistic,
                        :doCAVI=>true,:doGD=>!true,:doNGD=>!true)

convdictlist = Dict(:time_max=>1e4,:conv_max=>50000,:file_name=>"CASP",
                        :nInducing=>[100,200,500],:nMinibatch=>[50,100,200],:likelihood=>:Matern32, :doCAVI=>true,:doGD=>true,:doNGD=>true)
convdictlist = dict_list(convdictlist)
tmpfolder = projectdir("tmp","conv_exp",string(convdictlist[1][:file_name]),string(convdictlist[1][:likelihood]))
# map(convergence_exp,convdictlist)
##
files = tmpsave(convdictlist,tmpfolder)
try
    script = scriptsdir("_conv_exp.jl")
    for (i,r) in enumerate(files)
        @show fullpath = joinpath(tmpfolder,r)
        screenname = "Conv_"*string(convdictlist[i][:file_name])*"_"*string(convdictlist[i][:likelihood])*"_"*string(convdictlist[i][:nInducing])*"_"*string(convdictlist[i][:nMinibatch])
        submit = `screen -d -S $screenname -m julia $script $(fullpath)`# $(fullpath)`
        run(submit)
    end
catch e
    # rm(tmpfolder,recursive=true)
    rethrow(e)
end
# rm(tmpfolder,recursive=true)

# convergence_exp()
