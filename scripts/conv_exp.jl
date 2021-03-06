using DrWatson
quickactivate(joinpath(@__DIR__, ".."))
include(srcdir("intro.jl"))
reg_data = ["CASP", "airline"]
class_data = ["covtype", "SUSY", "HIGGS"]
defaultconvdict = Dict(
    :time_max => 1e4,
    :conv_max => 200,
    :file_name => "covtype",
    :nInducing => 50,
    :nMinibatch => 10,
    :likelihood => :Logistic,
    :doCAVI => true,
    :doGD => !true,
    :doNGD => !true,
)

convdictlist = Dict(
    :time_max => 1e4,
    :conv_max => 50000,
    :file_name => "CASP",
    :nInducing => [100, 200, 500],
    :nMinibatch => [50, 100, 200],
    :likelihood => :Matern32,
    :doCAVI => true,
    :doGD => true,
    :doNGD => true,
)
convdictlist = dict_list(convdictlist)
tmpfolder = projectdir(
    "tmp",
    "conv_exp",
    string(convdictlist[1][:file_name]),
    string(convdictlist[1][:likelihood]),
)
# map(convergence_exp,convdictlist)
##
files = tmpsave(convdictlist, tmpfolder)
try
    script = scriptsdir("_conv_exp.jl")
    for (i, r) in enumerate(files)
        fullpath = joinpath(tmpfolder, r)
        screenname =
            "Conv_" *
            string(convdictlist[i][:file_name]) *
            "_" *
            string(convdictlist[i][:likelihood]) *
            "_" *
            string(convdictlist[i][:nInducing]) *
            "_" *
            string(convdictlist[i][:nMinibatch])
        global create_screen = `screen -dmS $screenname`# $(fullpath)`
        global submit = `screen -r $screenname -p 0 -X stuff "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1^M"`
        global submit2 = `screen -r $screenname -p 0 -X stuff "julia $script $(fullpath) ^M"`
        @info "Created screen $screenname"
        run(create_screen)
        sleep(0.1)
        run(submit)
        run(submit2)
        @info "Started processes"
    end
catch e
    # rm(tmpfolder,recursive=true)
    rethrow(e)
end
# rm(tmpfolder,recursive=true)

# convergence_exp()
