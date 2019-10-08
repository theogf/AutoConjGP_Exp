using DrWatson
quickactivate(joinpath("..",@__DIR__))
include(joinpath(srcdir(),"intro.jl"))
using Turing, MCMCChains
using DataFramesMeta
using AugmentedGaussianProcesses
using MLDataUtils, CSV, StatsFuns
##
likelihood_name = "Matern32"
nSamples = 10000
epsilon = 0.1; n_step = 10

res = collect_results(datadir("part_1",likelihood_name),white_list=[:infos])
infos = vcat(res.infos...)
infosHMC = @linq infos |> where(:alg.=="HMC",:epsilon.==epsilon,:n_step.==n_step)
# infosHMC =  @linq infos |> where(:alg.=="GP",:nSamples.==nSamples)
# infosHMC.alg[1] = "HMC"; infosHMC.epsilon[1]=0.1; infosHMC.n_step[1]=10
infosMH =  @linq infos |> where(:alg.=="MH",:nSamples.==nSamples)
infosGS =  @linq infos |> where(:alg.=="GS",:nSamples.==nSamples)
dfresults = []
# paramsHMC = @ntuple nSamples
paramsHMC = @ntuple epsilon nSamples n_step
# chain = DrWatson.wload(datadir("part_1",likelihood_name,savename("GP",paramsHMC,"bson")))[:chains];
chain = DrWatson.wload(datadir("part_1",likelihood_name,savename("HMC",paramsHMC,"bson")))[:chains];
chain = unfold_chains(chain,nSamples);
@info "Chain loaded and unfolded"
dfchain = treat_chain(chain)
push!(dfresults,dfchain)
for alg in ["MH","GS"]
    params = @ntuple nSamples
    chain = DrWatson.wload(datadir("part_1",likelihood_name,savename(alg,params,"bson")))[:chains]
    chain = unfold_chains(chain,nSamples)
    @info "Chain loaded and unfolded"
    dfchain = treat_chain(chain)
    push!(dfresults,dfchain)
end

# push!(dfresults,res)
savedf = deepcopy(dfresults)
results_infos = hcat(vcat(infosHMC,infosMH,infosGS),vcat(dfresults...))


using CSVFiles
save(plotsdir("part_1",likelihood_name,"results.csv"),results_infos)


function treat_chain(chain)
    burnin = mean(mean.(getindex.(heideldiag(chain),Symbol("Burn-in"))));
    gelman = mean(gelmandiag(chain)[:PSRF]);
    autocorchain = autocor(chain,lags=collect(1:10));
    acorlag = zeros(10)
    for i in 1:10
        acorlag[i] = mean(mean.(abs,getindex.(autocorchain,Symbol("lag ",i))));
    end
    vals = vcat([[v] for v in acorlag],[[burnin],[gelman]])
    symbols = vcat([Symbol(:lag,i) for i in 1:10],[:mixtime,:gelman])
    return DataFrame(vals,symbols)
end

function unfold_chains(chain::DataFrame,nSamples::Int)
    nTot = length(chain[!,1])
    @show nVar = length(names(chain))
    nChains = Int64(nTot/nSamples)
    samples = zeros(nSamples,nVar,nChains)
    for j in 1:nChains
        samples[:,:,j] = Matrix(chain[Int64.(((j-1)*nSamples+1):(j*nSamples)),:])
        # push!(fold_chains,Chains(reshape(Matrix(chain[Int64.(((j-1)*nSamples+1):(j*nSamples)),:]),nSamples,:,1)))
    end
    Chains(samples)
end

##
include(srcdir("plots_tools.jl"))

likelihood_name= "Matern32"
data = CSV.read(plotsdir("part_1",likelihood_name,"results.csv"))
epsilon = 0.1
n_steps = 10
lagindices = 8:(8+9)

lagGS = @linq data |> where(:alg.=="GS")
lagGS = Vector(lagGS[1,lagindices])
lagHMC = @linq data |> where(:alg .== "HMC") |> where(:epsilon.==epsilon) |> where(:n_step.==n_steps)
lagHMC = Vector(lagHMC[1,lagindices])
lagMH = @linq data |> where(:alg.=="MH")
lagMH = Vector(lagMH[1,lagindices])

plot(xticks=collect(1:10),xlabel="Lag",ylabel="Correlation",legend=:right,title=likelihood_name*" Likelihood")
plot!(1:10,lagGS,lab="Gibbs (ours)",color=colors[1])
plot!(1:10,lagHMC,lab="HMC",linestyle=color=colors[2])
plot!(1:10,lagMH,lab="MH",linestyle=:dash,color=colors[3]) |> display
savefig(plotsdir("part_1",likelihood_name,"lag_plot.png"))
