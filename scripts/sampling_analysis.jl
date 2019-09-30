using DrWatson
quickactivate(joinpath("..",@__DIR__))
include(joinpath(srcdir(),"intro.jl"))
using Turing, MCMCChains
using AugmentedGaussianProcesses
using MLDataUtils, CSV, StatsFuns

likelihood_name = "Laplace"
nSamples = 10000

res = collect_results(datadir("part_1",likelihood_name),white_list=[:infos])
infos = vcat(res.infos...)
valid = findall(x->x==nSamples,infos.nSamples)
dfresults = []
# params = @ntuple nSamples
# chain = DrWatson.wload(datadir("part_1",likelihood_name,savename("GS",params,"bson")))[:chains]
@progress for v in valid
    @info "Processing alg $(infos.alg[v]), epsilon $(infos.epsilon[v]), n_step $(infos.n_step[v])"
    if infos.alg[v] == "HMC"
        epsilon = infos.epsilon[v]; n_step = infos.n_step[v]
        paramsHMC = @ntuple epsilon nSamples n_step
        chain = DrWatson.wload(datadir("part_1",likelihood_name,savename("HMC",paramsHMC,"bson")))[:chains];
        chain = unfold_chains(chain,nSamples);
        @info "Chain loaded and unfolded"
        dfchain = treat_chain(chain)
        push!(dfresults,dfchain)
    else
        params = @ntuple nSamples
        chain = DrWatson.wload(datadir("part_1",likelihood_name,savename(infos.alg[v],params,"bson")))[:chains]
        chain = unfold_chains(chain,nSamples)
        @info "Chain loaded and unfolded"
        dfchain = treat_chain(chain)
        push!(dfresults,dfchain)
    end
    # chain = unfold_chains.(view(res.chains,valid),nSamples)
end

hcat(infos[:alg,:epsilon,:n_step,:time],vcat(dfresults...))

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
results_analysis = vcat(treat_chain.(chains)...)
results_analysis= hcat(infos.alg[valid],results_analysis)
unsafesave()


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

@time unfolded_chain =  unfold_chains(chain,nSamples);
@time Chains(rand(10000,500,5));
