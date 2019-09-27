using DrWatson
quickactivate(joinpath("..",@__DIR__))
include(joinpath(srcdir(),"intro.jl"))
using Turing, MCMCChains
using AugmentedGaussianProcesses
using MLDataUtils, CSV, StatsFuns

likelihood_name = "Laplace"
nSamples = 10000


res = collect_results(datadir("part_1",likelihood_name))
infos = vcat(res.infos...)
valid = findall(x->x==nSamples,infos.nSamples)
chains = unfold_chains.(view(res.chains,valid),nSamples)

function treat_chain(chain)
    burnin = mean(mean.(getindex.(heideldiag(chain),Symbol("Burn-in"))))
    gelman = mean(gelmandiag(chain)[:PSRF])
    acorlag1 = mean(mean.(abs,getindex.(autocor(chain),Symbol("lag 1"))))
    acorlag5 = mean(mean.(abs,getindex.(autocor(chain),Symbol("lag 5"))))
    return DataFrame([[acorlag1], [acorlag5],[burnin], [gelman], ],[:lag1,:lag5,:mixtime,:gelman])
end
results_analysis = vcat(treat_chain.(chains)...)
results_analysis= hcat(infos.alg[valid],results_analysis)

function unfold_chains(chain::DataFrame,nSamples::Int)
    nTot = length(chain[!,1])
    nChains = nTot/nSamples
    fold_chains = []
    for j in 1:nChains
        push!(fold_chains,Chains(reshape(Matrix(chain[Int64.(((j-1)*nSamples+1):(j*nSamples)),:]),nSamples,:,1)))
    end
    chainscat(fold_chains...)
end
