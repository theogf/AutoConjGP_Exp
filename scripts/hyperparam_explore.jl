using AugmentedGaussianProcesses
using MLDataUtils, DelimitedFiles
using DrWatson
using Base.Threads
quickactivate(joinpath(@__DIR__,".."))
include(joinpath(srcdir(),"intro.jl"))
defaultdicthp = Dict(:nIterA=>30,:nIterVI=>500,
                    :file_name=>"heart.csv",:kernel=>RBFKernel,
                    :likelihood=>LogisticLikelihood(),
                    :AVI=>true,:VI=>true,:l=>1.0,:v=>1.0)
nGrid = 50
list_l = 10.0.^(range(0,4,length=nGrid))
list_v = 10.0.^range(0,4,length=nGrid)
alldicthp = Dict(:nIterA=>30,:nIterVI=>800,
                    :file_name=>"heart.csv",:kernel=>RBFKernel,
                    :likelihood=>LogisticLikelihood(),
                    :AVI=>true,:VI=>false,:l=>list_l,:v=>list_v)
listdict_hp = dict_list(alldicthp)
function run_vi_exp_hp(dict::Dict=defaultdicthp)
    # Parameters and data
    nIterA = dict[:nIterA];
    nIterVI = dict[:nIterVI];
    file_name = dict[:file_name];
    data = readdlm(joinpath(datadir(),"exp_raw/"*file_name),',');
    y = data[:,1]; X = data[:,2:end]; N = size(X,1)
    rescale!(X,obsdim=1);
    if length(unique(y))!=2
        rescale!(y,obsdim=1);
    end
    # l = initial_lengthscale(X)
    ktype = dict[:kernel]
    l = dict[:l]
    v = dict[:v]
    ker = ktype(l,variance=v)
    ll = dict[:likelihood]
    kfold = 3
    nfold = 1
    doAVI = dict[:AVI]
    doVI = dict[:VI]
    predic_results = DataFrame()
    latent_results = DataFrame()
    i = 1
    X_train = X_test = X
    y_train = y_test = y
    elbo_a = 0.0; elbo_vi = 0.0
    metric_a = 0.0; metric_vi = 0.0
    nll_a = 0.0; nll_vi = 0.0
    for ((X_train,y_train),(X_test,y_test)) in kfolds((X,y),obsdim=1,k=kfold)
        # Computing truth
        @info "Using l=$l and v=$v, X_train:$(size(X))"
        predic_results = hcat(predic_results,DataFrame([y_test],[:y_test]))
        # Computing the augmented model
        if doAVI
            @info "Starting training of augmented model";
            amodel = VGP(X_train,y_train,ker,ll,AnalyticVI(),verbose=2,optimizer=false)
            train!(amodel,iterations=nIterA)
            y_a,sig_a = proba_y(amodel,X_test)
            predic_results = hcat(predic_results,DataFrame([y_a,sig_a],[:y_a,:sig_a]))
            latent_results = hcat(latent_results,DataFrame([amodel.μ[1],diag(amodel.Σ[1])],[:μ_a,:Σ_a]))
            elbo_a = ELBO(amodel); metric_a = metric(ll,y_test,y_a,sig_a); nll_a = nll(ll,y_test,y_a,sig_a);
        end
        # Computing the classical model
        if doVI
            try
                @info "Starting training of classical model";
                vimodel = VGP(X_train,y_train,ker,ll,QuadratureVI(),verbose=2,optimizer=false)
                train!(vimodel,iterations=nIterVI)
                y_vi,sig_vi = proba_y(vimodel,X_test)
                predic_results = hcat(predic_results,DataFrame([y_vi,sig_vi],[:y_vi,:sig_vi]))
                latent_results = hcat(latent_results,DataFrame([vimodel.μ[1],diag(vimodel.Σ[1])],[:μ_vi,:Σ_vi]))
                elbo_vi = ELBO(vimodel); metric_vi=  metric(ll,y_test,y_vi,sig_vi); nll_vi=  nll(ll,y_test,y_vi,sig_vi);
            catch e
                # rethrow(e) #TODO treat error
                @show i+=1
                if i > nfold
                    break;
                else
                    continue
                end
            end
        end
        i+=1
        if i > nfold
            break;
        end
    end
    diff_elbo = elbo_vi - elbo_a
    analysis_results = DataFrame([[l],[v],[elbo_a],[elbo_vi],[diff_elbo],[metric_a],[metric_vi],[nll_a],[nll_vi]],[:LENGTHSCALE,:VARIANCE,:ELBO_A,:ELBO_VI,:DIFF_ELBO,:METRIC_A,:METRIC_VI,:NLL_A,:NLL_VI])
    likelihood = nameof(typeof(ll))
    params = @ntuple(likelihood,l,v)
    @tagsave(
        datadir("hp_search",file_name,savename(params,"bson")),
        @dict params predic_results latent_results analysis_results
    )
    return predic_results, latent_results, analysis_results
end
run_vi_exp_hp()
map(run_vi_exp_hp,listdict_hp)


res = collect_results(datadir("hp_search","heart.csv"))
results = vcat(res[:analysis_results]...)
names(res[:analysis_results][1])
using Plots

contourf(log10.(list_l),log10.(list_v),reshape(results[:ELBO_A],nGrid,nGrid))
contourf(log10.(list_l),log10.(list_v),reshape(results[:ELBO_VI],nGrid,nGrid))
contourf(log10.(list_l),log10.(list_v),reshape(results[:METRIC_A],nGrid,nGrid))
contourf(log10.(list_l),log10.(list_v),reshape(results[:METRIC_VI],nGrid,nGrid))
contourf(log10.(list_l),log10.(list_v),reshape(results[:NLL_A],nGrid,nGrid))
contourf(log10.(list_l),log10.(list_v),reshape(results[:NLL_VI],nGrid,nGrid))
function merge_results(results::Vector{DataFrame})
    n = length(results)
    if n ==1
        return results[1]
    else
        colnames = names(results[1])
        m_results = DataFrame()
        for colname in colnames
            m = mean(results[i][colname] for i in 1:n)
            v= var.(eachrow(hcat([results[i][colname] for i in 1:n]...)))
            m_results = hcat(m_results,DataFrame([m,v],[Symbol(colname,"_μ"),Symbol(colname,"_σ")]))
        end
        return m_results
    end
end
# merge_results(p_results)
