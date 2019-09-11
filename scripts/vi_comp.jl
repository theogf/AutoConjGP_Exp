using AugmentedGaussianProcesses
using MLDataUtils, DelimitedFiles
using DrWatson
quickactivate(joinpath(@__DIR__,".."))
include(joinpath(srcdir(),"intro.jl"))
defaultdic = Dict(:nSamples=>1000,:nIterA=>200,:nIterVI=>100,
                    :file_name=>"housing.csv",:kernel=>RBFKernel,
                    :likelihood=>LaplaceLikelihood(1.0),
                    :kfold=>10,:nfold=>2,:GS=>true,:AVI=>true,:VI=>true)
function run_vi_exp(dict=defaultdic)
# dict=defaultdic
    ## Parameters and data
    nSamples = dict[:nSamples];
    nIterA = dict[:nIterA];
    nIterVI = dict[:nIterVI];
    file_name = dict[:file_name];
    data = readdlm(joinpath(datadir(),"exp_raw/",file_name),',');
    y = data[:,1]; X = data[:,2:end]; (N,nDim) = size(X)
    rescale!(y,obsdim=1); rescale!(X,obsdim=1);
    l = initial_lengthscale(X)
    kerneltype = dict[:kernel]
    kernel = kerneltype(l)
    burnin=1
    likelihood = dict[:likelihood]
    kfold = dict[:kfold]
    nfold = dict[:nfold]
    doGS = dict[:GS]
    doAVI = dict[:AVI]
    doVI = dict[:VI]
    predic_results = [DataFrame() for _ in 1:nfold]
    latent_results = [DataFrame() for _ in 1:nfold]
    i=1
    for ((X_train,y_train),(X_test,y_test)) in kfolds((X,y),obsdim=1,k=kfold)
        ## Computing truth
        @info "Fold $i/$nfold, X_train:$(size(X))"
        predic_results[i] = hcat(predic_results[i],DataFrame([y_test],[:y_test]))
        if doGS
            @info "Starting sampling via Gibbs sampler"
            gsmodel = VGP(X_train,y_train,kernel,likelihood,GibbsSampling(samplefrequency=1,nBurnin=0),verbose=2,optimizer=false)
            train!(gsmodel,iterations=nSamples+1)
            y_gs,sig_gs = proba_y(gsmodel,X_test)
            predic_results[i] = hcat(predic_results[i],DataFrame([y_gs,sig_gs],[:y_gs,:sig_gs]))
            latent_results[i] = hcat(latent_results[i],DataFrame([gsmodel.μ[1],diag(gsmodel.Σ[1])],[:μ_gs,:Σ_gs]))
        end
        ## Computing the augmented model
        if doAVI
            @info "Starting training of augmented model"
            amodel = VGP(X_train,y_train,kernel,likelihood,AnalyticVI(),verbose=2,optimizer=false)
            train!(amodel,iterations=nIterA)
            y_a,sig_a = proba_y(amodel,X_test)
            predic_results[i] = hcat(predic_results[i],DataFrame([y_a,sig_a],[:y_a,:sig_a]))
            latent_results[i] = hcat(latent_results[i],DataFrame([amodel.μ[1],diag(amodel.Σ[1])],[:μ_a,:Σ_a]))
        end
        ## Computing the classical model
        if doVI
            try
                @info "Starting training of classical model"
                vimodel = VGP(X_train,y_train,kernel,likelihood,QuadratureVI(),verbose=2,optimizer=false)
                train!(vimodel,iterations=nIterVI)
                y_vi,sig_vi = proba_y(vimodel,X_test)
                predic_results[i] = hcat(predic_results[i],DataFrame([y_vi,sig_vi],[:y_vi,:sig_vi]))
                latent_results[i] = hcat(latent_results[i],DataFrame([vimodel.μ[1],diag(vimodel.Σ[1])],[:μ_vi,:Σ_vi]))
            catch e
                continue #TODO treat error
            end
        end
        i+=1
        if i > nfold
            break;
        end
    end
    return predic_results, latent_results
end

p_results, l_results = run_vi_exp()
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
merge_results(p_results)
