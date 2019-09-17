const AGP = AugmentedGaussianProcesses
## Generated Laplace likelihood
blaplace = 1.0
Claplace()=1/(2blaplace)
glaplaceg(y) = 0.0
αlaplace(y) = y^2
βlaplace(y) = 2*y
γlaplace(y) = 1.0
φlaplace(r) = exp(-sqrt(r)/blaplace)
∇φlaplace(r) = -exp(-sqrt(r)/b)/(2*b*sqrt(r))
lllaplace(y,x) = 1/(2b)*exp(-abs(y-x)/blaplace)
AGP.@augmodel(GenLaplace,Regression,Claplace,glaplace,αlaplace,βlaplace, γlaplace, φlaplace, ∇φlaplace)
Statistics.var(::GenLaplaceLikelihood) = 2*blaplace^2
function AGP.grad_quad(likelihood::GenLaplaceLikelihood{T},y::Real,μ::Real,σ²::Real,inference::Inference) where {T<:Real}
    nodes = inference.nodes*sqrt2*sqrt(σ²) .+ μ
    Edlogpdf = dot(inference.weights,AGP.grad_log_pdf.(likelihood,y,nodes))
    Ed²logpdf =  zero(T)#1/(b * sqrt(twoπ*σ²))
    return -Edlogpdf::T, Ed²logpdf::T
end

## Generated StudentT likelihood
ν = 3.0;
Cstudentt()= gamma(0.5*(ν+1))/(sqrt(ν*π)*gamma(0.5*ν))
gstudentt(y) = 0.0
αstudentt(y) = y^2
βstudentt(y) = 2*y
γstudentt(y) = 1.0
φstudentt(r) = (1+r/ν)^(-0.5*(ν+1))
∇φstudentt(r) = -(0.5*(1+ν)/ν)*(1+r/ν)^(-0.5*(ν+1)-1)
llstudentt(y,x) = pdf(LocationScale(y,1.0,TDist(ν)),x)
AGP.@augmodel(GenStudentT,Regression,Cstudentt,gstudentt,αstudentt,βstudentt, γstudentt, φstudentt, ∇φstudentt)
Statistics.var(::GenStudentTLikelihood) = ν/(ν-2.0)


## Generated logistic likelihood
Clogistic()= 0.5
glogistic(y) = 0.5*y
αlogistic(y) = 0.0
βlogistic(y) = 0.0
γlogistic(y) = 1.0
φlogistic(r) = sech.(0.5*sqrt.(r))
∇φlogistic(r) = -0.25*(sech.(0.5*sqrt.(r))*tanh.(0.5*sqrt.(r)))/(sqrt.(r))
lllogistic(y,x) = logistic(y*x)
AGP.@augmodel(GenLogistic,Classification,Clogistic,glogistic,αlogistic,βlogistic, γlogistic, φlogistic, ∇φlogistic)
AGP.hessian_log_pdf(::GenLogisticLikelihood{T},y::Real,f::Real) where {T<:Real} = -exp(y*f)/AGP.logistic(-y*f)^2
AGP.grad_log_pdf(::GenLogisticLikelihood{T},y::Real,f::Real) where {T<:Real} = y*AGP.logistic(-y*f)

## Generated Matern32 likelihood
Cmatern32()= sqrt(3.0)/4.0
gmatern32(y) = 0.0
αmatern32(y) = y^2
βmatern32(y) = 2*y
γmatern32(y) = 1.0
φmatern32(r) = (1+sqrt(3*r))*exp(-sqrt(3*r))
∇φmatern32(r) = -1.5*exp(-sqrt(3*r))
llmatern32(y,x) = sqrt(3.0)/4.0*(1.0+sqrt(3*abs2(x-y)))*exp(-sqrt(3*abs2(x-y)))
AGP.@augmodel(GenMatern32,Regression,Cmatern32,gmatern32,αmatern32,βmatern32, γmatern32, φmatern32, ∇φmatern32)
Statistics.var(::GenMatern32Likelihood) = 4.0/3.0
