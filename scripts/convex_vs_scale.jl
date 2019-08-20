using Optim
using LinearAlgebra: dot
using Plots
g(x) = -log(sech(sqrt(x/2)))
g(x) = -log(exp(-sqrt(x/2)))
g(x) = log(1+exp(sqrt(x)))
_obj(w,x) = dot(w,x)/2-g(x)
plot(10.0.^(-(0:0.001:5)),obj,xaxis=:log)
w = 3.0
obj(x) = _obj(w,x)
x₀=Complex(0.0)
result = optimize(obj,0,10000,Brent()); display(result)

##
xrange= -4.98:0.000001:-4.9
plot(xrange,x->real(obj(Complex(x))),lab="Re(Objective)")|> display
# plot!(xrange,x->imag(obj(Complex(x))),lab="Im(Objective)")
plot(xrange,x->real(greal(Complex(x))),lab="Re(g(x))")
plot!(xrange,x->imag(greal(Complex(x))),lab="Im(g(x))") |>display
plot(0:0.001:10,x->log(cosh(sqrt(x))))
