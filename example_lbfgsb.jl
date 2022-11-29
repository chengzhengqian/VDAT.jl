# example using LBFGSB
using Pkg
Pkg.add("LBFGSB")

function f(x)
    y = 0.25 * (x[1] - 1)^2
    for i = 2:length(x)
        y += (x[i] - x[i-1]^2)^2
    end
    4y
end

function g!(z, x)
    n = length(x)
    t₁ = x[2] - x[1]^2
    z[1] = 2 * (x[1] - 1) - 1.6e1 * x[1] * t₁
    for i = 2:n-1
        t₂ = t₁
        t₁ = x[i+1] - x[i]^2
        z[i] = 8 * t₂ - 1.6e1 * x[i] * t₁
    end
    z[n] = 8 * t₁
end

using LBFGSB
optimizer = L_BFGS_B(1024, 17)
n = 25  # the dimension of the problem
x = fill(Cdouble(3e0), n)  # the initial guess
# set up bounds
bounds = zeros(3, n)
for i = 1:n
    bounds[1,i] = 0  # represents the type of bounds imposed on the variables:
                     #  0->unbounded, 1->only lower bound, 2-> both lower and upper bounds, 3->only upper bound
    bounds[2,i] = isodd(i) ? 1e0 : -1e2  #  the lower bound on x, of length n.
    bounds[2,i] =0.3
    bounds[3,i] = 0.7  #  the upper bound on x, of length n.
end

fout, xout = optimizer(f, g!, x, bounds, m=5, factr=1e7, pgtol=1e-5, iprint=-1, maxfun=15000, maxiter=15000)


# test for finite difference
Pkg.add("FiniteDifferences")
using FiniteDifferences
a = randn(3, 3); a = a * a'
f(x) = 0.5 * x' * a * x
x = randn(3)

grad(central_fdm(2, 1), f, x)[1] - a * x
grad(central_fdm(2, 1), f, x)[1]

