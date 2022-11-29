# some code to initialize the calculation
# move them to test file later

N_spin_orbital=4
U=4.0
interaction=[(1,2,U),(1,3,U),(1,4,U),(2,3,U),(2,4,U),(3,4,U)]
nασ=[0.5,0.5,0.5,0.5]
G12ασ,x,Δαασ,βασ=load_para_two_band_half(U)
e_fn=gene_spline_band("./es_inf.dat")
es=gene_ϵs(e_fn,nασ[1])
eασ=[ es for _ in 1:N_spin_orbital]
para_init=[-0.3,x[1],x[7],x[11],βασ[1][1]]
para=[-0.3,x[1],x[7],x[11],βασ[1][1]]
U=6.0
U=10.0
interaction=[(1,2,U),(1,3,U),(1,4,U),(2,3,U),(2,4,U),(3,4,U)]

function cal_energy_safe(para)
    global θ_G12ασ,x,βασ
    try
        θ_G12ασ,x,βασ=[para[1] for _ in 1:4],[[para[2] for _ in 1:6]...,[para[3] for _ in 1:4]...,para[4]],[[para[5],para[5]] for _ in 1:4]
        cal_energy_direct(θ_G12ασ,x,βασ,nασ,eασ,interaction,regulate_knorm_3)[1]
    catch
        error("for $([θ_G12ασ,x,βασ])\n")
    end
end

θ_G12ασ,x,βασ=[[-0.4337838957543188, -0.4337838957543188, -0.4337838957543188, -0.4337838957543188], [-0.1309170746776563, -0.1309170746776563, -0.1309170746776563, -0.1309170746776563, -0.1309170746776563, -0.1309170746776563, -5.657319463772557e-10, -5.657319463772557e-10, -5.657319463772557e-10, -5.657319463772557e-10, 1.5710049014781928], [[9.840733802632716, 9.840733802632716], [9.840733802632716, 9.840733802632716], [9.840733802632716, 9.840733802632716], [9.840733802632716, 9.840733802632716]]]

res_LBFGS_direct=optimize(para->cal_energy_safe(para),para_init,LBFGS())
res_LBFGS_direct.minimum
para_init=res_LBFGS_direct.minimizer


res_NM_direct=optimize(para->cal_energy_direct([para[1] for _ in 1:4],[[para[2] for _ in 1:6]...,[para[3] for _ in 1:4]...,para[4]],[[para[5],para[5]] for _ in 1:4],nασ,eασ,interaction,regulate_knorm_3)[1],para_init)
para_init=res_NM_direct.minimizer
para=res_NM_direct.minimizer
res_NM_direct.minimum

# we now try to use LBFGSB to minimize the energy, we need to change how we parametrize the system, i.e, we have two constraint on x and G
U=4.0
interaction=[(1,2,U),(1,3,U),(1,4,U),(2,3,U),(2,4,U),(3,4,U)]
nασ=[0.5,0.5,0.5,0.5]
G12ασ,x,Δαασ,βασ=load_para_two_band_half(U)

l_G12αα=[0.5,0.5,0.5,0.5]
"""
using angle to parametrize x for 2 band degenerat case
θ1=-0.4
θ2=0.0
"""
function cal_x(θ1,θ2)
    x_1_6=sin(θ1)
    x_7_10=cos(θ1)*sin(θ2)
    x_11=cos(θ1)*cos(θ2)
    [[x_1_6 for _ in 1:6]...,[x_7_10 for _ in 1:4]...,x_11]
end

function cal_l_G(l_G)
    [l_G for _ in 1:4]
end

"""
cal_β(0.1)
"""
function cal_β(β)
    [[β,β] for _ in 1:4]
end

para=[0.8,0.2,-0.4,0.0,3.0]
"""
for the two band case
"""
function cal_energy_para(para)
    # print(" call $(para)\n")
    l_G12ασ=cal_l_G(para[1])
    l_x=para[2]
    x=cal_x(para[3],para[4])
    βασ=cal_β(para[5])
    cal_energy_direct(l_G12ασ,l_x,x,βασ,nασ,eασ,interaction,G12ασ_min)[1]
end

function convert_from_para(para)
    l_G12ασ=cal_l_G(para[1])
    l_x=para[2]
    x=cal_x(para[3],para[4])
    βασ=cal_β(para[5])
    l_G12ασ,l_x,x,βασ
end

U=4.0
para=[0.8,0.2,-0.4,0.1,2.0]
interaction=[(1,2,U),(1,3,U),(1,4,U),(2,3,U),(2,4,U),(3,4,U)]


energy_check1=cal_energy_para(para)
@time grad_check1=grad(central_fdm(2, 1), cal_energy_para, para)[1]

l_G12ασ,l_x,x,βασ=convert_from_para(para)
momentum_derivatives=cal_momentum_derivatives(nασ,l_G12ασ,l_x,x,βασ,eασ,G12ασ_min)
cal_energy_with_momentum_derivatives(l_G12ασ,l_x,x,βασ,momentum_derivatives,interaction,G12ασ_min)
@time grad_check2=grad(central_fdm(2, 1), cal_energy_para_with_momentum_derivatives, para)[1]
grad_check1-grad_check2
grad_check1-grad_chekc3
#
gradient(x->x^2,1.0)
@time grad_chekc3=gradient(cal_energy_para_with_momentum_derivatives,para)[1]
gradient(l_G12ασ->cal_energy_with_momentum_derivatives(l_G12ασ,l_x,x,βασ,momentum_derivatives,interaction,G12ασ_min),l_G12ασ)
s1=cal_energy_with_momentum_derivatives(l_G12ασ.+0.1,l_x,x,βασ,momentum_derivatives,interaction,G12ασ_min)
s2=cal_energy_with_momentum_derivatives(l_G12ασ.+0.4,l_x,x,βασ,momentum_derivatives,interaction,G12ασ_min)

y,dy=Zygote._pullback(x->x*10,1.0)
dy(1)
l_G12ασ=[0.4 for i in 1:4 ]
function cal_energy_para_with_momentum_derivatives(para)
    # print(" call $(para)\n")
    l_G12ασ=cal_l_G(para[1])
    l_x=para[2]
    x=cal_x(para[3],para[4])
    βασ=cal_β(para[5])
    cal_energy_with_momentum_derivatives(l_G12ασ,l_x,x,βασ,momentum_derivatives,interaction,G12ασ_min)[1]
    # cal_energy_direct(l_G12ασ,l_x,x,βασ,nασ,eασ,interaction,G12ασ_min)[1]
end


G12ασ_min=[0.4 for i in 1:N_spin_orbital]

using LBFGSB
using FiniteDifferences

optimizer = L_BFGS_B(5, 17)
grad(central_fdm(2, 1), cal_energy_para, para)[1]

bounds = zeros(3, 5)
#  0->unbounded, 1->only lower bound, 2-> both lower and upper bounds, 3->only upper bound
bounds[:,1]=[2,0.0,1]      # l_G
bounds[:,2]=[2,0.0,1-1e-3]      # l_x
bounds[:,5]=[1,0.0,0.0]      # β, only lower boundary

function cal_energy_grad!(d,para)
    d[:]=grad(central_fdm(2, 1), cal_energy_para, para)[1]
end

U=4.0
U=6.0
U=8.0
U=16.0
U=9.0
U=10.0
U=20.0
U=1.0
U=4.0
para=[0.8,0.2,-0.4,0.0,3.0]
interaction=[(1,2,U),(1,3,U),(1,4,U),(2,3,U),(2,4,U),(3,4,U)]
G12ασ_min=[0.35 for _ in 1:4]
energy_new, para = optimizer(cal_energy_para, cal_energy_grad!, para, bounds, m=10, factr=1e3, pgtol=1e-7, iprint=111, maxfun=1500, maxiter=1500)

res=optimize(x->x[1]^x[1],[0.1])
res=optimize(β->cal_energy_para([para[1:4]...,β[1]]),[para[5]])
res.minimizer
para=[para[1:4]...,res.minimizer[1]]
cal_energy_para(para)
para[5]
G12ασ_min=[0.40 for _ in 1:4]
res=optimize(lG12->cal_energy_para([lG12[1],para[2:end]...]),[0.1])
res.minimizer
para=[res.minimizer[1],para[2:end]...]

