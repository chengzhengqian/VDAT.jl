# test for five band case

N_spin_orbital=10
# we first examine the parametrization
VΓη,ηToIdx=cal_VΓη_ηToIdx(N_spin_orbital)

# the question is how to going to parametrize the x
# we first consider the degenerate case
using DataStructures
total_para=length(ηToIdx)
degenerate_case=counter(length.(ηToIdx))
#we need better way to do it
θs=[0.2]
function θ_to_x(θs)
    if(length(θs)==1)
        [sin(θs[1]),cos(θs[1])]
    else
        vcat([sin(θs[1])],cos(θs[1]).*θ_to_x(θs[2:end]))
    end    
end


gradient(x->θ_to_x(x)[end],[0.1,0.2,0.3])
# we can 
VΓη

reduce_para=length(degenerate_case)
Vxfull=zeros(total_para,reduce_para)
start_idx=1
for (idx,size) in enumerate( [degenerate_case[n] for n in 2:10])
    global start_idx
    end_idx=start_idx+size-1
    # print("$(start_idx) $(end_idx)\n")
    Vxfull[start_idx:end_idx,idx].=1
    start_idx=end_idx+1
end
VΓη_reduced=VΓη*Vxfull
or just
x=Vxfull*θ_to_x(rand(8))
# θs=randn(8)
# x=θ_to_x(θs)
# x=[1.0 for i in 1:9]
# VΓη_reduced*x
# before we

function cal_x(θs)
    Vxfull*θ_to_x(θs)
end
function cal_l_G(l_G)
    [l_G for _ in 1:N_spin_orbital]
end
function cal_β(β)
    [[β,β] for _ in 1:N_spin_orbital]
end

G12ασ_min=[0.3 for i in 1:10]
para=[0.3,0.2,rand(8)...,2.0]
l_G12ασ=cal_l_G(para[1])
l_x=para[2]
x=cal_x(para[3:(end-1)])
βασ=cal_β(para[5])
nασ=[0.5 for i in 1:10]
eασ=[es for i in 1:10]

@time cal_energy_direct(l_G12ασ,l_x,x,βασ,nασ,eασ,interaction,G12ασ_min)

# we first try some simple way to pamametrize x
function θ_to_x(θ)
    vcat(θ,[1-sum(θ)])
end

θ_to_x(rand(8))

@time momentum_derivatives=cal_momentum_derivatives(nασ,l_G12ασ,l_x,x,βασ,eασ,G12ασ_min)
@time cal_energy_with_momentum_derivatives(l_G12ασ,l_x,x,βασ,momentum_derivatives,interaction,G12ασ_min)
@time gradient((l_G12ασ,x)->cal_energy_with_momentum_derivatives(l_G12ασ,l_x,x,βασ,momentum_derivatives,interaction,G12ασ_min),l_G12ασ,x)

# we need to generate interaction
"""
α, index for orbital (1,2,..)
σ, index for spin (1,2)  for spin up and down
get_idx(1,1)
get_idx(1,2)
"""
function get_idx(α,σ)
    (α-1)*2+σ
end

"""
N_orbital=5
J=1
interaction=gene_interaction(U,J,N_orbital)
"""
function gene_interaction(U,J,N_orbital)
    interaction=[]
    # introband
    for orb in 1:N_orbital
        push!(interaction,(get_idx(orb,1),get_idx(orb,2),U))
    end
    # interband, opposite spin
    for orb1 in 1:(N_orbital-1)
        for orb2 in (orb1+1):(N_orbital)
            push!(interaction,(get_idx(orb1,1),get_idx(orb2,2),U-2*J))
            push!(interaction,(get_idx(orb1,2),get_idx(orb2,1),U-2*J))
        end
    end
    # interband, same spin
    for orb1 in 1:(N_orbital-1)
        for orb2 in (orb1+1):(N_orbital)
            push!(interaction,(get_idx(orb1,1),get_idx(orb2,1),U-3*J))
            push!(interaction,(get_idx(orb1,2),get_idx(orb2,2),U-3*J))
        end
    end
    interaction
end

# now, we organize the code to implement the minization
U=8.0
J=0.0
interaction=gene_interaction(U,J,5)
x_init=Vxfull*[-0.3,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
G12ασ_min=[0.3 for i in 1:N_spin_orbital]
nασ=[0.5 for i in 1:N_spin_orbital]
e_fn=gene_spline_band("./es_inf.dat")
eασ=[ gene_ϵs(e_fn,nασ[i]) for i in 1:N_spin_orbital]

# l_G,l_x,β,x
para=[0.5,0.1,2.0,x_init...]




function cal_l_G(l_G)
    [l_G for _ in 1:N_spin_orbital]
end
function cal_β(β)
    [[β,β] for _ in 1:N_spin_orbital]
end

function cal_total_energy(para_full)
    l_G12ασ=cal_l_G(para_full[1])
    l_x=para_full[2]
    βασ=cal_β(para_full[3])
    x=para_full[4:end]
    energy=cal_energy_direct(l_G12ασ,l_x,x,βασ,nασ,eασ,interaction,G12ασ_min)[1]
    print("call $(para_full[1:3]),get energy $(energy)\n")
    return energy
end

function cal_total_energy_grad!(dEdpara,para)
    l_G12ασ=cal_l_G(para[1])
    l_x=para[2]
    βασ=cal_β(para[3])
    x=para[4:end]
    momentum_derivatives=cal_momentum_derivatives(nασ,l_G12ασ,l_x,x,βασ,eασ,G12ασ_min)
    function cal_total_energy_with_momentum_derivatives(para)
        l_G12ασ=cal_l_G(para[1])
        l_x=para[2]
        βασ=cal_β(para[3])
        x=para[4:end]
        cal_energy_with_momentum_derivatives(l_G12ασ,l_x,x,βασ,momentum_derivatives,interaction,G12ασ_min)
    end
    dEdpara[:]=gradient(cal_total_energy_with_momentum_derivatives,para)[1]
    # check1=gradient(cal_total_energy_with_momentum_derivatives,para)[1]
    # check2=grad(central_fdm(2, 1), cal_total_energy, para)[1]
    # sum(abs.(check1-check2))
end


bounds = zeros(3, length(para))
bounds[:,1]=[2,0.0,1]      # l_G
bounds[:,2]=[2,0.01,1-1e-4]      # l_x
bounds[:,3]=[1,0.0,0.0]      # β, only lower boundary



optimizer = L_BFGS_B(length(para), 30)
para=para_new
energy_new, para_full_new = optimizer(cal_total_energy, cal_total_energy_grad!, para_full, bounds, m=10, factr=1e3, pgtol=1e-7, iprint=111, maxfun=10, maxiter=10)

# so we first use bruteforce to minimize
regulate_knorm=regulate_knorm_linear
para=[0.30,2.8,-0.001,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
function cal_total_energy_reduced(para)
    l_G12ασ=cal_l_G(para[1])
    βασ=cal_β(para[2])
    x_reduce=para[3:end]
    x=Vxfull*x_reduce
    energy=cal_energy_direct_regulate(l_G12ασ,x,βασ,nασ,eασ,interaction,G12ασ_min)[1]
    print("call $(para[1:3]) get $(energy)\n")
    return energy
end



res=optimize(x->x[1]^2,[0.1])
res=optimize(cal_total_energy_reduced,para,Optim.Options(iterations=1000))
para[2]=0.5
para[3]=2.8
para=res.minimizer
results=Dict()
results[(U,J)]=res
cal_total_energy_reduced(para)
x_reduce=para[4:end]
x=Vxfull*[x_reduce...,1.0+sum(x_reduce)]
para_full=[para[1],0.99,para[2],x...];cal_total_energy(para_full)

cal_Γασ(2,10)
pinv(VΓη)
sum(w02)
U_eff=0.1
p_eff(0.1,5)
function p_eff(U_eff,N)
    exp(-U_eff*N*(N-10)/2)
end
U_eff=4
w2=[p_eff(U_eff,sum(cal_Γασ(i,10))) for i in 1:2^10]
w2=w2/sum(w2)
k=w2-w02
x=pinv(VΓη)*k
x_reduced=pinv(Vxfull)*x
w=cal_w(x,nασ,G12ασ,regulate_knorm)

para=[0.4,3.0,0.1]
U=2.0
U=8.0
U=16.0
U=32.0
J=0.0
interaction=gene_interaction(U,J,5)
function cal_total_energy_reduced(para)
    l_G12ασ=cal_l_G(para[1])
    βασ=cal_β(para[2])
    U_eff=para[3]
    w2=[p_eff(U_eff,sum(cal_Γασ(i,10))) for i in 1:2^10]
    w2=w2/sum(w2)
    k=w2-w02
    x=pinv(VΓη)*k
    energy=cal_energy_direct_regulate(l_G12ασ,x,βασ,nασ,eασ,interaction,G12ασ_min)[1]
    print("call $(para[1:3]) get $(energy)\n")
    return energy
end

interaction=gene_interaction(U,J,5)
res=optimize(cal_total_energy_reduced,para,Optim.Options(iterations=1000))
energy_eff=res.minimum
para=res.minimizer
l_G12ασ=cal_l_G(para[1])
G12ασ_max=[0.5 for i in 1:N_spin_orbital]
G12ασ=(G12ασ_min.*l_G12ασ)+G12ασ_max.*(1.0 .- l_G12ασ)

U_eff=para[3]
w2=[p_eff(U_eff,sum(cal_Γασ(i,10))) for i in 1:2^10]
w2=w2/sum(w2)
k=w2-w02
x=pinv(VΓη)*k
x_reduced=pinv(Vxfull)*x
para_rf=[para[1:2]...,x_reduced[1],x_reduced[3],x_reduced[5],x_reduced[7],x_reduced[9]]

function cal_total_energy_reduced_full(para_rf)
    l_G12ασ=cal_l_G(para_rf[1])
    βασ=cal_β(para_rf[2])
    x_reduced=[para_rf[3],0,para_rf[4],0,para_rf[5],0,para_rf[6],0,para_rf[7]]
    x=Vxfull*x_reduced
    energy=cal_energy_direct_regulate(l_G12ασ,x,βασ,nασ,eασ,interaction,G12ασ_min)[1]
    print("call $(para_rf[1:3]) get $(energy)\n")
    return energy
end
res_rf=optimize(cal_total_energy_reduced_full,para_rf,Optim.Options(iterations=1000))
para_rf=res_rf.minimizer
res_rf.minimum
l_G12ασ=cal_l_G(para_rf[1])
G12ασ_max=[0.5 for i in 1:N_spin_orbital]
G12ασ=(G12ασ_min.*l_G12ασ)+G12ασ_max.*(1.0 .- l_G12ασ)



# now, we consider the case with J
Γ=cal_Γασ(10,10)

function p_eff(U1,U2,U3,Γ)
    s=0
    for orb in 1:5
        s+=U1*Γ[get_idx(orb,1)]*Γ[get_idx(orb,2)]
    end    
    for orb1 in 1:(N_orbital-1)
        for orb2 in (orb1+1):(N_orbital)
            s+=U2*Γ[get_idx(orb1,1)]*Γ[get_idx(orb2,2)]
            s+=U2*Γ[get_idx(orb1,2)]*Γ[get_idx(orb2,1)]
        end
    end
    for orb1 in 1:(N_orbital-1)
        for orb2 in (orb1+1):(N_orbital)
            s+=U3*Γ[get_idx(orb1,1)]*Γ[get_idx(orb2,1)]
            s+=U3*Γ[get_idx(orb1,2)]*Γ[get_idx(orb2,2)]
        end
    end
    μ=(U1+4*U2+4*U3)/2
    N=sum(Γ)
    exp(-s+μ*N)
end

para=[0.3,2.0,1.0,0.5,0.3]
function cal_total_energy_reduced_with_J(para)
    l_G12ασ=cal_l_G(para[1])
    βασ=cal_β(para[2])
    U1,U2,U3=para[3:end]
    w2=[p_eff(U1,U2,U3,cal_Γασ(i,10)) for i in 1:2^10]
    w2=w2/sum(w2)
    k=w2-w02
    x=pinv(VΓη)*k
    # sum(abs.(VΓη*x-k))
    energy=cal_energy_direct_regulate(l_G12ασ,x,βασ,nασ,eασ,interaction,G12ασ_min)[1]
    print("call $(para) get $(energy)\n")
    return energy
end

U=2; J=0.1;
U=2; J=0.2;
U=2; J=0.5;
U=4; J=0.2;
U=4; J=0.4;
interaction=gene_interaction(U,J,5)
# para=[0.3,2.0,1.0,0.5,0.3]
res=optimize(cal_total_energy_reduced_with_J,para,Optim.Options(iterations=1000))
energy_eff=res.minimum
para=res.minimizer
l_G12ασ=cal_l_G(para[1])
G12ασ_max=[0.5 for i in 1:N_spin_orbital]
G12ασ=(G12ασ_min.*l_G12ασ)+G12ασ_max.*(1.0 .- l_G12ασ)

