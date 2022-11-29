# local part for scheme1. Currently, we focus on scheme 1 (G12,β), though scheme 2 (Δ,A) should share most of the code

function cal_p_g12_mat(G12ασ::Array)
    pmatwασ=cal_pmatw.(G12ασ)
    g12matwSασ=cal_g12matwS.(G12ασ)
    pmatwασ,g12matwSασ
end

"""
to double check
"""
function cal_g11_mat(G12ασ::Array)
    cal_g11matwS.(G12ασ)    
end

"""
Gfull=[g012,g013,g023,g031,g032,g033]
"""
function cal_g33_mat_(Gfull)
    cal_g33matw(Gfull[1],Gfull[2],Gfull[3],Gfull[4],Gfull[5],Gfull[6])
end

function cal_g33_mat(Gfullασ)
    cal_g33_mat_.(Gfullασ)
end


function cal_neffασ(nασ,G12ασ)
    cal_neffInn.(nασ,G12ασ)
end


function cal_w02_Γ(Γ,N_spin_orbital,neffασ,peffασ)
    Γασ=cal_Γασ(Γ,N_spin_orbital)
    prod(Γασ.*neffασ+(1.0 .- Γασ).*peffασ)
end

"""
compute the non-interacting w02, with given (effective) denstiy constraint
cal_w02([0.5,0.5])
cal_w02([0.5,0.5,0.5,0.5])
"""
function cal_w02(neffασ)
    peffασ=1.0  .- neffασ
    N_spin_orbital=length(neffασ)
    N_Γ=2^N_spin_orbital
    w02=[cal_w02_Γ(Γ,N_spin_orbital,neffασ,peffασ) for Γ in 1:N_Γ]
end

"""
x=rand(16-4-1)*0.1
nασ=[0.2,0.3,0.4,0.5]
G12ασ=[0.4,0.4,0.4,0.4]
w=cal_w(x,nασ,G12ασ,regulate_knorm_1)
pmatwασ,g12matwSασ=cal_p_g12_mat(G12ασ)
g11matwSασ=cal_g11_mat(G12ασ)
N_spin_orbital=4
g12ασ=[expt(w,cal_Xmatfull(pmatwασ,g12matwSασ,i)) for i in 1:N_spin_orbital]
g11ασ=[expt(w,cal_Xmatfull(pmatwασ,g11matwSασ,i)) for i in 1:N_spin_orbital]
# we encapsulate them into a function
g12ασ=cal_Xασ(pmatwασ,g12matwSασ)
g11ασ=cal_Xασ(pmatwασ,g11matwSασ)
"""
function cal_w(x,nασ,G12ασ,regulate_knorm)
    N_spin_orbital=length(nασ)
    neffασ=cal_neffασ(nασ,G12ασ)
    w02=cal_w02(neffασ)
    VΓη,ηToIdx=cal_VΓη_ηToIdx(N_spin_orbital)
    k=VΓη*x
    k0,knorm=cal_k0_knorm(k)
    kmax=cal_maxnorm(w02,k0)
    # we regulate the norm
    knormscaled=regulate_knorm(knorm,kmax)
    k_reg=k0*knormscaled
    w=sqrt.(k_reg+w02)
end

"""
we use the restricted version
x is just used to compute k0, so the value of |x| does not matter here
we could use sin, cos to do a partition
l_x∈[0,1]
l_x=0.5
l_x=1.0
x=rand(11)
"""
function cal_w_restrict(x,l_x,nασ,G12ασ)
    N_spin_orbital=length(nασ)
    neffασ=cal_neffασ(nασ,G12ασ)
    w02=cal_w02(neffασ)
    VΓη,ηToIdx=cal_VΓη_ηToIdx(N_spin_orbital)
    k0=VΓη*x
    kscale=cal_maxnorm(w02,k0)
    kscale_restrict=kscale*l_x
    # # we regulate the norm
    # knormscaled=regulate_knorm(knorm,kmax)
    k_restrict=k0*kscale_restrict
    w=sqrt.(k_restrict+w02)
end


"""
also, we can use some initial guess of self-energy to update the range
regulate_knorm=regulate_knorm_2
U=4.0
nασ=[0.5,0.5,0.5,0.5]
G12ασ,x,Δαασ,βασ=load_para_two_band_half(U)
G12ασ_test=[0.5 for i in 1:N_spin_orbital]
G12ασ_test=G12ασ
cal_G12ασ_check(x,nασ,regulate_knorm,[0.5 for i in 1:N_spin_orbital],[0.0 for i in 1:N_spin_orbital])
cal_G12ασ_check(x,nασ,regulate_knorm,[0.5 for i in 1:N_spin_orbital],[0.125 for i in 1:N_spin_orbital])
regulate_knorm=regulate_knorm_2
This use G12ασ_test and x to esimate the self-energy and use Δασ_test and fixed point method to estimate the G12ασ corresponding Δασ_test
regulate_knorm=regulate_knorm_3
G12ασ_min=cal_G12ασ_check(x,nασ,regulate_knorm_3,[0.5 for i in 1:N_spin_orbital],[0.0 for i in 1:N_spin_orbital])
G12ασ_test=[0.5 for i in 1:N_spin_orbital]
Δασ_test=[0.0 for i in 1:N_spin_orbital]
new version
l_x=1.0
"""
function cal_G12ασ_check(l_x,x,nασ,G12ασ_test,Δασ_test)
    N_spin_orbital=length(nασ)
    w_test=cal_w_restrict(x,l_x,nασ,G12ασ_test)
    pmatwασ,g12matwSασ=cal_p_g12_mat(G12ασ_test)
    g12ασ=cal_Xασ(w_test,pmatwασ,g12matwSασ)
    Slocασ=cal_Slocασ(nασ,G12ασ_test,g12ασ)
    # r,c are the reparameterization of the self-energy so G11 are 1/2
    rασ_test=cal_rασ(Slocασ)
    # we regulate it, so there is no singularity
    rασ_test=max.(rασ_test,1e-4)
    cασ_test=cal_cασ(nασ,Δασ_test,rασ_test)
    G12ασ_test=cal_G12ασ(nασ,Δασ_test,rασ_test,cασ_test)
end

