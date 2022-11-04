# we put the local part information here

# see vdat_v2,vdat_v3 for more details about these functions

"""
G12ασ=[0.4 for _ in 1:N_spin_orbital]
pmatwασ,g12matwSασ=cal_p_g12_mat(G12ασ)
"""
function cal_p_g12_mat(G12ασ::Array)
    pmatwασ=cal_pmatw.(G12ασ)
    g12matwSασ=cal_g12matwS.(G12ασ)
    pmatwασ,g12matwSασ
end

"""
for single orbital
Gfull=[g012,g013,g023,g031,g032,g033]
"""
function cal_g33_mat_(Gfull)
    cal_g33matw(Gfull[1],Gfull[2],Gfull[3],Gfull[4],Gfull[5],Gfull[6])
end

function cal_g33_mat(Gfullασ)
    cal_g33_mat_.(Gfullασ)
end

"""
effetive density, use to construct w02
"""
function cal_neffασ(nασ,G12ασ)
    cal_neffInn.(nασ,G12ασ)
end

"""
for a given configuration index Γ
"""
function cal_w02_Γ(Γ,N_spin_orbital,neffασ,peffασ)
    Γασ=cal_Γασ(Γ,N_spin_orbital)
    prod(Γασ.*neffασ+(1.0 .- Γασ).*peffασ)
end

function cal_w02(neffασ)
    peffασ=1.0  .- neffασ
    N_spin_orbital=length(neffασ)
    N_Γ=2^N_spin_orbital
    w02=[cal_w02_Γ(Γ,N_spin_orbital,neffασ,peffασ) for Γ in 1:N_Γ]
end

# the functions to constraint k are moved to utils
"""
nασ=[0.5,0.5,0.5,0.5]
G12ασ=[0.4,0.4,0.4,0.4]
x=rand(16-4-1)*0.1
w=cal_w(x,nασ,G12ασ,regulate_knorm_1)

"""
function cal_w(x,nασ,G12ασ,regulate_knorm)
    N_spin_orbital=length(nασ)
    neffασ=cal_neffασ(nασ,G12ασ)
    w02=cal_w02(neffασ)
    VΓη,ηToIdx=cal_VΓη_ηToIdx(N_spin_orbital)
    k=VΓη*x
    k0,knorm=cal_k0_knorm(k)
    kmax=cal_maxnorm(w02,k0)
    knormscaled=regulate_knorm(knorm,kmax)
    kscaled=k0*knormscaled
    w=sqrt.(kscaled+w02)
end


# we now implement the self-consistency
"""
for a given Δασ, and x, (of course,nασ, and the given reglate_knorm),
we found the 
"""
