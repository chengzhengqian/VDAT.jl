# here, we orgaonize the code using the scheme 1, ie, G12,x,β as the parameters. The main challenging is that the physical range of G12 ->1/2 in the atomic limit. In this version, we use the G12=1/2 to estimate some self-energy, and use it to detemrine the G12_min, and then parametrize G12∈[G12_min,1/2]. This should improve the stability of the problem. (anothery way is when G12_min is close enough to 1/2, we just set G12 as 1/2 and α=0.0)

# the main goal of this version is try to clearly resolve the issure for the insulating phase.

# see vdat_v3.jl for details

using LinearAlgebra
using Statistics
using Combinatorics
using Roots
using Zygote
using Optim

include("./include_gene_code.jl")
# processing the band structure
include("./load_band.jl")
include("./utils.jl")

# local part,
# we also include the function to estimate G12ασ
include("./vdat_scheme1_local.jl")

# momnetum part
# we solve charge fluctuation and related derivatives
include("./vdat_scheme1_momentum.jl")



"""
compute total energy,
we first esimate the range of G12ασ
θ_G12ασ=[-0.3,-0.3,-0.3,-0.3]
l_G12ασ=sin.(θ_G12ασ).+0.5

regulate_knorm=regulate_knorm_2
(G12ασ_max-G12ασ)./(G12ασ_max-G12ασ_min)
x= [-26.752530823928637, -26.752530823928637, -26.752530823928637, -26.752530823928637, -26.752530823928637, -26.752530823928637, -0.002773243135255004, -0.002773243135255004, -0.002773243135255004, -0.002773243135255004, -1.2368972110757719]
G12ασ_min=cal_G12ασ_check(x,nασ,regulate_knorm_3,[0.5 for i in 1:N_spin_orbital],[0.0 for i in 1:N_spin_orbital])
# we use lbfgsb to minimize the problem
# so we need to paramtrize them differnetly
# first, l_G12ασ∈[0,1]
l_G12ασ=[0.5 for i in 1:N_spin_orbital]
l_x=0.5 controls and projection
x is only constrols the direction and should be parametrized by angle scheme
# it seems that once we restrict l_x, maybe the initial restriction is not necdesary
"""
function cal_energy_direct(l_G12ασ,l_x,x,βασ,nασ,eασ,interaction,G12ασ_min)
    N_spin_orbital=length(nασ)
    #  we use G12ασ=0.5 and Δασ=0.0 to estimate, Δασ=0.0 could be improved
    # G12ασ_min=cal_G12ασ_check(l_x,x,nασ,[0.5 for i in 1:N_spin_orbital],[0.0 for i in 1:N_spin_orbital])
    G12ασ_max=[0.5 for i in 1:N_spin_orbital]
    G12ασ=(G12ασ_min.*l_G12ασ)+G12ασ_max.*(1.0 .- l_G12ασ)
    # w=cal_w(x,nασ,G12ασ,regulate_knorm)
    w=cal_w_restrict(x,l_x,nασ,G12ασ)
    pmatwασ,g12matwSασ=cal_p_g12_mat(G12ασ)
    g12ασ=cal_Xασ(w,pmatwασ,g12matwSασ)
    Slocασ=cal_Slocασ(nασ,G12ασ,g12ασ)
    # Slocασ=[max.(Slocασ[i],1e-4) for i in 1:N_spin_orbital]
    # print("check G12 $(G12ασ)\n ")
    Δασ=cal_Δασ(g12ασ,Slocασ,nασ)
    Δασ=restrict_Δασ(Δασ,nασ)
    # we regulate Δασ
    # we should try to improve the estimation of G12ασ so we don't need restriction in this step
    # Δασ=min.(Δασ,nασ .- 1e-4)
    # Δασ=max.(Δασ, 1e-4)
    # we use 
    Aασ_below=[]
    Aασ_above=[]
    αασ=[]
    nk=[]
    K0=0
    # i=1
    for i in 1:N_spin_orbital
        # if(i==1)
        #     print("check n,Δ,β $([nασ[i],Δασ[i],βασ[i]])\n ")
        # end        
        Aασ_below_,Aασ_above_,Kbelow_,Kabove_,αασ_,nk_=cal_Abelow_Aabove_Kbelow_Kabove_αασ_nk_(nασ[i],Δασ[i],βασ[i],eασ[i])
        # K0ασ_=mean(nk_[1].*eασ[i][1])*nασ[i]+mean(nk_[2].*eασ[i][2])*(1-nασ[i])
        K0ασ_=Kbelow_+Kabove_
        K0+=K0ασ_
        push!(Aασ_below,Aασ_below_)
        push!(Aασ_above,Aασ_above_)
        push!(αασ,αασ_)
        push!(nk,nk_)
    end
    g33matwασ=[cal_g33_mat_(cal_Gfull(nασ[i],G12ασ[i],g12ασ[i],Aασ_below[i],Aασ_above[i],Slocασ[i])) for i in 1:N_spin_orbital]
    Eloc=sum([coefficient*expt(w,cal_Xmatfull(pmatwασ,g33matwασ,idx1,idx2)) for (idx1,idx2,coefficient) in interaction ])
    Eloc+K0,Eloc,K0
end


"""
we now use the momentum_derivaties from
momentum_derivatives=cal_momentum_derivatives(nασ,l_G12ασ,l_x,x,βασ,eασ,G12ασ_min)

"""
function cal_energy_with_momentum_derivatives(l_G12ασ,l_x,x,βασ,momentum_derivatives,interaction,G12ασ_min)
    (nασ,Δασ,nασ_below,nασ_above,αασ_below,αασ_above,βασ_below,βασ_above,nkασ_below,nkασ_above,Aασ_below,Aασ_above,Kασ_below,Kασ_above,∂Kασ∂nX_below,∂Kασ∂nX_above,∂Kασ∂βX_below,∂Kασ∂βX_above,∂Aασ∂nX_below,∂Aασ∂nX_above,∂Aασ∂βX_below,∂Aασ∂βX_above)=momentum_derivatives
    N_spin_orbital=length(nασ)
    G12ασ_max=[0.5 for i in 1:N_spin_orbital]
    G12ασ=(G12ασ_min.*l_G12ασ)+G12ασ_max.*(1.0 .- l_G12ασ)
    # return sum(G12ασ)  # break points to check where AD has probles
    # w=cal_w(x,nασ,G12ασ,regulate_knorm)
    w=cal_w_restrict(x,l_x,nασ,G12ασ)
    # return sum(w)
    pmatwασ,g12matwSασ=cal_p_g12_mat(G12ασ)
    g12ασ=cal_Xασ(w,pmatwασ,g12matwSασ)
    Slocασ=cal_Slocασ(nασ,G12ασ,g12ασ)
    # return Slocασ[1][1] 
    K0=sum(Kασ_below)+sum(Kασ_above)
    Δασ_track=cal_Δασ(g12ασ,Slocασ,nασ) # Δασ_track-Δασ
    # return sum(Δασ_track) # this has the problem
    Δασ_track=restrict_Δασ(Δασ_track,nασ)
    nασ_below_track=nασ-Δασ_track # nασ_below_track-nασ_below
    nασ_above_track=Δασ_track   #  nασ_above_track-nασ_above
    δnασ_below=nασ_below_track-nασ_below
    δnασ_above=nασ_above_track-nασ_above
    βασ_below_track=[βασ_[1] for βασ_ in βασ]
    βασ_above_track=[βασ_[2] for βασ_ in βασ]
    δβασ_below=βασ_below_track-βασ_below
    δβασ_above=βασ_above_track-βασ_above
    # we then compute the change of δβ
    # we use dot to sum, so we drop ασ
    δK_below=dot(∂Kασ∂nX_below,δnασ_below)+dot(∂Kασ∂βX_below,δβασ_below)
    δK_above=dot(∂Kασ∂nX_above,δnασ_above)+dot(∂Kασ∂βX_above,δβασ_above)
    δK=δK_below+δK_above
    K_track=K0+δK
    # now, we need to track A  part, Aασ_below
    Aασ_below_track=Aασ_below + ∂Aασ∂nX_below.*δnασ_below + ∂Aασ∂βX_below.*δβασ_below
    Aασ_above_track=Aασ_above + ∂Aασ∂nX_above.*δnασ_above + ∂Aασ∂βX_above.*δβασ_above
    g33matwασ=[cal_g33_mat_(cal_Gfull(nασ[i],G12ασ[i],g12ασ[i],Aασ_below_track[i],Aασ_above_track[i],Slocασ[i])) for i in 1:N_spin_orbital]
    # Eloc=sum([coefficient*expt(w,cal_Xmatfull(pmatwασ,g33matwασ,idx1,idx2)) for (idx1,idx2,coefficient) in interaction ])
    Eloc_M=sum([coefficient*cal_Xmatfull(pmatwασ,g33matwασ,idx1,idx2) for (idx1,idx2,coefficient) in interaction ])
    Eloc=expt(w,Eloc_M)
    E=Eloc+K_track
end


function cal_energy_direct_regulate(l_G12ασ,x,βασ,nασ,eασ,interaction,G12ασ_min)
    N_spin_orbital=length(nασ)
    #  we use G12ασ=0.5 and Δασ=0.0 to estimate, Δασ=0.0 could be improved
    # G12ασ_min=cal_G12ασ_check(l_x,x,nασ,[0.5 for i in 1:N_spin_orbital],[0.0 for i in 1:N_spin_orbital])
    G12ασ_max=[0.5 for i in 1:N_spin_orbital]
    G12ασ=(G12ασ_min.*l_G12ασ)+G12ασ_max.*(1.0 .- l_G12ασ)
    w=cal_w(x,nασ,G12ασ,regulate_knorm_linear)
    # w=cal_w_restrict(x,l_x,nασ,G12ασ)
    pmatwασ,g12matwSασ=cal_p_g12_mat(G12ασ)
    g12ασ=cal_Xασ(w,pmatwασ,g12matwSασ)
    Slocασ=cal_Slocασ(nασ,G12ασ,g12ασ)
    Slocασ=[max.(Slocασ[i],1e-4) for i in 1:N_spin_orbital]
    # print("check G12 $(G12ασ)\n ")
    Δασ=cal_Δασ(g12ασ,Slocασ,nασ)
    Δασ=restrict_Δασ(Δασ,nασ)
    # we regulate Δασ
    # we should try to improve the estimation of G12ασ so we don't need restriction in this step
    # Δασ=min.(Δασ,nασ .- 1e-4)
    # Δασ=max.(Δασ, 1e-4)
    # we use 
    Aασ_below=[]
    Aασ_above=[]
    αασ=[]
    nk=[]
    K0=0
    # i=1
    for i in 1:N_spin_orbital
        # if(i==1)
        #     print("check n,Δ,β $([nασ[i],Δασ[i],βασ[i]])\n ")
        # end        
        Aασ_below_,Aασ_above_,Kbelow_,Kabove_,αασ_,nk_=cal_Abelow_Aabove_Kbelow_Kabove_αασ_nk_(nασ[i],Δασ[i],βασ[i],eασ[i])
        # K0ασ_=mean(nk_[1].*eασ[i][1])*nασ[i]+mean(nk_[2].*eασ[i][2])*(1-nασ[i])
        K0ασ_=Kbelow_+Kabove_
        K0+=K0ασ_
        push!(Aασ_below,Aασ_below_)
        push!(Aασ_above,Aασ_above_)
        push!(αασ,αασ_)
        push!(nk,nk_)
    end
    g33matwασ=[cal_g33_mat_(cal_Gfull(nασ[i],G12ασ[i],g12ασ[i],Aασ_below[i],Aασ_above[i],Slocασ[i])) for i in 1:N_spin_orbital]
    Eloc=sum([coefficient*expt(w,cal_Xmatfull(pmatwασ,g33matwασ,idx1,idx2)) for (idx1,idx2,coefficient) in interaction ])
    Eloc+K0,Eloc,K0
end

#
"""
previously, we always want to fixed the density, but in pratice, we may also want to solve the problem with a given chemcial potential. Here, 
we first just use G12ασ, and use the local model to compute ,
so in the end, we tread w as varaitional parameters. and we compute nασ from local model
nασ,eασ
U_eff=1
w2=[p_eff(U_eff,sum(cal_Γασ(i,10))) for i in 1:2^10]
w2=w2/sum(w2)
w=sqrt.(w2)
G12ασ=[0.43 for i in 1:N_spin_orbital]
# we assume w is normalized, i.e, ∑ w^2 =1
we assume e_fn=gene_spline_band("./es_inf.dat") the band structure are same
right now, the way we generate the k point are accurate, but not efficient.
first, we can try use symmetry, but right now, we just not worry about it.
We also return other observables so we can store the results
@time Etotal,Eloc,K0,nασ,nn,αασ,βασ,eασ,Slocασ,Δασ,Aασ_below,Aασ_above=cal_energy_projective(G12ασ,w,βασ,e_fn,interaction)
"""
function cal_energy_projective(G12ασ,w,βασ,e_fn,interaction)
    N_spin_orbital=length(G12ασ)
    # G12ασ=clamp.(G12ασ,0.2,0.51)
    pmatwασ,g12matwSασ=cal_p_g12_mat(G12ασ)
    g11matwSασ=cal_g11_mat(G12ασ)
    g12ασ=cal_Xασ(w,pmatwασ,g12matwSασ)
    nασ=cal_Xασ(w,pmatwασ,g11matwSασ)
    nασ=restrict_nασ(nασ)
    # this approach is accurate, but not that efficient
    eασ=[ gene_ϵs(e_fn,nασ[i]) for _ in 1:N_spin_orbital]
    Slocασ=cal_Slocασ(nασ,G12ασ,g12ασ)
    Slocασ=[max.(Slocασ[i],1e-5) for i in 1:N_spin_orbital]
    # print("check G12 $(G12ασ)\n ")
    Δασ=cal_Δασ(g12ασ,Slocασ,nασ)
    Δασ=restrict_Δασ(Δασ,nασ; cutoff=1e-5)
    Aασ_below=[]
    Aασ_above=[]
    αασ=[]
    nk=[]
    K0=0
    # i=1
    for i in 1:N_spin_orbital
        Aασ_below_,Aασ_above_,Kbelow_,Kabove_,αασ_,nk_=cal_Abelow_Aabove_Kbelow_Kabove_αασ_nk_(nασ[i],Δασ[i],βασ[i],eασ[i])
        # K0ασ_=mean(nk_[1].*eασ[i][1])*nασ[i]+mean(nk_[2].*eασ[i][2])*(1-nασ[i])
        K0ασ_=Kbelow_+Kabove_
        K0+=K0ασ_
        push!(Aασ_below,Aασ_below_)
        push!(Aασ_above,Aασ_above_)
        push!(αασ,αασ_)
        push!(nk,nk_)
    end
    g33matwασ=[cal_g33_mat_(cal_Gfull(nασ[i],G12ασ[i],g12ασ[i],Aασ_below[i],Aασ_above[i],Slocασ[i])) for i in 1:N_spin_orbital]
    nn=[expt(w,cal_Xmatfull(pmatwασ,g33matwασ,idx1,idx2)) for (idx1,idx2,coefficient) in interaction]
    Eloc=sum([interaction[i][3]*nn[i]   for i in 1:length(interaction)])
    # Eloc=sum([coefficient*expt(w,cal_Xmatfull(pmatwασ,g33matwασ,idx1,idx2)) for (idx1,idx2,coefficient) in interaction ])
    Eloc+K0,Eloc,K0,nασ,nn,αασ,βασ,eασ,Slocασ,Δασ,Aασ_below,Aασ_above,G12ασ,w,nk
end

