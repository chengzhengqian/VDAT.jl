# here, we summarize the v2, and try some new way to regulate the x (i.e, we set some cutoff so the diagonal part of self-energy is always non-zero)
# we also clear the documents, see v2 for more detailed documents

using LinearAlgebra
using Statistics
using Combinatorics
using Roots
using Zygote

include("./include_gene_code.jl")
# processing the band structure
include("./load_band.jl")
include("./utils.jl")
"""
In practice, we only need this too, actualy, pmat is know too
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
We explicitly list the argument for AD of Zygote.jl
"""
function cal_g33_mat_(Gfull)
    cal_g33matw(Gfull[1],Gfull[2],Gfull[3],Gfull[4],Gfull[5],Gfull[6])
end

"""
Gfull=[0.5,0.4,0.5,-0.3,-0.4,0.5]
Gfullασ=[Gfull for _ in 1:4]
we use ασ to indicates it is spin orbital dependent
cal_g33_mat(Gfullασ)
"""
function cal_g33_mat(Gfullασ)
    cal_g33_mat_.(Gfullασ)
end

"""
extend Xmatασ[idx] to the full space, by assume it is in idx spin orbital
"""
function cal_Xmatfull(pmatwασ,Xmatασ,idx)
    kron(pmatwασ[1:(idx-1)]...,Xmatασ[idx],pmatwασ[(idx+1):end]...)
end

"""
extend Xmatασ[idx1]*Xmatασ[idx2] to the full space, assuming idx1<idx2
"""
function cal_Xmatfull(pmatwασ,Xmatασ,idx1,idx2)
    kron(pmatwασ[1:(idx1-1)]...,Xmatασ[idx1],pmatwασ[(idx1+1):(idx2-1)]...,Xmatασ[idx2],pmatwασ[(idx2+1):end]...)
end


"""
effetive density, use to construct w02
nασ=[0.4,0.5,0.6,0.7]
G12ασ=[0.3,0.4,0.5,0.5]
neffασ=cal_neffασ(nασ,G12ασ)
gradient((x,y)->sum(cal_neffασ(x,y)),nασ,G12ασ)
"""
function cal_neffασ(nασ,G12ασ)
    # N_spin_orbital=length(nασ)
    # [cal_neffInn(nασ[i],G12ασ[i])  for i in 1:N_spin_orbital]
    cal_neffInn.(nασ,G12ασ)
end


function cal_w02_Γ(Γ,N_spin_orbital,neffασ,peffασ)
    Γασ=cal_Γασ(Γ,N_spin_orbital)
    prod(Γασ.*neffασ+(1.0 .- Γασ).*peffασ)
end


"""
cal_w02([0.5,0.5])
cal_w02([0.5,0.5,0.5,0.5])
"""
function cal_w02(neffασ)
    peffασ=1.0  .- neffασ
    N_spin_orbital=length(neffασ)
    N_Γ=2^N_spin_orbital
    w02=[cal_w02_Γ(Γ,N_spin_orbital,neffασ,peffασ) for Γ in 1:N_Γ]
end

function cal_k0_knorm(k)
    knorm=norm1(k)
    k0=k/knorm
    k0,knorm
end

function norm1(k)
    sum(abs.(k))
end

Base.convert(::Type{Int64},x::Nothing)=0

function cal_maxnorm(w02,k0)
    kmax=minimum([-w02[i]/k0[i] for i in 1:length(w02)  if k0[i]<0])
end

"""
Here, we make the procedure general, we pass the function of regulate as parameters
# pass the regulated function as 
knormscaled=regulate_knorm(knorm,kmax)

x=rand(16-4-1)*0.1
nασ=[0.5,0.5,0.5,0.5]
G12ασ=[0.4,0.4,0.4,0.4]
regulate_knorm=(x,y)->x
regulate_knorm(2,1)
cal_w(x*1.61,nασ,G12ασ,regulate_knorm_1)
gradient(x->cal_w(x,nασ,G12ασ,regulate_knorm)[1],x)

# some notes
w02 is the vector of w0^2, 2^N_spin_orbital
η 1,..,(2^N_spin_orbital-N_spin_orbital-1), the irreducible varaitional parameters.
idx, [i,j,..,k], correponds a η, with the density fluctuation between n sites.
VΓη, maps the x (the vector of irreducible varaitional parameter) to the w representation.
kmax is determined for a given k0 and w02, so knorm ∈[0,kmax]  
regulate_knorm is a customize function to regulate knorm
"""
function cal_w(x,nασ,G12ασ,regulate_knorm)
    neffασ=cal_neffασ(nασ,G12ασ)
    w02=cal_w02(neffασ)
    N_spin_orbital=length(neffασ)
    VΓη,ηToIdx=cal_VΓη_ηToIdx(N_spin_orbital)
    k=VΓη*x
    k0,knorm=cal_k0_knorm(k)
    kmax=cal_maxnorm(w02,k0)
    # we regulate the norm
    knormscaled=regulate_knorm(knorm,kmax)
    kscaled=k0*knormscaled
    w=sqrt.(kscaled+w02)
end

# we set some cutoff
# we can tick this choice later
function regulate_knorm_1(knorm,kmax)
    kmax*(1-1e-4)*abs(sin(knorm/kmax))
end


"""
compute expetation values, w^2 is assumed to be normalizd,
i.e, compute from cal_w
"""
function expt(w,Xmatfull)
    dot(w,Xmatfull*w)
end

"""
compute self-energy, we may want to also regulate the self-energy, but as we have already reguate the 
for each entry, we have [(Sloc11,Sloc12),...]
cal_Slocασ(nασ,G12ασ,G12ασ)
"""
function cal_Slocασ(nασ,G12ασ,g12ασ)
    cal_sloc11sloc12.(nασ,G12ασ,g12ασ)
end

"""
X represent either below or above,
"""
function cal_nX(esX,αX,βX,weightX)
    nkX=[ cal_nk(αX,βX,esX_) for esX_ in esX]
    nX=mean(nkX)*weightX
end

"""
we should restrict nX from Δ, this should be important in the atomic limit
"""
function solve_αX_from_nX(esX,nX,βX,weightX)
    # this should be within 0.0 and 1.0
    # nX=constraint_nX(nX,weightX)
    n_mean=nX/weightX
    e_mean=mean(esX)
    αX_guess=cal_alpha(e_mean,βX,n_mean)
    αX=find_zero(αX->cal_nX(esX,αX,βX,weightX)-nX,αX_guess,Order0())
end

"""
for a given region
"""
function cal_nX_AX_KX_nkX(esX,αX,βX,weightX)
    nkX=[ cal_nk(αX,βX,esX_) for esX_ in esX]
    nX=mean(nkX)*weightX
    AX=mean(sqrt.(nkX.*(1.0 .- nkX)))*weightX
    KX=mean(esX.*nkX)*weightX
    nX,AX,KX,nkX
end

function cal_nX_AX_nkX(esX,αX,βX,weightX)
    nkX=[ cal_nk(αX,βX,esX_) for esX_ in esX]
    nX=mean(nkX)*weightX
    AX=mean(sqrt.(nkX.*(1.0 .- nkX)))*weightX
    nX,AX,nkX
end


function cal_Aασ_αασ_nk_(nασ_,Δασ_,βασ_,eασ_)
    βbelow,βabove=βασ_
    ebelow,eabove=eασ_
    nbelow=nασ_-Δασ_
    nabove=Δασ_
    αbelow=solve_αX_from_nX(ebelow,nbelow,βbelow,nασ_)
    αabove=solve_αX_from_nX(eabove,nabove,βabove,1-nασ_)
    nbelowcheck,Abelow,nkbelow=cal_nX_AX_nkX(ebelow,αbelow,βbelow,nασ_)
    nabovecheck,Aabove,nkabove=cal_nX_AX_nkX(eabove,αabove,βabove,1-nασ_)
    [Abelow,Aabove],[αbelow,αabove],[nkbelow,nkabove]
end



"""
the derivatives in terms of α, β
"""
function cal_dKX_dnX_dAX_dαβ(esX,αX,βX,weightX)
    # the output order is {dnkda,dnkdb,dAkda,dAkdb}
    # cal_dnAdab(α,β,ϵ)
    dnAXdαβ=[cal_dnAdab(αX,βX,ϵ) for ϵ in esX]
    # we first compute K
    N_k_points=length(dnAXdαβ)
    dKXdα=mean([esX[i]*dnAXdαβ[i][1] for i in 1:N_k_points])*weightX
    dKXdβ=mean([esX[i]*dnAXdαβ[i][2] for i in 1:N_k_points])*weightX
    dnXdα=mean([dnAXdαβ[i][1] for i in 1:N_k_points])*weightX
    dnXdβ=mean([dnAXdαβ[i][2] for i in 1:N_k_points])*weightX
    dAXdα=mean([dnAXdαβ[i][3] for i in 1:N_k_points])*weightX
    dAXdβ=mean([dnAXdαβ[i][4] for i in 1:N_k_points])*weightX
    dKXdα,dKXdβ,dnXdα,dnXdβ,dAXdα,dAXdβ
end

"""
the derivatives in terms of n,β
"""
function cal_dKX_dAX_dnβ(esX,αX,βX,weightX)
    # we first compute the derivatives in terms of αX,βX
    dKXdα,dKXdβ,dnXdα,dnXdβ,dAXdα,dAXdβ=cal_dKX_dnX_dAX_dαβ(esX,αX,βX,weightX)
    # we use underscore to indicate it is in n,β
    dKXdn_=dKXdα/dnXdα
    dKXdβ_=dKXdβ-dKXdα*dnXdβ/dnXdα
    dAXdn_=dAXdα/dnXdα
    dAXdβ_=dAXdβ-dAXdα*dnXdβ/dnXdα
    dKXdn_,dKXdβ_,dAXdn_,dAXdβ_
end

"""
we need to regulate Δασ∈[0,nασ]
regulate_Δασ.(Δασ,nασ)
# no regulation
regulate_Δασ(x,y)=x
"""
function cal_Δασ(g12ασ,Slocασ,nασ,regulate_Δασ)
    N_spin_orbital=length(g12ασ)
    # [ cal_delta(g12ασ[idx],Slocασ[idx]...) for idx in 1:N_spin_orbital]
    Δασ=[ cal_delta(g12ασ[idx],Slocασ[idx][1],Slocασ[idx][2]) for idx in 1:N_spin_orbital]
    regulate_Δασ.(Δασ,nασ)
end

function cal_Δασ_with_penalty(g12ασ,Slocασ,nασ,regulate_Δασ)
    N_spin_orbital=length(g12ασ)
    # [ cal_delta(g12ασ[idx],Slocασ[idx]...) for idx in 1:N_spin_orbital]
    Δασ=[ cal_delta(g12ασ[idx],Slocασ[idx][1],Slocασ[idx][2]) for idx in 1:N_spin_orbital]
    Δασ_new=regulate_Δασ.(Δασ,nασ)
    Δασ_new,penalty_Δασ(Δασ_new,Δασ)
end

"""
some version
Δασ_=0.5-1e-4
nασ_=0.5
regulate_Δασ_1(100,nασ_)
"""
function regulate_Δασ_1(Δασ_,nασ_)
    if(Δασ_<1e-4)
        0.9*(1e-4)*exp((Δασ_-(1e-4)))+0.1*(1e-4)
    elseif(Δασ_>nασ_-1e-4)
        nασ_-0.9*(1e-4)*exp((nασ_-Δασ_-(1e-4)))-0.1*(1e-4)
    else
        Δασ_
    end
end

function penalty_Δασ(Δασ,Δασ_new)
    100*sum((Δασ.-Δασ_new).^2)
end


"""
we update the constraint of Δασ, here we may want to add some penalty function, we first try without it.
N_spin_orbital=4
nασ=[0.5,0.5,0.5,0.5]
G12ασ=[0.4,0.4,0.4,0.4]
e_fn=gene_spline_band("./es_inf.dat")
es=gene_ϵs(e_fn,nασ[1])
eασ=[ es for _ in 1:N_spin_orbital]
βασ=[[1.0,1.0] for _ in 1:N_spin_orbital]
x=rand(2^N_spin_orbital-N_spin_orbital-1)
regulate_knorm=regulate_knorm_1
regulate_Δασ(x,y)=regulate_Δασ_1(x,y)
# all derivatives are in terms of n,β 
        (nασ,Δασ,nασ_below,nασ_above,αασ_below,αασ_above,βασ_below,βασ_above,nkασ_below,nkασ_above,Aασ_below,Aασ_above,Kασ_below,Kασ_above,∂Kασ∂nX_below,∂Kασ∂nX_above,∂Kασ∂βX_below,∂Kασ∂βX_above,∂Aασ∂nX_below,∂Aασ∂nX_above,∂Aασ∂βX_below,∂Aασ∂βX_above)=cal_momentum_derivatives(nασ,G12ασ,x,βασ,eασ,regulate_knorm_1,regulate_Δασ_1)
"""
function cal_momentum_derivatives(nασ,G12ασ,x,βασ,eασ,regulate_knorm,regulate_G12,regulate_Δασ)
    N_spin_orbital=length(nασ)
    w=cal_w(x,nασ,G12ασ,regulate_knorm)
    G12ασ=regulate_G12.(G12ασ)
    pmatwασ,g12matwSασ=cal_p_g12_mat(G12ασ)
    g12ασ=[expt(w,cal_Xmatfull(pmatwασ,g12matwSασ,i)) for i in 1:N_spin_orbital]
    Slocασ=cal_Slocασ(nασ,G12ασ,g12ασ)
    Δασ=cal_Δασ(g12ασ,Slocασ,nασ,regulate_Δασ)
    # we will compute the seperatly the below and above seperately
    # we add them, so don't need another function
    # we also add βασ_below and above so it is easier when write the energy differentiaion
    nkασ_below=[]
    nkασ_above=[]
    nασ_below=nασ-Δασ
    nασ_above=Δασ
    αασ_below=[]
    αασ_above=[]
    βασ_below=[]
    βασ_above=[]
    Aασ_below=[]
    Aασ_above=[]
    Kασ_below=[]
    Kασ_above=[]
    ∂Kασ∂nX_below=[]
    ∂Kασ∂nX_above=[]
    ∂Kασ∂βX_below=[]
    ∂Kασ∂βX_above=[]
    ∂Aασ∂nX_below=[]
    ∂Aασ∂nX_above=[]
    ∂Aασ∂βX_below=[]
    ∂Aασ∂βX_above=[]
    # i=1
    for i in 1:N_spin_orbital
        βασ_below_=βασ[i][1]
        αασ_below_=solve_αX_from_nX(eασ[i][1],nασ_below[i],βασ_below_,nασ[i])
        para_below=(eασ[i][1],αασ_below_,βασ[i][1],nασ[i])
        dKbelowdn_,dKbelowdβ_,dAbelowdn_,dAbelowdβ_=cal_dKX_dAX_dnβ(para_below...)
        nασ_below_,Aασ_below_,Kασ_below_,nkασ_below_=cal_nX_AX_KX_nkX(para_below...)
        βασ_above_=βασ[i][2]
        αασ_above_=solve_αX_from_nX(eασ[i][2],nασ_above[i],βασ_above_,1.0-nασ[i])
        para_above=(eασ[i][2],αασ_above_,βασ[i][2],1.0-nασ[i])
        dKabovedn_,dKabovedβ_,dAabovedn_,dAabovedβ_=cal_dKX_dAX_dnβ(para_above...)
        nασ_above_,Aασ_above_,Kασ_above_,nkασ_above_=cal_nX_AX_KX_nkX(para_above...)
        push!(nkασ_below,nkασ_below_)
        push!(nkασ_above,nkασ_above_)
        push!(αασ_below,αασ_below_)
        push!(αασ_above,αασ_above_)
        push!(βασ_below,βασ_below_)
        push!(βασ_above,βασ_above_)
        push!(Aασ_below,Aασ_below_)
        push!(Aασ_above,Aασ_above_)
        push!(Kασ_below,Kασ_below_)
        push!(Kασ_above,Kασ_above_)
        push!(∂Kασ∂nX_below,dKbelowdn_)
        push!(∂Kασ∂nX_above,dKabovedn_)
        push!(∂Kασ∂βX_below,dKbelowdβ_)
        push!(∂Kασ∂βX_above,dKabovedβ_)
        push!(∂Aασ∂nX_below,dAbelowdn_)
        push!(∂Aασ∂nX_above,dAabovedn_)
        push!(∂Aασ∂βX_below,dAbelowdβ_)
        push!(∂Aασ∂βX_above,dAabovedβ_)
    end
    (nασ,Δασ,nασ_below,nασ_above,αασ_below,αασ_above,βασ_below,βασ_above,nkασ_below,nkασ_above,Aασ_below,Aασ_above,Kασ_below,Kασ_above,∂Kασ∂nX_below,∂Kασ∂nX_above,∂Kασ∂βX_below,∂Kασ∂βX_above,∂Aασ∂nX_below,∂Aασ∂nX_above,∂Aασ∂βX_below,∂Aασ∂βX_above)
end


function cal_Gfull(nασ_,G12ασ_,g12ασ_,Aασ_below_,Aασ_above_,Slocασ_)
    gextraασ_=cal_glocReducedInA(Aασ_below_,Aασ_above_,Slocασ_[1],Slocασ_[2])
    Gfullασ_=cal_g0fullIngG12(nασ_,G12ασ_,g12ασ_,gextraασ_[1],gextraασ_[2],gextraασ_[3],gextraασ_[4])
end

"""
for direct version, where A_below and A_above  are combined into one entry
"""
function cal_Gfull(nασ_,G12ασ_,g12ασ_,Aασ_,Slocασ_)
    gextraασ_=cal_glocReducedInA(Aασ_[1],Aασ_[2],Slocασ_[1],Slocασ_[2])
    Gfullασ_=cal_g0fullIngG12(nασ_,G12ασ_,g12ασ_,gextraασ_[1],gextraασ_[2],gextraασ_[3],gextraασ_[4])
end

"""
some possible way to regulate
if we don't restrict G12, but add penalty?
regulate_G12_1(0.42)
regulate_G12_1(10)
"""
function regulate_G12_1(x)
    # 0.5*abs(sin(x/0.5))
    Gmax=0.5
    Gmin=0.3
    ΔG=0.02
    if(x>Gmax)
        Gmax+ΔG*(1-exp(Gmax-x))
    elseif(x<Gmin)
        Gmin-ΔG*(1-exp(x-Gmin))
    else
        x
    end    
end
# """
# we add panelty
# """
# function regulate_G12_1(x)
#     x
# end


"""
momentum_info=cal_momentum_derivatives(nασ,G12ασ,x,βασ,eασ,regulate_knorm_1,regulate_G12_1,regulate_Δασ_1)
U=2.0
regulate_G12_1.(G12ασ)
interaction=[(1,2,U),(1,3,U),(1,4,U),(2,3,U),(2,4,U),(3,4,U)]
E=cal_energy(G12ασ,x,βασ,momentum_info,interaction,regulate_knorm_1,regulate_G12_1,regulate_Δασ_1)
E_check,_,_=cal_energy_direct(G12ασ,x,βασ,nασ,eασ,interaction,regulate_knorm_1,regulate_G12_1,regulate_Δασ_1)
E-E_check
regulate_G12=x->x
"""
function cal_energy(G12ασ,x,βασ,momentum_info,interaction,regulate_knorm,regulate_G12,regulate_Δασ)
    (nασ,Δασ,nασ_below,nασ_above,αασ_below,αασ_above,βασ_below,βασ_above,nkασ_below,nkασ_above,Aασ_below,Aασ_above,Kασ_below,Kασ_above,∂Kασ∂nX_below,∂Kασ∂nX_above,∂Kασ∂βX_below,∂Kασ∂βX_above,∂Aασ∂nX_below,∂Aασ∂nX_above,∂Aασ∂βX_below,∂Aασ∂βX_above)=momentum_info
    # this is the fixed value, as denoted as star in notes, here, we just keep the name, but add _track (if we will use the same variable) to the value (zeroth order are same) computed in the procedure, so the automatically differentialion then can properly backpropagate teh derivatives from the momentum info
    N_spin_orbital=length(nασ)
    # we add restriction of G12ασ
    # here, we may want panelty when the self-consistency can not be satified
    # or we just regulate G12ασ, we first set a play holder for it
    G12ασ=regulate_G12.(G12ασ)
    w=cal_w(x,nασ,G12ασ,regulate_knorm)
    pmatwασ,g12matwSασ=cal_p_g12_mat(G12ασ)
    g12ασ=[expt(w,cal_Xmatfull(pmatwασ,g12matwSασ,i)) for i in 1:N_spin_orbital]
    Slocασ=cal_Slocασ(nασ,G12ασ,g12ασ)
    K0=sum(Kασ_below)+sum(Kασ_above)
    # Δασ_track=cal_Δασ(g12ασ,Slocασ,nασ,regulate_Δασ) # Δασ_track-Δασ
    Δασ_track,penalty=cal_Δασ_with_penalty(g12ασ,Slocασ,nασ,regulate_Δασ) # Δασ_track-Δασ
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
    Eloc=sum([coefficient*expt(w,cal_Xmatfull(pmatwασ,g33matwασ,idx1,idx2)) for (idx1,idx2,coefficient) in interaction ])
    # E=Eloc+K_track
    E=Eloc+K_track+penalty
end

"""
directly compute total energy
"""
function cal_energy_direct(G12ασ,x,βασ,nασ,eασ,interaction,regulate_knorm,regulate_G12,regulate_Δασ)
    # we first take the cal_momemtum_part(nασ,G12ασ,x,βασ,eασ), but with the scaled cal_w_scaled
    N_spin_orbital=length(nασ)
    # we add restriction of G12ασ
    G12ασ=regulate_G12.(G12ασ)
    # w=cal_w_scaled_v2(x,nασ,G12ασ)
    w=cal_w(x,nασ,G12ασ,regulate_knorm)
    pmatwασ,g12matwSασ=cal_p_g12_mat(G12ασ)
    g12ασ=[expt(w,cal_Xmatfull(pmatwασ,g12matwSασ,i)) for i in 1:N_spin_orbital]
    Slocασ=cal_Slocασ(nασ,G12ασ,g12ασ)
    Δασ,penalty=cal_Δασ_with_penalty(g12ασ,Slocασ,nασ,regulate_Δασ)
    # Δασ=cal_Δασ(g12ασ,Slocασ,nασ,regulate_Δασ)
    Aασ=[]
    αασ=[]
    nk=[]
    # we don't need derivatives
    K0=0                        # kinecit energy
    for i in 1:N_spin_orbital
        Aασ_,αασ_,nk_=cal_Aασ_αασ_nk_(nασ[i],Δασ[i],βασ[i],eασ[i])
        K0ασ_=mean(nk_[1].*eασ[i][1])*nασ[i]+mean(nk_[2].*eασ[i][2])*(1-nασ[i])
        K0+=K0ασ_
        push!(Aασ,Aασ_)
        push!(αασ,αασ_)
        push!(nk,nk_)
    end
    # now, we move to cal_energy_for_diff_old
    g33matwασ=[cal_g33_mat_(cal_Gfull(nασ[i],G12ασ[i],g12ασ[i],Aασ[i],Slocασ[i])) for i in 1:N_spin_orbital]
    Eloc=sum([coefficient*expt(w,cal_Xmatfull(pmatwασ,g33matwασ,idx1,idx2)) for (idx1,idx2,coefficient) in interaction ])
    Eloc+K0+penalty,Eloc,K0,penalty
    # Eloc+K0,Eloc,K0
end

"""
compute derivatives
∂E∂G12,∂E∂x,∂E∂βασ=cal_gradient(G12ασ,x,βασ,nασ,eασ,interaction,regulate_knorm,regulate_G12,regulate_Δασ)
"""
function cal_gradient(G12ασ,x,βασ,nασ,eασ,interaction,regulate_knorm,regulate_G12,regulate_Δασ)
    momentum_info=cal_momentum_derivatives(nασ,G12ασ,x,βασ,eασ,regulate_knorm,regulate_G12,regulate_Δασ)
    ∂E∂G12,∂E∂x,∂E∂βασ=gradient((G12ασ,x,βασ)->cal_energy(G12ασ,x,βασ,momentum_info,interaction,regulate_knorm,regulate_G12,regulate_Δασ),G12ασ,x,βασ)
end

