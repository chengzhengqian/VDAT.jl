# this is the version for the multiband case with diagonal G and N=3

# we will use Mathematica to generate some elementary building blocks
# we now test these functions
# using Pkg
# Pkg.add("Combinatorics")
using LinearAlgebra
using Statistics
using Combinatorics
using Roots
include("./include_gene_code.jl")

# this first step is to construct the local A blocks, as we have
# we use a two orbital example to illustract the process
# # 1up 1dn 2up 2dn
# nασ=[0.3,0.4,0.5,0.6]
# G12ασ=[0.4,0.3,0.4,0.3]

"""
only for a single orbital
cal_GA_block(0.3)
"""
function cal_GA_block(g012::Number)
    pmatw=cal_pmatw(g012)
    g11matwS=cal_g11matwS(g012)
    g12matwS=cal_g12matwS(g012)
    pmatw,g11matwS,g12matwS
end
"""
g0full=Gloc[1]
"""
function cal_G_block_(g0full)
    # g012,g013,g023,g031,g032,g033
    g33matw=cal_g33matw(g0full...)
end

function cal_G_block_all_(g0full)
    # g012,g013,g023,g031,g032,g033
    g33matw=cal_g33matw(g0full...)
    g13matw=cal_g13matw(g0full...)
    g23matw=cal_g23matw(g0full...)
    g31matw=cal_g31matw(g0full...)
    g32matw=cal_g32matw(g0full...)
    g33matw,g13matw,g23matw,g31matw,g32matw
end

"""
gAblockmat=cal_GA_block(G12ασ)
"""
function cal_GA_block(G12ασ::Array)
    pmatwασ=[]
    g11matwSασ=[]
    g12matwSασ=[]
    for g012 in G12ασ
        pmatw,g11matwS,g12matwS=cal_GA_block(g012)
        push!(pmatwασ,pmatw)
        push!(g11matwSασ,g11matwS)
        push!(g12matwSασ,g12matwS)
    end
    pmatwασ,g11matwSασ,g12matwSασ
end

"""
g33matwασ=cal_G_block(Gloc)
"""
function cal_G_block(Gloc)
    # g012,g013,g023,g031,g032,g033
    [cal_G_block_(g0full_) for g0full_ in Gloc]
end

"""
g33matwασ,g13matwασ,g23matwασ,g31matwασ,g32matwασ=cal_G_block_all(Gloc)
"""
function cal_G_block_all(Gloc)
    # g012,g013,g023,g031,g032,g033
    g33matwασ=[]
    g13matwασ=[]
    g23matwασ=[]
    g31matwασ=[]
    g32matwασ=[]
    for g0full in Gloc
        g33matw,g13matw,g23matw,g31matw,g32matw=cal_G_block_all_(g0full)
        push!(g33matwασ,g33matw)
        push!(g13matwασ,g13matw)
        push!(g23matwασ,g23matw)
        push!(g31matwασ,g31matw)
        push!(g32matwασ,g32matw)
    end
    g33matwασ,g13matwασ,g23matwασ,g31matwασ,g32matwασ
end

"""
compute the full form
g11matwSfull=cal_g11matwSfull(gAblockmat)
"""
function cal_g11matwSfull(gAblockmat)
    pmatwασ,g11matwSασ,g12matwSασ=gAblockmat
    g11matwSfull=[]
    # loop over all spin orbitals
    for idx in 1:length(pmatwασ)
        args=copy(pmatwασ)
        args[idx]=g11matwSασ[idx]
        push!(g11matwSfull,kron(args...))
    end
    g11matwSfull
end

"""
compute the full form
g12matwSfull=cal_g12matwSfull(gAblockmat)
"""
function cal_g12matwSfull(gAblockmat)
    pmatwασ,g11matwSασ,g12matwSασ=gAblockmat
    g12matwSfull=[]
    # loop over all spin orbitals
    for idx in 1:length(pmatwασ)
        args=copy(pmatwασ)
        args[idx]=g12matwSασ[idx]
        push!(g12matwSfull,kron(args...))
    end
    g12matwSfull
end

"""
just check, we need to get interaction
g33matwfull=cal_g33matwfull(gAblockmat,g33matwασ)
"""
function cal_g33matwfull(gAblockmat,g33matwασ)
    pmatwασ,g11matwSασ,g12matwSασ=gAblockmat
    g33matwfull=[]
    # loop over all spin orbitals
    for idx in 1:length(pmatwασ)
        args=copy(pmatwασ)
        args[idx]=g33matwασ[idx]
        push!(g33matwfull,kron(args...))
    end
    g33matwfull
end

"""
g33matwασ,g13matwασ,g23matwασ,g31matwασ,g32matwασ=cal_G_block_all(Gloc)
g13matwfull=cal_Xmatwfull(gAblockmat,g13matwασ)
g23matwfull=cal_Xmatwfull(gAblockmat,g23matwασ)
g31matwfull=cal_Xmatwfull(gAblockmat,g31matwασ)
g32matwfull=cal_Xmatwfull(gAblockmat,g32matwασ)
"""
function cal_Xmatwfull(gAblockmat,Xmatwασ)
    pmatwασ,g11matwSασ,g12matwSασ=gAblockmat
    Xmatwfull=[]
    # loop over all spin orbitals
    for idx in 1:length(pmatwασ)
        args=copy(pmatwασ)
        args[idx]=Xmatwασ[idx]
        push!(Xmatwfull,kron(args...))
    end
    Xmatwfull
end

"""
idx=[1,2]
"""
function cal_nnmatwfull(gAblockmat,g33matwασ,idx)
    pmatwασ,g11matwSασ,g12matwSασ=gAblockmat
    args=copy(pmatwασ)
    for idx_ in idx
        args[idx]=g33matwασ[idx]
    end
    kron(args...)
end

"""
we first check the non-interacting w^2
We need to get neffασ first
neffασ=cal_neffασ(nασ,G12ασ)
"""
function cal_neffασ(nασ,G12ασ)
    N_spin_orbital=length(nασ)
    neffασ=zeros(N_spin_orbital)
    for i in 1:N_spin_orbital
        neffασ[i]=cal_neffInn(nασ[i],G12ασ[i])
    end
    neffασ
end

function cal_g12nonIntασ(nασ,G12ασ)
    N_spin_orbital=length(nασ)
    g12nonIntασ=zeros(N_spin_orbital)
    for i in 1:N_spin_orbital
        g12nonIntασ[i]=cal_gloc12NonInteractingInnloc(nασ[i],G12ασ[i])
    end
    g12nonIntασ
end



"""
then we could compute w_0^2
using 
digits(0,base=2,pad=4)
to get the binary represenation
w02=cal_w02(neffασ)
nασcheck=[dot(sqrt.(w02),g11matwSfull[i]*sqrt.(w02)) for i in 1:N_spin_orbital]
sum(abs.(nασcheck-nασ))
# so things are conssitent now
g12ασ=[dot(sqrt.(w02),g12matwSfull[i]*sqrt.(w02)) for i in 1:N_spin_orbital]
G12ασ
# we can also check g12ασ, as in its pure density projection
g12ασcheck=cal_g12nonIntασ(nασ,G12ασ)
sum(abs.(g12ασ-g12ασcheck))
"""
function cal_w02(neffασ)
    peffασ=1.0  .- neffασ
    N_spin_orbital=length(neffασ)
    N_Γ=2^N_spin_orbital
    # Γ=3                         # some test
    # we should reverse the order to agree with our definition and direct product
    w02=zeros(N_Γ)
    for Γ in 1:N_Γ
        Γασ=reverse(digits(Γ-1,base=2,pad=N_spin_orbital))
        w02[Γ]=prod(Γασ.*neffασ+(1.0 .- Γασ).*peffασ)
    end
    w02
end

"""
N_spin_orbital=4
idx=[1,2]
vη=cal_Vη(N_spin_orbital,[2,3])
vη=cal_Vη(N_spin_orbital,[2,3,4])
i=2
sum(vη)
[dot(diag(g11matwSfull[i]),vη) for i in 1:N_spin_orbital]
# we check they indeed satified the condition
"""
function cal_Vη(N_spin_orbital,idx)
    N_Γ=2^N_spin_orbital
    # Γ=3                         # some test
    # we should reverse the order to agree with our definition and direct product
    Vη=zeros(N_Γ)
    for Γ in 1:N_Γ
        Γασ=reverse(digits(Γ-1,base=2,pad=N_spin_orbital))
        Vη[Γ]=prod((Γασ.-0.5)[idx])
    end
    Vη
end

"""
now, we need a conventon to generate all idx (combination with n>=2)
i.e for N_spin_orbital=4
we have 
[1,2] ...
[1,2,3] ...
[1,2,3,4]..
The order does not matter, as long as we are consistent through the calculation
ηToIdx=gene_idx(N_spin_orbital)

"""
function gene_idx(N_spin_orbital)
    [idx for idx in combinations(1:N_spin_orbital) if length(idx)>=2]
end

"""
#2^Mx(2^M-M-1)matrix,Misnumberofspinorbital
#VΓη*xgivesthechangefromw0^2
x=rand(11)
# and we check the folllowing two are indeed zeros
k=VΓη*x
sum(VΓη*x)
[dot(diag(g11matwSfull[i]),VΓη*x) for i in 1:N_spin_orbital]
VΓη=gene_VΓη(N_spin_orbital)
"""
function gene_VΓη(N_spin_orbital)
    ηToIdx=gene_idx(N_spin_orbital)
    N_Γ=2^N_spin_orbital
    N_η=length(ηToIdx)
    VΓη=zeros(N_Γ,N_η)
    for i in 1:N_η
        VΓη[:,i]=cal_Vη(N_spin_orbital,ηToIdx[i])
    end
    VΓη
end

function gene_VΓη_ηToIdx(N_spin_orbital)
    ηToIdx=gene_idx(N_spin_orbital)
    N_Γ=2^N_spin_orbital
    N_η=length(ηToIdx)
    VΓη=zeros(N_Γ,N_η)
    for i in 1:N_η
        VΓη[:,i]=cal_Vη(N_spin_orbital,ηToIdx[i])
    end
    VΓη,ηToIdx
end

function norm1(k)
    sum(abs.(k))
end

function cal_k0_knorm(k)
    knorm=norm1(k)
    k0=k/knorm
    k0,knorm
end

"""
k=VΓη*x
k0,knorm=cal_k0_knorm(k)
kmax=cal_kmax(w02,k0)
w02+kmax*k0
"""
function cal_kmax(w02,k0)
    kmax=minimum([k for k in -w02./k0 if k>=0])
end

"""
we need some scheme to 
assuming k>0,kmax>0
regulate(0.5,1)
regulate(1.5,1)
regulate(10,1)
"""
function regulate(k,kmax)
    if(k>kmax)
        kmax*exp(-(k-kmax))
    else
        k
    end    
end

"""
now, we could combine this to regulate k, which is computed from
k=VΓη*x
w2=regulate_k(k*0.13,w02)+w02
# we may need to optimize regulate function, but the basis idea is to create a minimum at k=kmax
kmax*k0+w02
kmax/knorm
sqrt.(regulate_k(k*kmax/knorm,w02)+w02)
"""
function regulate_k(k,w02)
    k0,knorm=cal_k0_knorm(k)
    kmax=cal_kmax(w02,k0)
    regulate(knorm,kmax)*k0
end

"""
we combine prevoius result together
# the meaning of x is termined from 
w=cal_w(x,nασ,G12ασ)
ηToIdx=gene_idx(N_spin_orbital)
# the corresponding is negative for projection
w=cal_w(x,nασ,G12ασ)
g11matwSfull=cal_g11matwSfull(gAblockmat)
g12matwSfull=cal_g12matwSfull(gAblockmat)
[dot(w,g11matwSfull[i]*w) for i in 1:N_spin_orbital]
[dot(w,g12matwSfull[i]*w) for i in 1:N_spin_orbital]
"""
function cal_w(x,nασ,G12ασ)
    neffασ=cal_neffασ(nασ,G12ασ)
    w02=cal_w02(neffασ)
    N_spin_orbital=length(neffασ)
    VΓη=gene_VΓη(N_spin_orbital)
    k=VΓη*x
    w=sqrt.(regulate_k(k,w02)+w02)
end

function cal_w(x,nασ,G12ασ,VΓη)
    neffασ=cal_neffασ(nασ,G12ασ)
    w02=cal_w02(neffασ)
    N_spin_orbital=length(neffασ)
    # VΓη=gene_VΓη(N_spin_orbital)
    k=VΓη*x
    w=sqrt.(regulate_k(k,w02)+w02)
end

function cal_w(x,nασ,G12ασ,VΓη,w02)
    N_spin_orbital=length(neffασ)
    # VΓη=gene_VΓη(N_spin_orbital)
    k=VΓη*x
    w=sqrt.(regulate_k(k,w02)+w02)
end

"""
summarize previous code
gAblockmat=cal_GA_block(G12ασ)
g12matwSfull=cal_g12matwSfull(gAblockmat)
g12ασ=cal_g12ασ(w,g12matwSfull)
"""
function cal_g12ασ(w,g12matwSfull)
    N_spin_orbital=length(g12matwSfull)
    g12ασ=[dot(w,g12matwSfull[i]*w) for i in 1:N_spin_orbital]
end

function cal_gijασ(w,gijmatwSfull)
    N_spin_orbital=length(gijmatwSfull)
    gijασ=[dot(w,gijmatwSfull[i]*w) for i in 1:N_spin_orbital]
end

# now, we can compute the self-energy
# cal_sloc11sloc12(nloc,g012,g12)
"""
for each entry, we have [(Sloc11,Sloc12),...]
Slocασ=cal_Slocασ(nασ,G12ασ,g12ασ)
"""
function cal_Slocασ(nασ,G12ασ,g12ασ)
    N_spin_orbital=length(nασ)
    [cal_sloc11sloc12(nασ[idx],G12ασ[idx],g12ασ[idx]) for idx in 1:N_spin_orbital]
end

# now, we need to compute Δ, A<,A> from the momentum distribution
include("./load_band.jl")
"""
compute nX and AX for a given region of ασ, for given 
generate test data
e_fn=gene_spline_band("./es_inf.dat")
es=gene_ϵs(e_fn,nασ[1])
esX=es[1]
esX=[0.4]                       #  to test cal_alpha
weightX=nασ[1]
nX=weightX-0.04
αX=0.5
βX=0.3
cal_nk(α,β,ϵ)
we also need nkX to compute kinetic energy
"""
function cal_nX_AX_nkX(esX,αX,βX,weightX)
    N_points=length(esX)
    nkX=[ cal_nk(αX,βX,esX[idx]) for idx in 1:N_points]
    nX=mean(nkX)*weightX
    AX=mean(sqrt.(nkX.*(1.0 .- nkX)))*weightX
    nX,AX,nkX
end

function cal_nX(esX,αX,βX,weightX)
    N_points=length(esX)
    nkX=[ cal_nk(αX,βX,esX[idx]) for idx in 1:N_points]
    nX=mean(nkX)*weightX
    nX
end

__nX__cut_off__=0.0001
function constraint_nX(nX,weightX)
    global __nX__cut_off__
    if(nX>=weightX)
        nX=weightX-__nX__cut_off__
    elseif(nX<=0)
        nX=__nX__cut_off__
    else
        nX
    end
end

"""
# constraint_nX(-0.01,0.5)
"""
function solve_αX_from_nX(esX,nX,βX,weightX)
    # this should be within 0.0 and 1.0
    nX=constraint_nX(nX,weightX)
    n_mean=nX/weightX
    e_mean=mean(esX)
    αX_guess=cal_alpha(e_mean,βX,n_mean)
    αX=find_zero(αX->cal_nX(esX,αX,βX,weightX)-nX,αX_guess,Order0())
end

# we could first compute Δασ
"""
compute the charge transform 
Δασ=cal_Δασ(g12ασ,Slocασ)
"""
function cal_Δασ(g12ασ,Slocασ)
    N_spin_orbital=length(g12ασ)
    [ cal_delta(g12ασ[idx],Slocασ[idx]...) for idx in 1:N_spin_orbital]
end

# then we could compute Abelow and Aabove
# we compute then together Aασ as  [Abelow,Aabove] for each spin orbital
"""
for a given spin orbital
idx=1
nασ_=nασ[1]
Δασ_=Δασ[1]
# βbelow,βabove
βασ_=[0.1,0.1]
βασ=[[0.1,0.1] for _ in 1:N_spin_orbital]
eασ_=es
eασ=[ es for _ in 1:N_spin_orbital]
we should check the range of Δασ at some point
# we also return α in case we need
Aασ_,αασ_,nk_=cal_Aασ_αασ_nk_(nασ_,Δασ_,βασ_,eασ_)
# we also need to compute kinetic energy
"""
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

# id3=Matrix{Float64}(I,3,3)

"""
for a given spin orbital
# [s11,s12]
Slocασ_=Slocασ[1]
# cal_glocInnA(nloc,delta,Abelow,Aabove,sloc11,sloc12)
# we may consider replace this part with analytical expression in the future
# gloc_,Gασfull_=cal_gloc_Gloc_(nασ_,Δασ_,βασ_,eασ_,Slocασ_)
gloc_,Gασfull_,nk_,Ek_,αασ_=cal_gloc_Gloc_nk_Ek_αασ_(nασ_,Δασ_,βασ_,eασ_,Slocασ_)
we update to use the analytic form, its also useful to returne α
nασ_,Δασ_,βασ_,eασ_,Slocασ_=nασ[idx],Δασ[idx],βασ[idx],eασ[idx],Slocασ[idx]
"""
function cal_gloc_Gloc_nk_Ek_αασ_(nασ_,Δασ_,βασ_,eασ_,Slocασ_)
    Aασ_,αασ_,nk_=cal_Aασ_αασ_nk_(nασ_,Δασ_,βασ_,eασ_)
    Ek_=mean(nk_[1].*eασ_[1])*nασ_+mean(nk_[2].*eασ_[2])*(1-nασ_)
    gloc_=cal_glocInnA(nασ_,Δασ_,Aασ_...,Slocασ_...)
    s11,s12=Slocασ_
    sloc3_=[s11 s12 0; (-s12) s11 0; 0 0 1]
    # matrix form
    Gloc_=inv(id3+(inv(gloc_)-id3)*inv(sloc3_))
    # alternative way using the analytic way
    # g12,g13,g23,g31,g32=gloc_[1,2],gloc_[1,3],gloc_[2,3],gloc_[3,1],gloc_[3,2]
    # gloc_parts=gloc_[1,2],gloc_[1,3],gloc_[2,3],gloc_[3,1],gloc_[3,2]
    # Gloc_=cal_g0IngG12(nloc,g012,gloc_parts...)
    # Gloc_-Gloc_check
    # g012,g013,g023,g031,g032,g033
    Gασfull_=Gloc_[1,2],Gloc_[1,3],Gloc_[2,3],Gloc_[3,1],Gloc_[3,2],Gloc_[3,3]
    gloc_,Gασfull_,nk_,Ek_,αασ_
end
"""
using the analytic formula, G_remain as function of GA and gloc
"""
function cal_gloc_Gloc_nk_Ek_alternative(nασ_,Δασ_,βασ_,eασ_,Slocασ_,G12ασ_)
    Aασ_,αασ_,nk_=cal_Aασ_αασ_nk_(nασ_,Δασ_,βασ_,eασ_)
    Ek_=mean(nk_[1].*eασ_[1])*nασ_+mean(nk_[2].*eασ_[2])*(1-nασ_)
    gloc_=cal_glocInnA(nασ_,Δασ_,Aασ_...,Slocασ_...)
    # s11,s12=Slocασ_
    # sloc3_=[s11 s12 0; (-s12) s11 0; 0 0 1]
    # matrix form
    # Gloc_=inv(id3+(inv(gloc_)-id3)*inv(sloc3_))
    # alternative way using the analytic way
    g12,g13,g23,g31,g32=gloc_[1,2],gloc_[1,3],gloc_[2,3],gloc_[3,1],gloc_[3,2]
    gloc_parts=gloc_[1,2],gloc_[1,3],gloc_[2,3],gloc_[3,1],gloc_[3,2]
    Gloc_=cal_g0IngG12(nloc,G12ασ_,gloc_parts...)
    # Gloc_-Gloc_check
    # g012,g013,g023,g031,g032,g033
    Gασfull_=Gloc_[1,2],Gloc_[1,3],Gloc_[2,3],Gloc_[3,1],Gloc_[3,2],Gloc_[3,3]
    gloc_,Gασfull_,nk_,Ek_
end

"""
# gloc,Gloc=cal_gloc_Gloc(nασ,Δασ,βασ,eασ,Slocασ)
gloc,Gloc,nk,Ek=cal_gloc_Gloc_nk_Ek(nασ,Δασ,βασ,eασ,Slocασ)
g33matwασ=cal_G_block(Gloc)
g33matwfull=cal_g33matwfull(gAblockmat,g33matwασ)
nασcheck=[dot(w,g33matwfull[i]*w) for i in 1:N_spin_orbital]
nασcheck-nασ
g13check=[dot(w,g13matwfull[i]*w) for i in 1:N_spin_orbital]
g13check2=[gloc_[1,3] for gloc_ in gloc]
g13check-g13check2
g23check=[dot(w,g23matwfull[i]*w) for i in 1:N_spin_orbital]
g23check2=[gloc_[2,3] for gloc_ in gloc]
g23check-g23check2
g31check=[dot(w,g31matwfull[i]*w) for i in 1:N_spin_orbital]
g31check2=[gloc_[3,1] for gloc_ in gloc]
g31check-g31check2
g32check=[dot(w,g32matwfull[i]*w) for i in 1:N_spin_orbital]
g32check2=[gloc_[3,2] for gloc_ in gloc]
g32check-g32check2
# generate density interaction
nn12matwfull=cal_nnmatwfull(gAblockmat,g33matwασ,[1,2])
nn12check=dot(w,nn12matwfull*w)
nασ[1]*nασ[2]
g12,g13,g23,g31,g32=gloc_[1,2],gloc_[1,3],gloc_[2,3],gloc_[3,1],gloc_[3,2]
g012=Gασfull_[1]
nloc=gloc_[1,1]
@time Gloc_check=cal_g0IngG12(nloc,g012,g12,g13,g23,g31,g32)
Gασfull_check=Gloc_check[1,2],Gloc_check[1,3],Gloc_check[2,3],Gloc_check[3,1],Gloc_check[3,2],Gloc_check[3,3]
Gασfull_check.-Gασfull_

gloc,Gloc,nk,Ek,αασ=cal_gloc_Gloc_nk_Ek_αασ(nασ,Δασ,βασ,eασ,Slocασ)
idx=1
G12ασ
g12ασ
"""
function cal_gloc_Gloc_nk_Ek_αασ(nασ,Δασ,βασ,eασ,Slocασ)
    # in matrix form
    gloc=[]
    # in reduced form
    Gloc=[]
    nk=[]
    Ek=0
    αασ=[]
    N_spin_orbital=length(nασ)
    for idx in 1:N_spin_orbital
        gloc_,Gασfull_,nk_,Ek_,αασ_=cal_gloc_Gloc_nk_Ek_αασ_(nασ[idx],Δασ[idx],βασ[idx],eασ[idx],Slocασ[idx])
        # there are some speed up to use the analytic form, but not too much, so we just use the numerical approach.
        # @time gloc_check,Gασfull_check,nk_check,Ek_check=cal_gloc_Gloc_nk_Ek_alternative(nασ[idx],Δασ[idx],βασ[idx],eασ[idx],Slocασ[idx],G12ασ[idx])
        # Gασfull_.-Gασfull_check
        push!(gloc,gloc_)
        push!(Gloc,Gασfull_)
        push!(nk,nk_)
        push!(αασ,αασ_)
        Ek+=Ek_
    end
    gloc,Gloc,nk,Ek,αασ
end

# now, we could compute local-interaction
"""
we could simplified the caculation
here we just to use the general function to do this first
nloc=0.5 
g012ασ=0.5
N_spin_orbital=4
# explore the choice of x_reduce
    # 2,3,4
    # there are somthing funny about this
    # for only 0 and 4, negatnic U
    # we also check 
    x_reduce=[0.25,0.0,1.0]
    x=[ x_reduce[length(ηToIdx[i])-1] for i in 1:length(ηToIdx)]
    idx_n2=[i for i in 1:16 if count_particle_number(i)==2]
    w_target=zeros(16)
    w_target[idx_n2].=1
    # w_target[1]=sqrt(2)/2
    # w_target[16]=sqrt(2)/2
    # there are some correlation
    w_target2=w_target.^2
    w_target2=w_target2./(sum(w_target2))
    x_target=pinv(VΓη)*(w_target2-w02)
    x_reduce=[mean(x_target[1:6]),mean(x_target[7:10]),x_target[11]]
x_reduce=[-0.04,0.0,0.8]
βασ_=[0.5,0.5]
βασ_=[10.0,10.0]
# the band structure
e_fn=gene_spline_band("./es_inf.dat")
es=gene_ϵs(e_fn,nloc)
idx=1
N_spin_orbital=6
x_reduce=rand(N_spin_orbital-1)
N_spin_orbital=4
x_reduce=[-0.04,0.0,0.5]
nloc=0.5
g012ασ=0.38
Etot=compute_degenerate_case(nloc,g012ασ,x_reduce,Ueff,βασ_,N_spin_orbital)
using Optim
# res=optimize(x->( (x.-2.0).^2)[1],[0.0],BFGS())
# dump(typeof(res))
# res.minimizer

U=1.0
Ueff=6*U
para=[0.38,-0.1,0.0,0.5,0.2]
@time res=optimize(para->compute_degenerate_case(nloc,para[1],para[2:4],Ueff,[para[5],para[5]],N_spin_orbital)[1],para)
para=res.minimizer
E0=res.minimum
# we also return other observables
"""
function compute_degenerate_case(nloc,g012ασ,x_reduce,eασ,Ueff,βασ_,N_spin_orbital)
    nασ=[nloc for _ in 1:N_spin_orbital]
    G12ασ=[g012ασ for _ in 1:N_spin_orbital]
    βασ=[βασ_ for _ in 1:N_spin_orbital]
    # include pmat, g11matS, g12matS
    gAblockmat=cal_GA_block(G12ασ)
    g11matwSfull=cal_g11matwSfull(gAblockmat)
    g12matwSfull=cal_g12matwSfull(gAblockmat)
    # neffασ=cal_neffασ(nασ,G12ασ)
    # VΓη=gene_VΓη(N_spin_orbital)
    VΓη,ηToIdx=gene_VΓη_ηToIdx(N_spin_orbital)
    neffασ=cal_neffασ(nασ,G12ασ)
    w02=cal_w02(neffασ)
    # each entry corresponding the N-particle fluctuation (n_i-1/2)*...* (N-terms, for N=2, i.e idx=1 for x_reduce)
    x=[ x_reduce[length(ηToIdx[i])-1] for i in 1:length(ηToIdx)]
    w=cal_w(x,nασ,G12ασ,VΓη,w02)
    g12ασ=cal_gijασ(w,g12matwSfull)
    g11ασ=cal_gijασ(w,g11matwSfull)
    Slocασ=cal_Slocασ(nασ,G12ασ,g12ασ)
    Δασ=cal_Δασ(g12ασ,Slocασ)
    # Gloc is in the redueced form
    gloc,Gloc,nk,Ek,αασ=cal_gloc_Gloc_nk_Ek_αασ(nασ,Δασ,βασ,eασ,Slocασ)
    g33matwασ=cal_G_block(Gloc)
    nn12matwfull=cal_nnmatwfull(gAblockmat,g33matwασ,[1,2])
    nn12expt=cal_expt(w,nn12matwfull)
    Eloc=Ueff*nn12expt
    Etotal=Eloc+Ek
    Etotal,Eloc,Ek,αασ,nk,Δασ,Slocασ,nn12expt,x,w
end

"""
idx from 1
idx=2
[count_particle_number(i) for i in 1:16]
"""
function count_particle_number(idx)
    sum(digits(idx-1,base=2))
end

"""
assuming w is normalized
"""
function cal_expt(w,matfull)
    dot(w,matfull*w)
end
