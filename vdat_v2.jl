# here we add automatic differentiation

using LinearAlgebra
using Statistics
using Combinatorics
using Roots
using Zygote
include("./include_gene_code.jl")
# processing the band structure
include("./load_band.jl")

"""
only for a single orbital
pmatw,g11matwS,g12matwS=cal_p_g11_g12_mat(0.3)
t=rand(2,2)
gradient(x->sum(cal_p_g11_g12_wmatS(x)[3]),0.1)
we use G12 in the new code to consistent with the notation in paper
w indicate in the diagonal representation of pmatw and g11matwS
it seems there is not need to wrap these functions
we use a _ to indicate the single orbital version, in case the parameter type can full indicte it
"""
function cal_p_g11_g12_mat_(G12::Number)
    pmatw=cal_pmatw(G12)
    g11matwS=cal_g11matwS(G12)
    g12matwS=cal_g12matwS(G12)
    pmatw,g11matwS,g12matwS
end

"""
for single orbital
Gfull indicate all independend component 
we use Gloc to the matrix form
Update the code to reflect this convension
cal_g33_mat_([0.5,0.4,0.5,-0.3,-0.4,0.5])
para=[0.5,0.4,0.5,-0.3,-0.4,0.5]
gradient(x->[cal_g33_mat_(x),cal_g33_mat_(x)][2][2],para)
# so map works but the element wise apply does not work
gradient(x->map(cal_g33_mat_,[x,x])[2][2],para)
gradient(x->cal_g33_mat_.([x,x])[2][2],para)

"""
function cal_g33_mat_(Gfull)
    # g012,g013,g023,g031,g032,g033
    # g33matw=cal_g33matw(Gfull...)
    # cal_g33matw(Gfull...)
    cal_g33matw(Gfull[1],Gfull[2],Gfull[3],Gfull[4],Gfull[5],Gfull[6])
end

"""
cal_g33_g12_g23_g31_g32_mat_([0.5,0.4,0.5,-0.3,-0.4,0.5])
"""
function cal_g33_g12_g23_g31_g32_mat_(Gfull)
    # g012,g013,g023,g031,g032,g033
    # g33matw=cal_g33matw(Gfull...)
    # we should explicit write the arguments
    # cal_g33matw(Gfull[1],Gfull[2],Gfull[3],Gfull[4],Gfull[5],Gfull[6])
    g33matw=cal_g33matw(Gfull[1],Gfull[2],Gfull[3],Gfull[4],Gfull[5],Gfull[6])
    g13matw=cal_g13matw(Gfull[1],Gfull[2],Gfull[3],Gfull[4],Gfull[5],Gfull[6])
    g23matw=cal_g23matw(Gfull[1],Gfull[2],Gfull[3],Gfull[4],Gfull[5],Gfull[6])
    g31matw=cal_g31matw(Gfull[1],Gfull[2],Gfull[3],Gfull[4],Gfull[5],Gfull[6])
    g32matw=cal_g32matw(Gfull[1],Gfull[2],Gfull[3],Gfull[4],Gfull[5],Gfull[6])
    g33matw,g13matw,g23matw,g31matw,g32matw
end

"""
we rewrite it for easy automatic diffferentiation
G12ασ=[0.5,0.5,0.5,0.5]
pmatwασ,g11matwSασ,g12matwSασ=cal_p_g11_g12_mat(G12ασ)
f=x->sum(kron(cal_p_g11_g12_mat(x)[3]...))
@time gradient(f,[0.5,0.5])
"""
function cal_p_g11_g12_mat(G12ασ::Array)
    pmatwασ=cal_pmatw.(G12ασ)
    g11matwSασ=cal_g11matwS.(G12ασ)
    g12matwSασ=cal_g12matwS.(G12ασ)
    pmatwασ,g11matwSασ,g12matwSασ
end

"""
in practice, we only need this too, actualy, pmat is know too
pmatwασ,g12matwSασ=cal_p_g12_mat(G12ασ)
"""
function cal_p_g12_mat(G12ασ::Array)
    pmatwασ=cal_pmatw.(G12ασ)
    g12matwSασ=cal_g12matwS.(G12ασ)
    pmatwασ,g12matwSασ
end

"""
Gfull=[0.5,0.4,0.5,-0.3,-0.4,0.5]
we use ασ to indicates it is spin orbital dependent
Gfullασ=[Gfull for _ in 1:4]
sum(sum(cal_g33_mat(Gfullασ)))
# this works
gradient(x->cal_g33_mat([x,x])[1][2],Gfull)
gradient(x->sum(sum(cal_g33_mat([x,x]))),Gfull)
gradient(x->cal_g33_mat_.([x,x])[1][2],Gfull)
# but not, it seems we should pass it 
gradient((x...)->cal_g33_mat([x[1],x[2]])[1][2],Gfullασ[1],Gfullασ[2])
# there are some problem that some combination are not defined, 
# we should use map
# it seems one should define ProjtectTo 
# checkout how it works
gradient(x->sum(x[1]+x[2]),[[1,2],[2,3]])
"""
function cal_g33_mat(Gfullασ)
    cal_g33_mat_.(Gfullασ)
end

"""
G12ασ=[0.5,0.5,0.5,0.5]
pmatwασ,g11matwSασ,g12matwSασ=cal_p_g11_g12_mat(G12ασ)
# how to represent this?
idx=2
check1=cal_g11matwSfull(pmatwασ,g11matwSασ,2)
check2=cal_g11matwSfull_old([pmatwασ,g11matwSασ,g12matwSασ])
sum(abs.(check1-check2[2]))
"""
# function cal_g11matwSfull(pmatwασ,g11matwSασ,idx)
#     # # pmatwασ,g11matwSασ,g12matwSασ=gAblockmat
#     # g11matwSfull=[]
#     # # loop over all spin orbitals
#     # for idx in 1:length(pmatwασ)
#     #     args=copy(pmatwασ)
#     #     args[idx]=g11matwSασ[idx]
#     #     push!(g11matwSfull,kron(args...))
#     # end
#     # g11matwSfull
#     kron(pmatwασ[1:(idx-1)]...,g11matwSασ[idx],pmatwασ[(idx+1):end]...)
# end

"""
we can merge into a general function
G12ασ=[0.5,0.5,0.5,0.5]
f=x->(();)))
f([0.0,0.0])
x=[0.1,0.1]
function test(x)
    pmatwασ,g11matwSασ,g12matwSασ=cal_p_g11_g12_mat(x)
    sum(abs.(cal_Xmatfull(pmatwασ,g12matwSασ,1)))
end
gradient(test,[0.3,0.2,0.4,0.2])
"""
function cal_Xmatfull(pmatwασ,Xmatασ,idx)
    # # pmatwασ,Xmatασ,g12matασ=gAblockmat
    # Xmatfull=[]
    # # loop over all spin orbitals
    # for idx in 1:length(pmatwασ)
    #     args=copy(pmatwασ)
    #     args[idx]=Xmatασ[idx]
    #     push!(Xmatfull,kron(args...))
    # end
    # Xmatfull
    kron(pmatwασ[1:(idx-1)]...,Xmatασ[idx],pmatwασ[(idx+1):end]...)
end

# function cal_g11matwSfull_old(gAblockmat)
#     pmatwασ,g11matwSασ,g12matwSασ=gAblockmat
#     g11matwSfull=[]
#     # loop over all spin orbitals
#     for idx in 1:length(pmatwασ)
#         args=copy(pmatwασ)
#         args[idx]=g11matwSασ[idx]
#         push!(g11matwSfull,kron(args...))
#     end
#     g11matwSfull
# end
"""
we assume idx1<idx2
G12ασ=[0.5,0.5,0.5,0.5]
pmatwασ,g11matwSασ,g12matwSασ=cal_p_g11_g12_mat(G12ασ)
check1=cal_Xmatfull(pmatwασ,g12matwSασ,1,4)
check2=cal_Xmatfull_old(pmatwασ,g12matwSασ,[1,4])
sum(abs.(check1-check2))
# right now, we just need 
"""
function cal_Xmatfull(pmatwασ,Xmatασ,idx1,idx2)
    kron(pmatwασ[1:(idx1-1)]...,Xmatασ[idx1],pmatwασ[(idx1+1):(idx2-1)]...,Xmatασ[idx2],pmatwασ[(idx2+1):end]...)
end

# """
# general, but old approach (not suitable to automatic differentiation)
# """
# function cal_Xmatfull_old(pmatwασ,g33matwασ,idx)
#     args=copy(pmatwασ)
#     for idx_ in idx
#         args[idx]=g33matwασ[idx]
#     end
#     kron(args...)
# end

"""
effective density, use to construct w02
nασ=[0.4,0.5,0.6,0.7]
G12ασ=[0.3,0.4,0.5,0.5]
neffασ=cal_neffασ(nασ,G12ασ)
gradient((x,y)->sum(cal_neffασ(x,y)),nασ,G12ασ)
"""
function cal_neffασ(nασ,G12ασ)
    N_spin_orbital=length(nασ)
    # neffασ=zeros(N_spin_orbital)
    # for i in 1:N_spin_orbital
    #     neffασ[i]=cal_neffInn(nασ[i],G12ασ[i])
    # end
    # neffασ
    [cal_neffInn(nασ[i],G12ασ[i])  for i in 1:N_spin_orbital]
end

"""
w02=cal_w02(neffασ)
gradient((x,y)->cal_w02(cal_neffασ(x,y))[1],[0.2,0.3],[0.2,0.2])

"""
function cal_w02(neffασ)
    peffασ=1.0  .- neffασ
    N_spin_orbital=length(neffασ)
    N_Γ=2^N_spin_orbital
    # Γ=3                         # some test
    # we should reverse the order to agree with our definition and direct product
    # w02=zeros(N_Γ)
    # for Γ in 1:N_Γ
    #     Γασ=reverse(digits(Γ-1,base=2,pad=N_spin_orbital))
    #     w02[Γ]=prod(Γασ.*neffασ+(1.0 .- Γασ).*peffασ)
    # end
    # w02
    w02=[cal_w02_Γ(Γ,N_spin_orbital,neffασ,peffασ) for Γ in 1:N_Γ]
end

"""
cal_Γασ(1,4)
# the zygote does note like reverse
# so we need to store it 
"""
function cal_Γασ(Γ,N_spin_orbital)
    global __global__Γασ__
    __global__Γασ__[N_spin_orbital][Γ]
end
# one need to update
const __global__Γασ__=Dict()

"""
one should run these before using 
update__global__Γασ__(4)
update__global__Γασ__(2)
"""
function update__global__Γασ__(N_spin_orbital)
    global __global__Γασ__
    __global__Γασ__[N_spin_orbital]=[reverse(digits(Γ-1,base=2,pad=N_spin_orbital)) for Γ in 1:2^N_spin_orbital]
end



"""
gradient(neffασ->cal_w02_Γ(1,4,neffασ,1.0 .- neffασ),neffασ)
N_spin_orbital=4
Γασ=cal_Γασ(10,N_spin_orbital)
cal_w02_Γ(10,4,neffασ,1.0 .- neffασ)
"""
function cal_w02_Γ(Γ,N_spin_orbital,neffασ,peffασ)
    Γασ=cal_Γασ(Γ,N_spin_orbital)
    # Γασ=[1,0,0,1]
    prod(Γασ.*neffασ+(1.0 .- Γασ).*peffασ)
end

"""
The fluctuation matrix in w representation, only dependes on idx
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

# """
# #2^Mx(2^M-M-1)matrix,Misnumberofspinorbital
# #VΓη*xgivesthechangefromw0^2
# x=rand(11)
# # and we check the folllowing two are indeed zeros
# k=VΓη*x
# sum(VΓη*x)
# [dot(diag(g11matwSfull[i]),VΓη*x) for i in 1:N_spin_orbital]
# VΓη=gene_VΓη(N_spin_orbital)
# """
# function gene_VΓη(N_spin_orbital)
#     ηToIdx=gene_idx(N_spin_orbital)
#     N_Γ=2^N_spin_orbital
#     N_η=length(ηToIdx)
#     VΓη=zeros(N_Γ,N_η)
#     for i in 1:N_η
#         VΓη[:,i]=cal_Vη(N_spin_orbital,ηToIdx[i])
#     end
#     VΓη
# end

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

# we similary store the result to avoid problem in automatic differentiation
const __global_VΓη_ηToIdx__=Dict()
"""
similarly to update__global_Γασ__, one need to update
update__global_VΓη_ηToIdx__(4)
update__global_VΓη_ηToIdx__(2)
__global_VΓη_ηToIdx__[4]
"""
function update__global_VΓη_ηToIdx__(N_spin_orbital)
    __global_VΓη_ηToIdx__[N_spin_orbital]=gene_VΓη_ηToIdx(N_spin_orbital)
end


# there are some function to constraint the input of x, we will put is later
"""
# here, we don't restrict x, will will check when update the parameter, this is used to compute the energy and the gradient
nασ=[0.6,0.6,0.4,0.4]
G12ασ=[0.4,0.4,0.4,0.4]
x=rand(2^4-1-4)*0.01
cal_w(x,nασ,G12ασ)
sum(w02)
gradient(x->sum(cal_w(x,nασ,G12ασ)),x)
we add the scaled version
"""
function cal_w(x,nασ,G12ασ)
    neffασ=cal_neffασ(nασ,G12ασ)
    w02=cal_w02(neffασ)
    N_spin_orbital=length(neffασ)
    VΓη,ηToIdx=__global_VΓη_ηToIdx__[N_spin_orbital]
    k=VΓη*x
    # w=sqrt.(regulate_k(k,w02)+w02)
    w=sqrt.(k+w02)
end

"""
we use an exponential scale (tanh) so that we have a smooth function
cal_w_scaled(x*10,nασ,G12ασ)

gradient(x->sum(cal_w_scaled(x,nασ,G12ασ)),x*10)

so now, x can be treat a free varaibles
"""
function cal_w_scaled(x,nασ,G12ασ)
    neffασ=cal_neffασ(nασ,G12ασ)
    w02=cal_w02(neffασ)
    N_spin_orbital=length(neffασ)
    VΓη,ηToIdx=__global_VΓη_ηToIdx__[N_spin_orbital]
    k=VΓη*x
    k0,knorm=cal_k0_knorm(k)
    kmax=cal_maxnorm(w02,k0)
    # we regulate
    knormscaled=kmax*tanh(knorm/kmax)
    kscaled=k0*knormscaled
    w=sqrt.(kscaled+w02)
end


function norm1(k)
    sum(abs.(k))
end


"""
k0 is the unit vector of k, and knorm is its norm, using norm1
k0,knorm=cal_k0_knorm(k)
norm1(k0)
cal_maxnorm(w02,k0)
"""
function cal_k0_knorm(k)
    knorm=norm1(k)
    k0=k/knorm
    k0,knorm
end

Base.convert(::Type{Int64},x::Nothing)=0

"""
found the maximal norm
map(x->if(x==0) ,[0.0,0.1])
cal_maxnorm(w02,k0)
gradient(k0->cal_maxnorm(w02,k0),k0)
w02=[0.062 for i in 1:16]
k0=rand(16).-0.5
gradient(k0->cal_maxnorm(w02,k0),k0)

typeof(Nothing)
a=Nothing()
typeof(a)
typeof(Int)
"""
function cal_maxnorm(w02,k0)
    # print("maxnorm with k0 $(k0)")
    kmax=minimum([-w02[i]/k0[i] for i in 1:length(w02)  if k0[i]<0])
end


"""
G12ασ=[0.4,0.4,0.4,0.4]
pmatwασ,g11matwSασ,g12matwSασ=cal_p_g11_g12_mat(G12ασ)
w=sqrt.(w02)
!! cal g12ασ
x=rand(2^4-1-4)*0.05
w=cal_w_scaled(x,nασ,G12ασ)
g12ασ=[expt(w,cal_Xmatfull(pmatwασ,g12matwSασ,i)) for i in 1:N_spin_orbital]
"""
function expt(w,Xmatfull)
    dot(w,Xmatfull*w)
end

"""
for each entry, we have [(Sloc11,Sloc12),...]
Slocασ=cal_Slocασ(nασ,G12ασ,g12ασ)
gradient(x->cal_Slocασ(nασ,G12ασ,x)[2][1],g12ασ)
"""
function cal_Slocασ(nασ,G12ασ,g12ασ)
    N_spin_orbital=length(nασ)
    [cal_sloc11sloc12(nασ[idx],G12ασ[idx],g12ασ[idx]) for idx in 1:N_spin_orbital]
end


# now, we explroe the momentum part

"""
for each region, X (< or >)
we can either use αX, or βX
or nX and AX
These are just Legendre transformations
KX(αX,βX)=\sum_k∈X ϵkX*nkX
as we minimize 
KX(nX,AX)-αX*nX-βX*AX
from the beginning
->KX(nX,AX)
∂KX/∂nX=αX
∂KX/∂AX=βX
# we can verify this numerically
e_fn=gene_spline_band("./es_inf.dat")
es=gene_ϵs(e_fn,nασ[1])
esX=es[1]
# esX=[0.4]                       #  to test cal_alpha
weightX=nασ[1]
nX=weightX-0.04
αX=4.0
βX=2.0
nX,AX,nkX=cal_nX_AX_nkX(esX,αX,βX,weightX)

"""
function cal_nX_AX_nkX(esX,αX,βX,weightX)
    nkX=[ cal_nk(αX,βX,esX_) for esX_ in esX]
    nX=mean(nkX)*weightX
    AX=mean(sqrt.(nkX.*(1.0 .- nkX)))*weightX
    nX,AX,nkX
end

"""
we check the kinectic energy
"""
function cal_nX_AX_KX_nkX(esX,αX,βX,weightX)
    nkX=[ cal_nk(αX,βX,esX_) for esX_ in esX]
    nX=mean(nkX)*weightX
    AX=mean(sqrt.(nkX.*(1.0 .- nkX)))*weightX
    KX=mean(esX.*nkX)*weightX
    nX,AX,KX,nkX
end

"""
we only compute nX, to perform the Legendre transformation
"""
function cal_nX(esX,αX,βX,weightX)
    nkX=[ cal_nk(αX,βX,esX_) for esX_ in esX]
    nX=mean(nkX)*weightX
end

# one should constraint Δ, or nX, we will do that later
"""
we check the relation
αX=solve_αX_from_nX(esX,weightX-0.1,βX,weightX)
dnXdαdβ=gradient((αX,βX)->cal_nX_AX_nkX(esX,αX,βX,weightX)[1],αX,βX)
dAXdαdβ=gradient((αX,βX)->cal_nX_AX_nkX(esX,αX,βX,weightX)[2],αX,βX)
dKdαdβ=gradient((αX,βX)->mean(cal_nX_AX_nkX(esX,αX,βX,weightX)[3].*esX)*weightX,αX,βX)
dnXdAdαdβ=vcat(reshape([dnXdαdβ...],1,2), reshape( [dAXdαdβ...],1,2))
dαdβdnXdA=inv(dnXdAdαdβ)
dKdαdβ=reshape([dKdαdβ...],1,2)
dKdnXdA=dKdαdβ*dαdβdnXdA
dKdnXdA.-[αX βX]
# we check they are indeed same
"""
function solve_αX_from_nX(esX,nX,βX,weightX)
    # this should be within 0.0 and 1.0
    # nX=constraint_nX(nX,weightX)
    n_mean=nX/weightX
    e_mean=mean(esX)
    αX_guess=cal_alpha(e_mean,βX,n_mean)
    αX=find_zero(αX->cal_nX(esX,αX,βX,weightX)-nX,αX_guess,Order0())
end


#we use the analytic formulas to compute the deriviates
"""
We compute the six derivatives. Kx,nX,AX, regarding with α,β
We numerically check the derivatives
((cal_nX_AX_KX_nkX(esX,αX+1e-4,βX,weightX)[3]-cal_nX_AX_KX_nkX(esX,αX,βX,weightX)[3])/(1e-4))/dKXdα
((cal_nX_AX_KX_nkX(esX,αX,βX+1e-4,weightX)[3]-cal_nX_AX_KX_nkX(esX,αX,βX,weightX)[3])/(1e-4))/dKXdβ
((cal_nX_AX_KX_nkX(esX,αX+1e-4,βX,weightX)[1]-cal_nX_AX_KX_nkX(esX,αX,βX,weightX)[1])/(1e-4))/dnXdα
((cal_nX_AX_KX_nkX(esX,αX,βX+1e-4,weightX)[1]-cal_nX_AX_KX_nkX(esX,αX,βX,weightX)[1])/(1e-4))/dnXdβ
((cal_nX_AX_KX_nkX(esX,αX+1e-4,βX,weightX)[2]-cal_nX_AX_KX_nkX(esX,αX,βX,weightX)[2])/(1e-4))/dAXdα
((cal_nX_AX_KX_nkX(esX,αX,βX+1e-4,weightX)[2]-cal_nX_AX_KX_nkX(esX,αX,βX,weightX)[2])/(1e-4))/dAXdβ
# we have checked that all derivatives are correct
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
now, we compute the derivatives in terms of nX,βX
so only need to compute dKX and dAX
dKXdn_,dKXdβ_,dAXdn_,dAXdβ_=cal_dKX_dAX_dnβ(esX,αX,βX,weightX)
nX,AX,KX,nkX=cal_nX_AX_KX_nkX(esX,αX,βX,weightX)
αXnew=solve_αX_from_nX(esX,nX+1e-4,βX,weightX)
nXnew,AXnew,KXnew,nkXnew=cal_nX_AX_KX_nkX(esX,αXnew,βX,weightX)
((KXnew-KX)/1e-4)/dKXdn_
((AXnew-AX)/1e-4)/dAXdn_
# so we have checked that these two are correct
αXnew=solve_αX_from_nX(esX,nX,βX+1e-4,weightX)
nXnew,AXnew,KXnew,nkXnew=cal_nX_AX_KX_nkX(esX,αXnew,βX+1e-4,weightX)
((KXnew-KX)/1e-4)/dKXdβ_
((AXnew-AX)/1e-4)/dAXdβ_
# and these two are correct too.

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
using the new scheme
We also need to scale the k so w2 is well defined
nασ=[0.5,0.5,0.5,0.5]
G12ασ=[0.4,0.4,0.4,0.4]
e_fn=gene_spline_band("./es_inf.dat")
es=gene_ϵs(e_fn,nασ[1])
eασ=[ es for _ in 1:N_spin_orbital]
βασ=[[1.0,1.0] for _ in 1:N_spin_orbital]
We seperate the below and above region, so it is easy for the next step
# all derivatives are in n,β basis
# (Δασ,nασ_below,nασ_above,αασ_below,αασ_above,Aασ_below,Aασ_above,Kασ_below,Kασ_above,∂Kασ∂nX_below,∂Kασ∂nX_above,∂Kασ∂βX_below,∂Kασ∂βX_above,∂Aασ∂nX_below,∂Aασ∂nX_above,∂Aασ∂βX_below,∂Aασ∂βX_above)=cal_momentum_part_in_nβ(nασ,G12ασ,x,βασ,eασ)
(Δασ,nασ_below,nασ_above,αασ_below,αασ_above,nkασ_below,nkασ_above,Aασ_below,Aασ_above,Kασ_below,Kασ_above,∂Kασ∂nX_below,∂Kασ∂nX_above,∂Kασ∂βX_below,∂Kασ∂βX_above,∂Aασ∂nX_below,∂Aασ∂nX_above,∂Aασ∂βX_below,∂Aασ∂βX_above)=cal_momentum_part_in_nβ(nασ,G12ασ,x,βασ,eασ)

"""
function cal_momentum_part_in_nβ(nασ,G12ασ,x,βασ,eασ)
    N_spin_orbital=length(nασ)
    w=cal_w_scaled(x,nασ,G12ασ)
    pmatwασ,g12matwSασ=cal_p_g12_mat(G12ασ)
    g12ασ=[expt(w,cal_Xmatfull(pmatwασ,g12matwSασ,i)) for i in 1:N_spin_orbital]
    Slocασ=cal_Slocασ(nασ,G12ασ,g12ασ)
    Δασ=cal_Δασ(g12ασ,Slocασ)
    # we will compute the seperatly the below and above seperately
    # we add them, so don't need another function
    nkασ_below=[]
    nkασ_above=[]
    nασ_below=nασ-Δασ
    nασ_above=Δασ
    αασ_below=[]
    αασ_above=[]
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
        αασ_below_=solve_αX_from_nX(eασ[i][1],nασ_below[i],βασ[i][1],nασ[i])
        para_below=(eασ[i][1],αασ_below_,βασ[i][1],nασ[i])
        dKbelowdn_,dKbelowdβ_,dAbelowdn_,dAbelowdβ_=cal_dKX_dAX_dnβ(para_below...)
        nασ_below_,Aασ_below_,Kασ_below_,nkασ_below_=cal_nX_AX_KX_nkX(para_below...)
        αασ_above_=solve_αX_from_nX(eασ[i][2],nασ_above[i],βασ[i][2],1.0-nασ[i])
        para_above=(eασ[i][2],αασ_above_,βασ[i][2],1.0-nασ[i])
        dKabovedn_,dKabovedβ_,dAabovedn_,dAabovedβ_=cal_dKX_dAX_dnβ(para_above...)
        nασ_above_,Aασ_above_,Kασ_above_,nkασ_above_=cal_nX_AX_KX_nkX(para_above...)
        push!(nkασ_below,nkασ_below_)
        push!(nkασ_above,nkασ_above_)
        push!(αασ_below,αασ_below_)
        push!(αασ_above,αασ_above_)
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
    (Δασ,nασ_below,nασ_above,αασ_below,αασ_above,nkασ_below,nkασ_above,Aασ_below,Aασ_above,Kασ_below,Kασ_above,∂Kασ∂nX_below,∂Kασ∂nX_above,∂Kασ∂βX_below,∂Kασ∂βX_above,∂Aασ∂nX_below,∂Aασ∂nX_above,∂Aασ∂βX_below,∂Aασ∂βX_above)
end


"""
compute the charge transform 
Δασ=cal_Δασ(g12ασ,Slocασ)
there are some problems
it seems unpack data will cost some problem, in projectTo 
"""
function cal_Δασ(g12ασ,Slocασ)
    N_spin_orbital=length(g12ασ)
    # [ cal_delta(g12ασ[idx],Slocασ[idx]...) for idx in 1:N_spin_orbital]
    [ cal_delta(g12ασ[idx],Slocασ[idx][1],Slocασ[idx][2]) for idx in 1:N_spin_orbital]
end

"""
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
# I guess once we put nk to the cal_momentum_part_in_nβ, we actually don't need it
# remove this in future!!
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


"""
Extra indicates the b,c,d blocks
Slocασ_=Slocασ[1]
G12ασ_=G12ασ[1]
g12ασ_=g12ασ[1]
nασ_=nασ[1]
gextraασ_,Gfullασ_=cal_gextra_Gfull(nασ_,G12ασ_,g12ασ_,Aασ_,Slocασ_)
cal_g33_mat_(Gfullασ_)
gradient(nασ_->sum(cal_gextra_Gfull(nασ_,G12ασ_,g12ασ_,Aασ_,Slocασ_)[2]),nασ_)
we could directly write down the indeces
# remove this function in future
"""
function cal_gextra_Gfull(nασ_,G12ασ_,g12ασ_,Aασ_,Slocασ_)
    # glocReduced=cal_glocReducedInA(Abelow,Aabove,sloc11,sloc12)
    # # [gloc[1,3],gloc[2,3],gloc[3,1],gloc[3,2]].-glocReduced
    gextraασ_=cal_glocReducedInA(Aασ_...,Slocασ_...)
    # cal_g0IngG12(nloc,g012,g12,g13,g23,g31,g32)
    # Gloc_=cal_g0IngG12(nασ_,G12ασ_,g12ασ_,gextraασ_...)
    # Gfullασ_check=[Gloc_[1,2],Gloc_[1,3],Gloc_[2,3],Gloc_[3,1],Gloc_[3,2],Gloc_[3,3]]
    # we could combine this to one function
    Gfullασ_=cal_g0fullIngG12(nασ_,G12ασ_,g12ασ_,gextraασ_...)
    # Gfullασ_.-Gfullασ_check # we checked, it is correct
    gextraασ_,Gfullασ_
end

"""
to make it easy for automatic differnentiaion, one may want to explicit list the arguments
we use the new scheme, split A to below and above,
#remove this function in future
"""
function cal_Gfull(nασ_,G12ασ_,g12ασ_,Aασ_,Slocασ_)
    # glocReduced=cal_glocReducedInA(Abelow,Aabove,sloc11,sloc12)
    # # [gloc[1,3],gloc[2,3],gloc[3,1],gloc[3,2]].-glocReduced
    # gextraασ_=cal_glocReducedInA(Aασ_...,Slocασ_...)
    gextraασ_=cal_glocReducedInA(Aασ_[1],Aασ_[2],Slocασ_[1],Slocασ_[2])
    # cal_g0IngG12(nloc,g012,g12,g13,g23,g31,g32)
    # Gloc_=cal_g0IngG12(nασ_,G12ασ_,g12ασ_,gextraασ_...)
    # Gfullασ_check=[Gloc_[1,2],Gloc_[1,3],Gloc_[2,3],Gloc_[3,1],Gloc_[3,2],Gloc_[3,3]]
    # we could combine this to one function
    # Gfullασ_=cal_g0fullIngG12(nασ_,G12ασ_,g12ασ_,gextraασ_...)
    Gfullασ_=cal_g0fullIngG12(nασ_,G12ασ_,g12ασ_,gextraασ_[1],gextraασ_[2],gextraασ_[3],gextraασ_[4])
    # Gfullασ_.-Gfullασ_check # we checked, it is correct
end

"""
as we now break Aασ to below and above
"""
function cal_Gfull(nασ_,G12ασ_,g12ασ_,Aασ_below_,Aασ_above_,Slocασ_)
    # glocReduced=cal_glocReducedInA(Abelow,Aabove,sloc11,sloc12)
    # # [gloc[1,3],gloc[2,3],gloc[3,1],gloc[3,2]].-glocReduced
    # gextraασ_=cal_glocReducedInA(Aασ_...,Slocασ_...)
    gextraασ_=cal_glocReducedInA(Aασ_[1],Aασ_[2],Slocασ_[1],Slocασ_[2])
    # cal_g0IngG12(nloc,g012,g12,g13,g23,g31,g32)
    # Gloc_=cal_g0IngG12(nασ_,G12ασ_,g12ασ_,gextraασ_...)
    # Gfullασ_check=[Gloc_[1,2],Gloc_[1,3],Gloc_[2,3],Gloc_[3,1],Gloc_[3,2],Gloc_[3,3]]
    # we could combine this to one function
    # Gfullασ_=cal_g0fullIngG12(nασ_,G12ασ_,g12ασ_,gextraασ_...)
    Gfullασ_=cal_g0fullIngG12(nασ_,G12ασ_,g12ασ_,gextraασ_[1],gextraασ_[2],gextraασ_[3],gextraασ_[4])
    # Gfullασ_.-Gfullασ_check # we checked, it is correct
end


# now, we start to compute the derivatives,
# there are two perspectives,
# single particle density function and local mixed approach
# here, we use the latter
"""
we treat Δασ as the in
(notice we also need to expand AXασ too)
We first solve the momentum part, i.e find the αX for the given Δ
nασ=[0.5,0.5,0.5,0.5]
G12ασ=[0.4,0.4,0.4,0.4]
N_spin_orbital=length(nασ)
x=rand(16-4-1)*0.1
pmatwασ,g11matwSασ,g12matwSασ=cal_p_g11_g12_mat(G12ασ)
# just to check
g11ασ=[expt(w,cal_Xmatfull(pmatwασ,g11matwSασ,i)) for i in 1:N_spin_orbital]
Notice this part, we don't need to make the derivatives    
e_fn=gene_spline_band("./es_inf.dat")
es=gene_ϵs(e_fn,nασ[1])
eασ_=es
eασ=[ es for _ in 1:N_spin_orbital]
αασ,Δασ,Aασ,K0,∂K∂Δασ,∂K∂Aασ=cal_momemtum_part(nασ,G12ασ,x,βασ,eασ)
"""
function cal_momemtum_part(nασ,G12ασ,x,βασ,eασ)
    N_spin_orbital=length(nασ)
    w=cal_w(x,nασ,G12ασ)
    pmatwασ,g12matwSασ=cal_p_g12_mat(G12ασ)
    g12ασ=[expt(w,cal_Xmatfull(pmatwασ,g12matwSασ,i)) for i in 1:N_spin_orbital]
    Slocασ=cal_Slocασ(nασ,G12ασ,g12ασ)
    Δασ=cal_Δασ(g12ασ,Slocασ)
    Aασ=[]
    αασ=[]
    nk=[]
    K0=0                        # kinecit energy
    ∂K∂Δασ=[]
    ∂K∂Aασ=[]
    for i in 1:N_spin_orbital
        Aασ_,αασ_,nk_=cal_Aασ_αασ_nk_(nασ[i],Δασ[i],βασ[i],eασ[i])
        K0ασ_=mean(nk_[1].*eασ[i][1])*nασ[i]+mean(nk_[2].*eασ[i][2])*(1-nασ[i])
        K0+=K0ασ_
        push!(Aασ,Aασ_)
        push!(αασ,αασ_)
        push!(nk,nk_)
        push!(∂K∂Δασ,αασ_[2]-αασ_[1])
        push!(∂K∂Aασ,βασ[i])
    end
    # we also
    αασ,Δασ,Aασ,K0,∂K∂Δασ,∂K∂Aασ
end

"""
now, we can compute the derivatives
We have expand the kinetic  energy to the first order
We need to compute the derivatives to G12ασ,x,Aασ to the first order
ΔασStar=Δασ
We have remove the constrant part, this is only for automatic differentiaion
We also provide a general mechanism to pass the 2 body density-density interaction
U=1.0
interaction=[(1,2,U),(1,3,U),(1,4,U),(2,3,U),(2,4,U),(3,4,U)]
format
[(idx1,idx2,coupling)...]
cal_energy_for_diff(G12ασ,x,Aασ,nασ,∂K∂Δασ,∂K∂Aασ,interaction)
gradient((G12ασ,x,Aασ)->cal_energy_for_diff(G12ασ,x,Aασ,nασ,∂K∂Δασ,∂K∂Aασ,interaction),G12ασ,x,Aασ)
"""
function cal_energy_for_diff(G12ασ,x,Aασ,nασ,∂K∂Δασ,∂K∂Aασ,interaction)
    N_spin_orbital=length(nασ)
    w=cal_w(x,nασ,G12ασ)
    pmatwασ,g12matwSασ=cal_p_g12_mat(G12ασ)
    g12ασ=[expt(w,cal_Xmatfull(pmatwασ,g12matwSασ,i)) for i in 1:N_spin_orbital]
    Slocασ=cal_Slocασ(nασ,G12ασ,g12ασ)
    Δασ=cal_Δασ(g12ασ,Slocασ)
    dK_from_Δ=dot(∂K∂Δασ,Δασ)
    dK_from_A=sum([dot(∂K∂Aασ[i],Aασ[i]) for i in 1:N_spin_orbital])
    dK_from_Δ+dK_from_A
    g33matwασ=[cal_g33_mat_(cal_Gfull(nασ[i],G12ασ[i],g12ασ[i],Aασ[i],Slocασ[i])) for i in 1:N_spin_orbital]
    Eloc=sum([coefficient*expt(w,cal_Xmatfull(pmatwασ,g33matwασ,idx1,idx2)) for (idx1,idx2,coefficient) in interaction ])
    Eloc+dK_from_Δ+dK_from_A
end

# now, we compute gradient
"""
    # using the two band half-fiing case
    data_dir="./two_band_degenerate_inf_half"
    N_spin_orbital=4
    nασ=[0.5 for _ in 1:N_spin_orbital]
    U=2.0
    interaction=[(1,2,U),(1,3,U),(1,4,U),(2,3,U),(2,4,U),(3,4,U)]
    filename_base=replace("$(data_dir)/U_$(U)_n_$(nασ)",","=>"_"," "=>"","["=>"","]"=>"")
    G12ασ=[i[1] for i in loadAsSpinOrbital(filename_base,"G12",N_spin_orbital)]
    βασ=[reshape(i,2) for i in loadAsSpinOrbital(filename_base,"beta",N_spin_orbital)]
    αασ_check=[reshape(i,2) for i in loadAsSpinOrbital(filename_base,"alpha",N_spin_orbital)]
    K0_check=loadData("$(filename_base)_Etotal_Eloc_Ek.dat")[end]
    x= reshape(loadData("$(filename_base)_x.dat"),:)
    w= reshape(loadData("$(filename_base)_w.dat"),:)
    # we try some different value
    G12ασ=[0.44 for _ in 1:N_spin_orbital]
    w02=cal_w02(cal_neffασ(nασ,G12ασ))
    VΓη,ηToIdx=__global_VΓη_ηToIdx__[N_spin_orbital]
    x=pinv(VΓη)*(w.^2-w02)
    # x+=rand(11)*0.1
    e_fn=gene_spline_band("./es_inf.dat")
    eασ=[gene_ϵs(e_fn,nασ[i]) for _ in 1:N_spin_orbital]
    αασ,Δασ,Aασ,K0,∂K∂Δασ,∂K∂Aασ=cal_momemtum_part(nασ,G12ασ,x,βασ,eασ)
    gradient((G12ασ,x,Aασ)->cal_energy_for_diff(G12ασ,x,Aασ,nασ,∂K∂Δασ,∂K∂Aασ,interaction),G12ασ,x,Aασ)
## we assume they are reasonable
"""
function cal_gradient(G12ασ,x,βασ,nασ,eασ,interaction)
    # we can now check this derivatives from the previous numerical minimization
    # we first load the parameters to check
    # it seems the gradients are reasonably close to zero
    # there are some problem for large U, check the reason
    αασ,Δασ,Aασ,K0,∂K∂Δασ,∂K∂Aασ=cal_momemtum_part(nασ,G12ασ,x,βασ,eασ)
    ∂E∂G12ασ,∂E∂x,∂E∂Aασ= gradient((G12ασ,x,Aασ)->cal_energy_for_diff(G12ασ,x,Aασ,nασ,∂K∂Δασ,∂K∂Aασ,interaction),G12ασ,x,Aασ)
    ∂E∂G12ασ,∂E∂x,∂E∂Aασ,αασ,Δασ,Aασ,K0,∂K∂Δασ,∂K∂Aασ
end


"""
we change the interaction
U=2.5
U=3.0
U=4.0
# the basic procedure
interaction=[(1,2,U),(1,3,U),(1,4,U),(2,3,U),(2,4,U),(3,4,U)]
G12ασ,x,βασ,∂E∂G12ασ,∂E∂x,∂E∂Aασ=solve_N3(G12ασ,x,βασ,nασ,eασ,interaction)
obs=cal_obs(G12ασ,x,βασ,nασ,eασ,interaction)
saveObs(obs,data_dir,tag,∂E∂G12ασ,∂E∂x,∂E∂Aασ)
"""
function solve_N3(G12ασ,x,βασ,nασ,eασ,interaction; λG=0.01,λx=0.001,λβ=0.1,N_iter=100)
    local ∂E∂G12ασ,∂E∂x,∂E∂Aασ
    for i in 1:N_iter
        ∂E∂G12ασ,∂E∂x,∂E∂Aασ,αασ,Δασ,Aασ,K0,∂K∂Δασ,∂K∂Aασ=cal_gradient(G12ασ,x,βασ,nασ,eασ,interaction)
        G12ασ=G12ασ-λG*∂E∂G12ασ
        x=x-λx*∂E∂x
        βασ=βασ-λβ*∂E∂Aασ
        x=constraint_para_x(x,G12ασ,nασ)
        print("i $(i) βασ $(βασ)\n i $(i) x $(x) \n")
    end
    G12ασ,x,βασ,∂E∂G12ασ,∂E∂x,∂E∂Aασ
end

# finnally, we calcuate the observables
"""
this will serve as a complete illustration of the calculation
E,Eloc,Ek,local_info,αασ,βασ,Δασ,Aασ,Slocασ,G12ασ,x,w,nk,nασ,eασ=cal_obs(G12ασ,x,βασ,nασ,eασ,interaction)
obs=cal_obs(G12ασ,x,βασ,nασ,eασ,interaction)
"""
function cal_obs(G12ασ,x,βασ,nασ,eασ,interaction)
    N_spin_orbital=length(nασ)
    w=cal_w(x,nασ,G12ασ)
    pmatwασ,g12matwSασ=cal_p_g12_mat(G12ασ)
    g12ασ=[expt(w,cal_Xmatfull(pmatwασ,g12matwSασ,i)) for i in 1:N_spin_orbital]
    Slocασ=cal_Slocασ(nασ,G12ασ,g12ασ)
    Δασ=cal_Δασ(g12ασ,Slocασ)
    Aασ=[]
    αασ=[]
    nk=[]
    Ek=0                        # kinecit energy
    for i in 1:N_spin_orbital
        Aασ_,αασ_,nk_=cal_Aασ_αασ_nk_(nασ[i],Δασ[i],βασ[i],eασ[i])
        Ekασ_=mean(nk_[1].*eασ[i][1])*nασ[i]+mean(nk_[2].*eασ[i][2])*(1-nασ[i])
        Ek+=Ekασ_
        push!(Aασ,Aασ_)
        push!(αασ,αασ_)
        push!(nk,nk_)
    end
    g33matwασ=[cal_g33_mat_(cal_Gfull(nασ[i],G12ασ[i],g12ασ[i],Aασ[i],Slocασ[i])) for i in 1:N_spin_orbital]
    # (idx1,idx2,coefficient,expt_nidx1_nidx2)
    local_info=[[idx1,idx2,coefficient,expt(w,cal_Xmatfull(pmatwασ,g33matwασ,idx1,idx2))] for (idx1,idx2,coefficient) in interaction ]
    Eloc=sum([coefficient*nn for (idx1,idx2,coefficient,nn) in local_info ])
    E=Ek+Eloc
    E,Eloc,Ek,local_info,αασ,βασ,Δασ,Aασ,Slocασ,G12ασ,x,w,nk,nασ,eασ
end

"""
mainly restrict x that w2 are postive
"""
function constraint_para_x(x,G12ασ,nασ)
    N_spin_orbital=length(nασ)
    w02=cal_w02(cal_neffασ(nασ,G12ασ))
    VΓη,ηToIdx=__global_VΓη_ηToIdx__[N_spin_orbital]
    k=VΓη*x
    k0,knorm=cal_k0_knorm(k)
    kmaxnorm=cal_maxnorm(w02,k0)*(1-1E-8)
    if(knorm>=kmaxnorm)
        x=kmaxnorm/knorm*x
    end    
    x
end


function loadAsSpinOrbital(filename_base,qauntity_name,N_spin_orbital)
    [loadData("$(filename_base)_$(qauntity_name)_spin_orb_$(i).dat") for i in 1:N_spin_orbital]
end

function saveAsSpinOrbital(val,filename_base,qauntity_name)
    N_spin_orbital=length(val)
    for i in 1:N_spin_orbital
        saveData(val[i],"$(filename_base)_$(qauntity_name)_spin_orb_$(i).dat")
    end    
end

"""
data_dir="./two_band_inf_diff/"
mkdir(data_dir)
tag="U_$(U)"
U
also, with gradient
"""
function saveObs(obs,data_dir,tag,∂E∂G12ασ,∂E∂x,∂E∂Aασ)
    E,Eloc,Ek,local_info,αασ,βασ,Δασ,Aασ,Slocασ,G12ασ,x,w,nk,nασ,eασ=obs
    filename_base=replace("$(data_dir)/$(tag)_n_$(nασ)",","=>"_"," "=>"","["=>"","]"=>"")
    saveData([E,Eloc,Ek],"$(filename_base)_Etotal_Eloc_Ek.dat")
    saveData(local_info,"$(filename_base)_local_info.dat")
    saveAsSpinOrbital(αασ,filename_base,"alpha")
    saveAsSpinOrbital(βασ,filename_base,"beta")
    saveAsSpinOrbital(Δασ,filename_base,"Delta")
    saveAsSpinOrbital(Aασ,filename_base,"A")
    saveAsSpinOrbital(nασ,filename_base,"n")
    saveAsSpinOrbital(G12ασ,filename_base,"G12")
    saveAsSpinOrbital(Slocασ,filename_base,"S")
    saveData(x,"$(filename_base)_x.dat")
    saveData(w,"$(filename_base)_w.dat")
    saveData(∂E∂G12ασ,"$(filename_base)_dEdG12.dat")
    saveData(∂E∂x,"$(filename_base)_dEdx.dat")
    saveData(∂E∂Aασ,"$(filename_base)_dEdA.dat")
end
