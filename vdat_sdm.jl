# this implements the single particle density matrix formalism of SCDA

using LinearAlgebra
using Statistics
using Combinatorics
using Roots
using Zygote

include("./include_gene_code.jl")
# processing the band structure
include("./load_band.jl")
include("./utils.jl")

n# we first process momentum part, we write them to a seperate file
# the main function is
# Δασ,Aασ_below,Aασ_above,K=cal_Δ_Abelow_Aabove_K(Δαασ,βασ,eασ,nασ) 
# which computes the change transfer and change flutuation for a given set of parameters
include("./vdat_sdm_momentum.jl")
include("./vdat_sdm_local.jl")
# then we need to deal with local part, and self-consistency of G12
# load initial parameters

# set band structure
N_spin_orbital=4
e_fn=gene_spline_band("./es_inf.dat")
nασ=[0.5,0.5,0.5,0.5]
es=gene_ϵs(e_fn,nασ[1])
eασ=[ es for _ in 1:N_spin_orbital]

# we use a linear way to regulate it, so it agrees with prevoius results
regulate_knorm=regulate_knorm_2

step_x=1e-3
mix_Δ=1e-1
mix_A=1e-1

function update(step_x,mix_Δ,mix_A,N_iter=10)
    global x,Δασ,Aασ_below,Aασ_above,G12ασ,βασ,Δαασ
    for i in 1:N_iter
        G12ασ,C,dEdx,dEdΔ,dEdAbelow,dEdAabove=cal_G12ασ_C_dEdx_dEdΔ_dEdAbelow_dEdAabove(x,nασ,G12ασ,Δασ,Aασ_below,Aασ_above,interaction,regulate_knorm)
        # update the parameters
        Δαασ=dEdΔ
        βασ=[[-dEdAbelow[i],-dEdAabove[i]]  for i in 1:N_spin_orbital]
        x=x-step_x*dEdx
        Δασ_new,Aασ_below_new,Aασ_above_new,K=cal_Δ_Abelow_Aabove_K(Δαασ,βασ,eασ,nασ)
        Δασ=Δασ_new*mix_Δ+Δασ*(1-mix_Δ)
        Aασ_below=Aασ_below_new*mix_A+Aασ_below*(1-mix_A)
        Aασ_above=Aασ_above_new*mix_A+Aασ_above*(1-mix_A)
        print("dEdx:$(sum(abs.(dEdx))),δΔ:$(sum(abs.(Δασ_new-Δασ)))\n")
    end    
end

U=2.0
interaction=[(1,2,U),(1,3,U),(1,4,U),(2,3,U),(2,4,U),(3,4,U)]
G12ασ,x,Δαασ,βασ=load_para_two_band_half(U)
Δασ,Aασ_below,Aασ_above,K=cal_Δ_Abelow_Aabove_K(Δαασ,βασ,eασ,nασ)

update(1.0*1e-3,1*1e-1,1*1e-1,1000)
βασ
Δαασ
x
U=6.0
interaction=[(1,2,U),(1,3,U),(1,4,U),(2,3,U),(2,4,U),(3,4,U)]

# we try a new scheme, where we use fixed point to update

N_spin_orbital=4
e_fn=gene_spline_band("./es_inf.dat")
nασ=[0.5,0.5,0.5,0.5]
es=gene_ϵs(e_fn,nασ[1])
eασ=[ es for _ in 1:N_spin_orbital]

# we use a linear way to regulate it, so it agrees with prevoius results
regulate_knorm=regulate_knorm_2
U=2.0
interaction=[(1,2,U),(1,3,U),(1,4,U),(2,3,U),(2,4,U),(3,4,U)]
G12ασ,x,Δαασ,βασ=load_para_two_band_half(U)

# momentum part
Δασ,Aασ_below,Aασ_above,K=cal_Δ_Abelow_Aabove_K(Δαασ,βασ,eασ,nασ)
# update G12                          
# update derivatives
# we may need to restrict G12
function update_scheme_1(N_iter=10)
    global x,Δασ,Aασ_below,Aασ_above,G12ασ,βασ,Δαασ
    for i in 1:N_iter
        dEdx,dEdΔ,dEdAbelow,dEdAabove=cal_dEdx_dEdΔ_dEdAbelow_dEdAabove(x,nασ,G12ασ,Δασ,Aασ_below,Aασ_above,interaction,regulate_knorm)
        Δαασ=dEdΔ
        # assume same
        βασ=[[-(dEdAbelow[i]+dEdAabove[i])*0.5,-(dEdAbelow[i]+dEdAabove[i])*0.5]  for i in 1:N_spin_orbital]
        # compute second order derivatives
        d2Sdx2=cal_second_order(x,nασ,G12ασ,Δασ,Aασ_below,Aασ_above,interaction,regulate_knorm)
        # eigen(d2Sdx2)
        # id=Matrix(I,size(d2Sdx2))
        # M=cal_metric(x,nασ,G12ασ,regulate_knorm)
        M_inv=inv(d2Sdx2+id)
        x=x-step_x*M_inv*dEdx
        # x=x-step_x*dEdx
        # x=x-step_x*inv(d2Sdx2)*dEdx
        Δασ_new,Aασ_below_new,Aασ_above_new,K=cal_Δ_Abelow_Aabove_K(Δαασ,βασ,eασ,nασ)
        Δασ=Δασ_new*mix_Δ+Δασ*(1-mix_Δ)
        Aασ_below=Aασ_below_new*mix_A+Aασ_below*(1-mix_A)
        Aασ_above=Aασ_above_new*mix_A+Aασ_above*(1-mix_A)
        G12ασ_new=cal_G12ασ_fix_point(x,nασ,G12ασ,Δασ,regulate_knorm)
        G12ασ_new=min.(G12ασ_new,0.5)
        G12ασ=G12ασ_new*mix_G+G12ασ*(1-mix_G)
        print("interaction:$(interaction[1])\n")
        print("x:$(x[1]),G12,$(G12ασ[1]),β:$(βασ[1])\n")
        print("dEdx:$(sum(abs.(dEdx))),δΔ:$(sum(abs.(Δασ_new-Δασ)))\n")
    end    
end

U=2.0
G12ασ,x,Δαασ,βασ=load_para_two_band_half(U)
Δασ,Aασ_below,Aασ_above,K=cal_Δ_Abelow_Aabove_K(Δαασ,βασ,eασ,nασ)
# linear form
regulate_knorm=regulate_knorm_2
# non lienar form
regulate_knorm=regulate_knorm_1 
# non linear form, one global maximum
regulate_knorm=regulate_knorm_3

U=10.0
U=2.0
U=4.0
U=8.0
interaction=[(1,2,U),(1,3,U),(1,4,U),(2,3,U),(2,4,U),(3,4,U)]
interaction=[(1,2,U+1.0),(1,3,U),(1,4,U),(2,3,U),(2,4,U),(3,4,U+1.0)]

# alternative way to compute the metric
VΓη,ηToIdx=cal_VΓη_ηToIdx(N_spin_orbital)
M=transpose(VΓη)*VΓη
M_inv=inv(M)

step_x=0
step_x=2*1e-3
step_x=1*1e-4
step_x=3*1e-4
step_x=6*1e-4
step_x=4*1e-3
step_x=4*1e-2
step_x=8*1e-2
step_x=16*1e-2
step_x=32*1e-2
step_x=64*1e-2
step_x=128*1e-2
step_x=256*1e-2
step_x=500*1e-2
step_x=1000*1e-2
step_x=2000*1e-2
step_x=5000*1e-2
mix_Δ=0.5
mix_A=0.5
mix_G=0.5


step_x=100
step_x=2*1e-4
step_x=4*1e-3
step_x=4*1e-2
step_x=8*1e-2
step_x=2*1e-1
step_x=3*1e-1
step_x=1000
# maybe one should restrict G12
update_scheme_1(100)
