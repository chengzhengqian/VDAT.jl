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

"""
test for second order
"""
function cal_w2(x,nασ,G12ασ)
    N_spin_orbital=length(nασ)
    neffασ=cal_neffασ(nασ,G12ασ)
    w02=cal_w02(neffασ)
    VΓη,ηToIdx=cal_VΓη_ηToIdx(N_spin_orbital)
    k=VΓη*x
    # k0,knorm=cal_k0_knorm(k)
    # kmax=cal_maxnorm(w02,k0)
    # knormscaled=regulate_knorm(knorm,kmax)
    # kscaled=k0*knormscaled
    k+w02
end

"""
there are some ambiguity in defining the metric 
this is not correct
"""
function cal_metric(x,nασ,G12ασ,regulate_knorm)
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
    transpose(VΓη)*diagm(1.0 ./w)*VΓη
end


# we now implement the self-consistency
"""
for a given Δασ, and x, (of course,nασ, and the given reglate_knorm),
we found the corresponding G12ασ
regulate_knorm=regulate_knorm_1
Δασ=[0.1,0.1,0.1,0.1]
G12ασ=[0.4,0.4,0.4,0.4]
G12ασ_new=[0.4,0.4,0.4,0.4]
we start with some inital guess

Δασ=cal_Δασ(g12ασ,Slocασ,nασ)
Δασ=Δασ+rand(4)*0.0005
for i in 1:10
    global  G12ασ,G12ασ_new                               
    G12ασ=cal_G12ασ_fix_point(x,nασ,G12ασ,Δασ,regulate_knorm)
    print("$(G12ασ)\n")
end

"""
function cal_G12ασ_fix_point(x,nασ,G12ασ,Δασ,regulate_knorm)
    N_spin_orbital=length(nασ)
    w=cal_w(x,nασ,G12ασ,regulate_knorm)
    pmatwασ,g12matwSασ=cal_p_g12_mat(G12ασ)
    g12ασ=[expt(w,cal_Xmatfull(pmatwασ,g12matwSασ,i)) for i in 1:N_spin_orbital]
    Slocασ=cal_Slocασ(nασ,G12ασ,g12ασ)
    rασ=cal_rασ(Slocασ)
    cασ=cal_cασ(nασ,Δασ,rασ)
    G12ασ_new=cal_G12ασ(nασ,Δασ,rασ,cασ)
end

# the fixed point method is not that fast, we can try use the gradient method, i.e using ∂C/∂G12

#
"""
compute constraint
gradient(G12ασ->cal_C(x,nασ,G12ασ,Δασ,regulate_knorm)[1],G12ασ)
"""
function cal_C(x,nασ,G12ασ,Δασ,regulate_knorm)
    N_spin_orbital=length(nασ)
    w=cal_w(x,nασ,G12ασ,regulate_knorm)
    pmatwασ,g12matwSασ=cal_p_g12_mat(G12ασ)
    g12ασ=[expt(w,cal_Xmatfull(pmatwασ,g12matwSασ,i)) for i in 1:N_spin_orbital]
    Slocασ=cal_Slocασ(nασ,G12ασ,g12ασ)
    [g12ασ[i]*Slocασ[i][2]-Δασ[i]*Slocασ[i][1]   for i in 1:N_spin_orbital]
end

"""
∂C∂x,∂C∂G,∂C∂Δ=cal_∂C∂x_∂C∂G_∂C∂Δ(x,nασ,G12ασ,Δασ,regulate_knorm)
"""
function cal_∂C∂x_∂C∂G_∂C∂Δ(x,nασ,G12ασ,Δασ,regulate_knorm)
    N_spin_orbital=length(nασ)
    ∂C∂x=zeros(N_spin_orbital,length(x))
    ∂C∂G=zeros(N_spin_orbital,length(G12ασ))
    ∂C∂Δ=zeros(N_spin_orbital,length(Δασ))
    # i=1
    for i in 1:N_spin_orbital
        ∂C∂x_,∂C∂G_,∂C∂Δ_=gradient((x,G12ασ,Δασ)->cal_C(x,nασ,G12ασ,Δασ,regulate_knorm)[i],x,G12ασ,Δασ)
        ∂C∂x[i,:]=∂C∂x_
        ∂C∂G[i,:]=∂C∂G_
        ∂C∂Δ[i,:]=∂C∂Δ_
    end
    ∂C∂x,∂C∂G,∂C∂Δ
end

"""
cal_C(x,nασ,G12ασ,Δασ,regulate_knorm)
cal_C(x_1,nασ,G12ασ_1,Δασ_1,regulate_knorm)
G12ασ,∂G∂x,∂G∂Δ,C=cal_G12ασ_∂G∂x_∂G∂Δ_C(x,nασ,G12ασ,Δασ,regulate_knorm;N_iter=10)
# we now check the derivaties
x_1=x+rand(length(x))*1e-5
Δασ_1=Δασ+rand(length(Δασ))*1e-5
G12ασ_1,∂G∂x_1,∂G∂Δ_1,C_1=cal_G12ασ_∂G∂x_∂G∂Δ_C(x_1,nασ,G12ασ,Δασ_1,regulate_knorm;N_iter=10)
δG=G12ασ_1-G12ασ
δG_check=∂G∂x*(x_1-x)+∂G∂Δ*(Δασ_1-Δασ)
δG_check./δG
"""
function cal_G12ασ_∂G∂x_∂G∂Δ_C(x,nασ,G12ασ,Δασ,regulate_knorm;N_iter=10)
    local ∂C∂x,∂C∂G,∂C∂Δ,∂G∂C,C
    for i in 1:N_iter
        C=cal_C(x,nασ,G12ασ,Δασ,regulate_knorm)
        ∂C∂x,∂C∂G,∂C∂Δ=cal_∂C∂x_∂C∂G_∂C∂Δ(x,nασ,G12ασ,Δασ,regulate_knorm)
        ∂G∂C=inv(∂C∂G)
        ΔG=-∂G∂C*C
        G12ασ=G12ασ+ΔG
    end
    G12ασ,-∂G∂C*∂C∂x,-∂G∂C*∂C∂Δ,C
end

"""
We may want to use fixed point method to update G12ασ
"""
function cal_∂G∂x_∂G∂Δ(x,nασ,G12ασ,Δασ,regulate_knorm)
    ∂C∂x,∂C∂G,∂C∂Δ=cal_∂C∂x_∂C∂G_∂C∂Δ(x,nασ,G12ασ,Δασ,regulate_knorm)
    ∂G∂C=inv(∂C∂G)
    -∂G∂C*∂C∂x,-∂G∂C*∂C∂Δ
end


function cal_Gfull(nασ_,G12ασ_,g12ασ_,Aασ_below_,Aασ_above_,Slocασ_)
    gextraασ_=cal_glocReducedInA(Aασ_below_,Aασ_above_,Slocασ_[1],Slocασ_[2])
    Gfullασ_=cal_g0fullIngG12(nασ_,G12ασ_,g12ασ_,gextraασ_[1],gextraασ_[2],gextraασ_[3],gextraασ_[4])
end


"""
compute local energy
# we first compute 
Δασ,Aασ_below,Aασ_above,K=cal_Δ_Abelow_Aabove_K(Δαασ,βασ,eασ,nασ) 
# we should mix them smoothly
Δασ=cal_Δασ(g12ασ,Slocασ,nασ)
G12ασ=solve_G12ασ_gradient(x,nασ,G12ασ,Δασ,regulate_knorm;N_iter=10)
U=2.0
interaction=[(1,2,U),(1,3,U),(1,4,U),(2,3,U),(2,4,U),(3,4,U)]

"""
function cal_Eloc(x,nασ,G12ασ,Δασ,Aασ_below,Aασ_above,interaction,regulate_knorm)
    N_spin_orbital=length(nασ)
    w=cal_w(x,nασ,G12ασ,regulate_knorm)
    pmatwασ,g12matwSασ=cal_p_g12_mat(G12ασ)
    g12ασ=[expt(w,cal_Xmatfull(pmatwασ,g12matwSασ,i)) for i in 1:N_spin_orbital]
    Slocασ=cal_Slocασ(nασ,G12ασ,g12ασ)
    # check for C, should be zero
    # [g12ασ[i]*Slocασ[i][2]-Δασ[i]*Slocασ[i][1]   for i in 1:N_spin_orbital]
    g33matwασ=[cal_g33_mat_(cal_Gfull(nασ[i],G12ασ[i],g12ασ[i],Aασ_below[i],Aασ_above[i],Slocασ[i])) for i in 1:N_spin_orbital]
    Eloc=sum([coefficient*expt(w,cal_Xmatfull(pmatwασ,g33matwασ,idx1,idx2)) for (idx1,idx2,coefficient) in interaction ])
end

"""
compute second order
or we use natural gradient
regulate_knorm=regulate_knorm_2
"""
function cal_second_order(x,nασ,G12ασ,Δασ,Aασ_below,Aασ_above,interaction,regulate_knorm)
    # N_spin_orbital=length(nασ)
    w2=cal_w2(x,nασ,G12ασ)
    # pmatwασ,g12matwSασ=cal_p_g12_mat(G12ασ)
    # g12ασ=[expt(w,cal_Xmatfull(pmatwασ,g12matwSασ,i)) for i in 1:N_spin_orbital]
    # Slocασ=cal_Slocασ(nασ,G12ασ,g12ασ)
    # g33matwασ=[cal_g33_mat_(cal_Gfull(nασ[i],G12ασ[i],g12ασ[i],Aασ_below[i],Aασ_above[i],Slocασ[i])) for i in 1:N_spin_orbital]
    # # Eloc_M=sum([coefficient*cal_Xmatfull(pmatwασ,g33matwασ,idx1,idx
    # 2) for (idx1,idx2,coefficient) in interaction ])
    # Eloc_check=expt(w,Eloc_M)
    # Eloc_check_2=sum([coefficient*expt(w,cal_Xmatfull(pmatwασ,g33matwασ,idx1,idx2)) for (idx1,idx2,coefficient) in interaction ])
    # Zygote.hessian(x->dot(x,x),[1.0,2.0])
    # d2Edx2=Zygote.hessian(x->expt(cal_w(x,nασ,G12ασ,regulate_knorm),Eloc_M),x)
    d2Sdx2_1=Zygote.hessian(x->relative_entropy(cal_w2(x,nασ,G12ασ),w2),x)
    d2Sdx2_2=Zygote.hessian(x->entropy(cal_w2(x,nασ,G12ασ)),x)
    sum(abs.(d2Sdx2_1- d2Sdx2_2))
    eigen(d2Sdx2)

end

function relative_entropy(v,v0)
    dot(v,log.(v./v0))
end



function cal_∂Eloc∂G12_∂Eloc∂x_∂Eloc∂Abelow_∂Eloc∂Aabove(x,nασ,G12ασ,Δασ,Aασ_below,Aασ_above,interaction,regulate_knorm)
    ∂Eloc∂G12,∂Eloc∂x,∂Eloc∂Abelow,∂Eloc∂Aabove=gradient((G12ασ,x,Aασ_below,Aασ_above)->cal_Eloc(x,nασ,G12ασ,Δασ,Aασ_below,Aασ_above,interaction,regulate_knorm),G12ασ,x,Aασ_below,Aασ_above)
end

"""
the true gradient of Eloc to x,Δ,A
G12ασ,C,dEdx,dEdΔ,dEdAbelow,dEdAabove=cal_G12ασ_C_dEdx_dEdΔ_dEdAbelow_dEdAabove(x,nασ,G12ασ,Δασ,Aασ_below,Aασ_above,interaction,regulate_knorm)
"""
function cal_G12ασ_C_dEdx_dEdΔ_dEdAbelow_dEdAabove(x,nασ,G12ασ,Δασ,Aασ_below,Aασ_above,interaction,regulate_knorm)
    G12ασ,∂G∂x,∂G∂Δ,C=cal_G12ασ_∂G∂x_∂G∂Δ_C(x,nασ,G12ασ,Δασ,regulate_knorm;N_iter=3)
    ∂Eloc∂G12,∂Eloc∂x,∂Eloc∂Abelow,∂Eloc∂Aabove=cal_∂Eloc∂G12_∂Eloc∂x_∂Eloc∂Abelow_∂Eloc∂Aabove(x,nασ,G12ασ,Δασ,Aασ_below,Aασ_above,interaction,regulate_knorm)
    dEdx=∂Eloc∂x+reshape(reshape(∂Eloc∂G12,1,:)*∂G∂x,:)
    dEdΔ=reshape(reshape(∂Eloc∂G12,1,:)*∂G∂Δ,:)
    dEdAbelow=∂Eloc∂Abelow
    dEdAabove=∂Eloc∂Aabove
    G12ασ,C,dEdx,dEdΔ,dEdAbelow,dEdAabove
end


"""
remove the udpate of G12, compute local energy derivatives
"""
function cal_dEdx_dEdΔ_dEdAbelow_dEdAabove(x,nασ,G12ασ,Δασ,Aασ_below,Aασ_above,interaction,regulate_knorm)
    ∂G∂x,∂G∂Δ=cal_∂G∂x_∂G∂Δ(x,nασ,G12ασ,Δασ,regulate_knorm)
    ∂Eloc∂G12,∂Eloc∂x,∂Eloc∂Abelow,∂Eloc∂Aabove=cal_∂Eloc∂G12_∂Eloc∂x_∂Eloc∂Abelow_∂Eloc∂Aabove(x,nασ,G12ασ,Δασ,Aασ_below,Aασ_above,interaction,regulate_knorm)
    dEdx=∂Eloc∂x+reshape(reshape(∂Eloc∂G12,1,:)*∂G∂x,:)
    dEdΔ=reshape(reshape(∂Eloc∂G12,1,:)*∂G∂Δ,:)
    dEdAbelow=∂Eloc∂Abelow
    dEdAabove=∂Eloc∂Aabove
    dEdx,dEdΔ,dEdAbelow,dEdAabove
end

