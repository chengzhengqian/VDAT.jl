# we put the functions about the momentum part in this file
"""
we first generate some preliminary data
N_spin_orbital=4
e_fn=gene_spline_band("./es_inf.dat")
nασ=[0.5,0.5,0.5,0.5]
es=gene_ϵs(e_fn,nασ[1])
eασ=[ es for _ in 1:N_spin_orbital]
# we test for below
esX=es[1]
αX=2.0
βX=1.0
weightX=nασ[1]
cal_nX(esX,αX,βX,weightX)
"""
function cal_nX(esX,αX,βX,weightX)
    nkX=[ cal_nk(αX,βX,esX_) for esX_ in esX]
    nX=mean(nkX)*weightX
end

"""
for a given region
"""
function cal_nX_AX_KX(esX,αX,βX,weightX)
    nkX=[ cal_nk(αX,βX,esX_) for esX_ in esX]
    nX=mean(nkX)*weightX
    AX=mean(sqrt.(nkX.*(1.0 .- nkX)))*weightX
    KX=mean(esX.*nkX)*weightX
    nX,AX,KX
end



"""
we have 
Δα_=α_below-α_above
βασ_=[β_below,β_above]
for a given spin orbital
nασ_=nασ[1]
Δα_=4.0
βασ_=[1.0,1.0]
eασ_=es
α_below=2.0
"""
function cal_nασ_(α_below,Δα_,βασ_,eασ_,nασ_)
    α_above=α_below-Δα_
    β_below,β_above=βασ_
    n_below=cal_nX(eασ_[1],α_below,β_below,nασ_)
    n_above=cal_nX(eασ_[2],α_above,β_above,1-nασ_)
    n_below+n_above
end

"""
Δασ_,Aασ_below_,Aασ_above_,K_=cal_Δασ_A_below_A_above_K_(α_below,Δα_,βασ_,eασ_,nασ_)
"""
function cal_Δασ_A_below_A_above_K_(α_below,Δα_,βασ_,eασ_,nασ_)
    α_above=α_below-Δα_
    β_below,β_above=βασ_
    n_below,A_below,K_below=cal_nX_AX_KX(eασ_[1],α_below,β_below,nασ_)
    n_above,A_above,K_above=cal_nX_AX_KX(eασ_[2],α_above,β_above,1-nασ_)
    Δασ_=n_above
    Δασ_,A_below,A_above,K_below+K_above
end



"""
nασ_=0.5
α_below=solve_α_below_(Δα_,βασ_,eασ_,nασ_)
"""
function solve_α_below_(Δα_,βασ_,eασ_,nασ_)
    α_below=find_zero(α_below->cal_nασ_(α_below,Δα_,βασ_,eασ_,nασ_)-nασ_,Δα_/2)
end

"""
Unlike previous convension, we store them seperately 
Δασ_,Aασ_below_,Aασ_above_,K_=cal_Δ_Abelow_Aabove_K_(Δα_,βασ_,eασ_,nασ_)
# now, check the derivatives relation.
i.e, ∂K/∂Δ=-Δα=α_above-α_below, ∂K/∂AX=βX, (another one is ∂K/∂n=α_below); We first check the first three (X=below, or above)
Δα_0=2.0
βασ_0=[1.0,2.0]
nασ_=0.4
Δασ_0,Aασ_below_0,Aασ_above_0,K_0=cal_Δ_Abelow_Aabove_K_(Δα_0,βασ_0,eασ_,nασ_)
Δα_1=Δα_0+rand(1)[1]*1e-4
βασ_1=βασ_0+rand(2)*1e-4
Δασ_1,Aασ_below_1,Aασ_above_1,K_1=cal_Δ_Abelow_Aabove_K_(Δα_1,βασ_1,eασ_,nασ_)
δK=K_1-K_0
∂K∂Δ=-Δα_0
∂K∂A=βασ_0
δK_check=∂K∂Δ*(Δασ_1-Δασ_0)+dot(∂K∂A,[Aασ_below_1-Aασ_below_0,Aασ_above_1-Aασ_above_0])
δK_check/δK                     # checked
# now, we check the forth formula
nασ_1=nασ_+rand(1)[1]*1e-4
Δασ_0,Aασ_below_0,Aασ_above_0,K_0=cal_Δ_Abelow_Aabove_K_(Δα_0,βασ_0,eασ_,nασ_)
Δασ_1,Aασ_below_1,Aασ_above_1,K_1=cal_Δ_Abelow_Aabove_K_(Δα_1,βασ_1,eασ_,nασ_)
δK=K_1-K_0
# this assume the weight is unchanged
∂K∂n=solve_α_below_(Δα_0,βασ_0,eασ_,nασ_)
δK_check=∂K∂Δ*(Δασ_1-Δασ_0)+dot(∂K∂A,[Aασ_below_1-Aασ_below_0,Aασ_above_1-Aασ_above_0])+∂K∂n*(nασ_-nασ_)
δK/δK_check
# there are something about the  ∂K∂n, somehow, α_below is not right, !! check this later (I think the reason is the weightX, we will explore it later)
"""
function cal_Δ_Abelow_Aabove_K_(Δα_,βασ_,eασ_,nασ_)
    α_below=solve_α_below_(Δα_,βασ_,eασ_,nασ_)
    Δασ_,Aασ_below_,Aασ_above_,K_=cal_Δασ_A_below_A_above_K_(α_below,Δα_,βασ_,eασ_,nασ_)
end

"""
we then get the vector version
Δαασ=[2.0 for _ in 1:N_spin_orbital]
βασ=[[1.0,1.0] for _ in  1:N_spin_orbital]
nασ=[0.5 for _ in 1:N_spin_orbital]
Δασ,Aασ_below,Aασ_above,K=cal_Δ_Abelow_Aabove_K(Δαασ,βασ,eασ,nασ)
"""
function cal_Δ_Abelow_Aabove_K(Δαασ,βασ,eασ,nασ)
    N_spin_orbital=length(Δαασ)
    Δασ=[]
    Aασ_below=[]
    Aασ_above=[]
    K=0
    for i in 1:N_spin_orbital
        Δασ_,Aασ_below_,Aασ_above_,K_=cal_Δ_Abelow_Aabove_K_(Δαασ[i],βασ[i],eασ[i],nασ[i])
        push!(Δασ,Δασ_)
        push!(Aασ_below,Aασ_below_)
        push!(Aασ_above,Aασ_above_)
        K+=K_
    end
    Δασ,Aασ_below,Aασ_above,K
end
