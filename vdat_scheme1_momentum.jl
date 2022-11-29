# solve the momentum part


"""
X represent either below or above,
"""
function cal_nX(esX,αX,βX,weightX)
    nkX=[ cal_nk(αX,βX,esX_) for esX_ in esX]
    nX=mean(nkX)*weightX
end


function solve_αX_from_nX(esX,nX,βX,weightX)
    # n_mean should be within 0.0 and 1.0
    # or nX∈[0,weightX]
    # nX=constraint_nX(nX,weightX)
    n_mean=nX/weightX
    e_mean=mean(esX)
    αX_guess=cal_alpha(e_mean,βX,n_mean)
    αX=find_zero(αX->cal_nX(esX,αX,βX,weightX)-nX,αX_guess,Order0())
end

"""
this compute all necessary quantities for a given region
"""
function cal_nX_AX_KX_nkX(esX,αX,βX,weightX)
    nkX=[ cal_nk(αX,βX,esX_) for esX_ in esX]
    nX=mean(nkX)*weightX
    AX=mean(sqrt.(nkX.*(1.0 .- nkX)))*weightX
    KX=mean(esX.*nkX)*weightX
    nX,AX,KX,nkX
end

"""
to break Aασ to Aασ_below and Aασ_above, so we can use the same form as the differential case. See ...v3.jl for details
nασ_,Δασ_,βασ_,eασ_=nασ[i],Δασ[i],βασ[i],eασ[i]
"""
function cal_Abelow_Aabove_Kbelow_Kabove_αασ_nk_(nασ_,Δασ_,βασ_,eασ_)
    # try
    βbelow,βabove=βασ_
    ebelow,eabove=eασ_
    nbelow=nασ_-Δασ_
    nabove=Δασ_
    αbelow=solve_αX_from_nX(ebelow,nbelow,βbelow,nασ_)
    αabove=solve_αX_from_nX(eabove,nabove,βabove,1-nασ_)
    nbelowcheck,Abelow,Kbelow,nkbelow=cal_nX_AX_KX_nkX(ebelow,αbelow,βbelow,nασ_)
    nabovecheck,Aabove,Kabove,nkabove=cal_nX_AX_KX_nkX(eabove,αabove,βabove,1-nασ_)
    Abelow,Aabove,Kbelow,Kabove,[αbelow,αabove],[nkbelow,nkabove]
    # catch
    #     error("cal A get $([nασ_,Δασ_,βασ_])\n")
    # end    
end


"""
the derivatives in terms of α, β, for a given region X
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
compute Gfull (irreducible form)
"""
function cal_Gfull(nασ_,G12ασ_,g12ασ_,Aασ_below_,Aασ_above_,Slocασ_)
    # try
    gextraασ_=cal_glocReducedInA(Aασ_below_,Aασ_above_,Slocασ_[1],Slocασ_[2])
    Gfullασ_=cal_g0fullIngG12(nασ_,G12ασ_,g12ασ_,gextraασ_[1],gextraασ_[2],gextraασ_[3],gextraασ_[4])
    # catch
    #     error("get $([nασ_,G12ασ_,g12ασ_,Aασ_below_,Aασ_above_,Slocασ_])")
    # end    
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
we use the new comversion to regulate x,with l_x and l_G12ασ (interpolating between G12ασ_min and G12ασ_max)
momentum_derivaties=cal_momentum_derivatives(nασ,l_G12ασ,l_x,x,βασ,eασ,G12ασ_min)
"""
function cal_momentum_derivatives(nασ,l_G12ασ,l_x,x,βασ,eασ,G12ασ_min)
    N_spin_orbital=length(nασ)
    G12ασ_max=[0.5 for i in 1:N_spin_orbital]
    G12ασ=(G12ασ_min.*l_G12ασ)+G12ασ_max.*(1.0 .- l_G12ασ)
    w=cal_w_restrict(x,l_x,nασ,G12ασ)
    # G12ασ=regulate_G12.(G12ασ)
    # w=cal_w(x,nασ,G12ασ,regulate_knorm)
    pmatwασ,g12matwSασ=cal_p_g12_mat(G12ασ)
    g12ασ=[expt(w,cal_Xmatfull(pmatwασ,g12matwSασ,i)) for i in 1:N_spin_orbital]
    Slocασ=cal_Slocασ(nασ,G12ασ,g12ασ)
    Δασ=cal_Δασ(g12ασ,Slocασ,nασ)
    Δασ=restrict_Δασ(Δασ,nασ)
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

