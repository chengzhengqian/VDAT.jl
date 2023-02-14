# previous, we focus on fix the density, here, we provide the interface of using w (para the local projector)
# include("./vdat_scheme1_v1.jl")


# we first list the test parameters, we need to move them to the file for execuation later.
# we focus on the degnerate case with J/U=0
# we assume the user should fixed the density already
# notice we need to ensure the density constraint is satified, which should be trivial in this case.
#  half-filling first
# N_spin_orbital=10
# n_=0.5
# n_target=[n_]
# symmetry=collect.([1:10])
# e_fn=gene_spline_band("./es_inf.dat")
# update__global__Γασ__(N_spin_orbital)
# nασ=extend_with_symmetry(n_target,symmetry,N_spin_orbital)
# eασ=cal_eασ(e_fn,nασ,symmetry)

## we use effective energy, as exponential form is easy to control

"""
for degenerate case, we compute effective energy
N_spin_orbital=10
Γ=cal_Γασ(10,N_spin_orbital)
(N_orbital+1) terms,
but from the normalization, we can assume E_eff_for_N_orbital=0
w_para=[E_eff_for_0,...,E_eff_for_(N_orbital-1)]
So we have (N_orbital) terms, independent terms
And we have
E_for_M=E_for_(N_spin_orbital-M)
cal_E_eff_degenerate_half([1.0],[0,0],2)
cal_E_eff_degenerate_half([1.0],[1,0],2)
cal_E_eff_degenerate_half([1.0],[0,1],2)
cal_E_eff_degenerate_half([1.0],[1,1],2)
"""
function cal_E_eff_degenerate_half(w_para,Γ,N_spin_orbital)
    N_orbital=trunc(Int,N_spin_orbital/2)
    N_particle=sum(Γ)
    if(N_particle<N_orbital)
        return w_para[N_particle+1]
    elseif(N_particle>N_orbital)
        return w_para[(N_spin_orbital-N_particle)+1]
    else
        return 0.0
    end    
end

"""
w_para=[0.5,0.4,0.3,0.2,0.1]
w=cal_w_degenerate_half(w_para,N_spin_orbital)
Notice
w_para=[E_eff_for_0,...,E_eff_for_(N_orbital-1)]
So we have (N_orbital) terms, independent terms
"""
function cal_w_degenerate_half(w_para,N_spin_orbital)
    E_eff_for_w=[cal_E_eff_degenerate_half(w_para,cal_Γασ(i,N_spin_orbital),N_spin_orbital) for i in 1:2^N_spin_orbital]
    E_eff_min=minimum(E_eff_for_w)
    w2=exp.(-(E_eff_for_w.-E_eff_min))
    w2=w2./sum(w2)
    w=sqrt.(w2)
end

function gene_interaction_degenerate_J_0(U,N_spin_orbital)
    [(1,2,U*N_spin_orbital*(N_spin_orbital-1)/2)]
end


"""
we directly pass w.
Check vdat_scheme_1_local.jl to see the linear way to restrict density
# we could mimic the flood, in ratio to solve the equation?, how to fixed the density, there should be a better way.
# use hypercube to control density, this should be very general
G12_para=[0.4]
β_below_para=β_above_para=[2.0]
Notice w_para assumes to provide the density we want. So  eασ[i]  is assumed to correspond given density
cal_w_func=cal_w_degenerate_half
U=1.0
interaction=gene_interaction_degenerate(U,0,N_spin_orbital)
# for only U, this is simplier, we can test these two are same

interaction_test=gene_interaction_degenerate_J_0(U,N_spin_orbital)

@time result1=cal_energy_with_symmetry_using_w(G12_para,β_below_para,β_above_para,w_para,cal_w_func,eασ,interaction,symmetry,N_spin_orbital)
@time result2=cal_energy_with_symmetry_using_w(G12_para,β_below_para,β_above_para,w_para,cal_w_func,eασ,interaction_test,symmetry,N_spin_orbital)
result1[1]-result2[1]           # checked, they are same
"""
function cal_energy_with_symmetry_using_w(G12_para,β_below_para,β_above_para,w_para,cal_w_func,eασ,interaction,symmetry,N_spin_orbital;cutoff_n=1e-4,cutoff_Δ=1e-5,cutoff_Sloc=1e-5)
    N_orbital=trunc(Int,N_spin_orbital/2)
    G12ασ=cal_G(G12_para,symmetry,N_spin_orbital)
    βασ=cal_β(β_below_para,β_above_para,symmetry,N_spin_orbital)
    pmatwασ,g12matwSασ=cal_p_g12_mat(G12ασ)
    g11matwSασ=cal_g11_mat(G12ασ)
    w=cal_w_func(w_para,N_spin_orbital)
    g12ασ=cal_Xασ(w,pmatwασ,g12matwSασ,symmetry)
    # we will output the density so we can double check whether cal_w_func and w_para is reasonable
    nασ=cal_Xασ(w,pmatwασ,g11matwSασ,symmetry)
    Slocασ=cal_Slocασ(nασ,G12ασ,g12ασ)
    Slocασ=restrict_Slocασ(Slocασ,cutoff=cutoff_Sloc)
    Δασ=cal_Δασ(g12ασ,Slocασ,nασ)
    Δασ=restrict_Δασ(Δασ,nασ; cutoff=cutoff_Δ)
    Aασ_below,Aασ_above,αασ,nk=[Array{Any}(undef,N_spin_orbital) for _ in 1:4]
    K0=0
    # use the symmetry to simplify the calcuation
    for term in symmetry
        i=term[1] 
        Aασ_below_,Aασ_above_,Kbelow_,Kabove_,αασ_,nk_=cal_Abelow_Aabove_Kbelow_Kabove_αασ_nk_(nασ[i],Δασ[i],βασ[i],eασ[i])
        K0ασ_=Kbelow_+Kabove_
        K0+=K0ασ_*length(term)
        for idx in term
            Aασ_below[idx]=Aασ_below_
            Aασ_above[idx]=Aασ_above_
            αασ[idx]=αασ_
            nk[idx]=nk_
        end        
    end    
    g33matwασ=[cal_g33_mat_(cal_Gfull(nασ[i],G12ασ[i],g12ασ[i],Aασ_below[i],Aασ_above[i],Slocασ[i])) for i in 1:N_spin_orbital]
    nn=[expt(w,cal_Xmatfull(pmatwασ,g33matwασ,idx1,idx2)) for (idx1,idx2,coefficient) in interaction]
    Eloc=sum([interaction[i][3]*nn[i]   for i in 1:length(interaction)])
    Eloc+K0,Eloc,K0,nασ,nn,αασ,βασ,eασ,Slocασ,Δασ,Aασ_below,Aασ_above,G12ασ,w,nk
end

