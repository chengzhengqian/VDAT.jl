# in this version, we focus on the symmetry and the parametrization
# it is adopted from vdat_scheme1_5_band_para.jl
# and provides some interface so we can run on cluster easily
# so we don't need to compute derivatives in this version.
using Pkg
Pkg.add("Optim")

using LinearAlgebra
using Statistics
using Combinatorics
using Roots
using Zygote
using Optim

include("./include_gene_code.jl")
# processing the band structure
# we should improve the efficiency of this part, but maybe use symmetry is enought
include("./load_band.jl")
# we can move other auxillary functions to this files
include("./utils.jl")

# local part
# most of the part are dedicated to find the vector space of w that consistent with the given density. In the new approach, we just parametrize w directly, and use the momentum density distribution to fix the constraint.
include("./vdat_scheme1_local.jl")

# there are also the momentum derivatives, which we don't need in this approach
include("./vdat_scheme1_momentum.jl")

# the key part of version to the use symmetry to reduce the computation cost of tne energy, and directly minize it.
# first, user can provide the funciton to compute G12ασ,βασ from

# we first define the band, some basic information
# we make sure there should be no global parameters
e_fn=gene_spline_band("./es_inf.dat")
N_spin_orbital=10

function extend_with_symmetry(para,symmetry,N_spin_oribtal)
    result=zeros(N_spin_orbital)
    for (idx,term) in enumerate(symmetry)
        result[term].=para[idx]
    end
    result
end

# 
symmetry=collect.([1:5,6:10])
# G_para same as the length of symmetry
"""
G_para=[0.4,0.44]
cal_G(G_para,symmetry,N_spin_orbital)
"""
function cal_G(G_para,symmetry,N_spin_orbital)
    extend_with_symmetry(G_para,symmetry,N_spin_orbital)
end

"""
β_para_below=[3.0,4.0]
β_para_above=[3.1,4.1]
"""
function cal_β(β_para_below,β_para_above,symmetry,N_spin_orbital)
    β_para_below_ext=extend_with_symmetry(β_para_below,symmetry,N_spin_orbital)
    β_para_above_ext=extend_with_symmetry(β_para_above,symmetry,N_spin_orbital)
    [[β_para_below_ext[i],β_para_above_ext[i]] for i in 1:N_spin_orbital]
end

"""
assume the band structure are same
idx=1
"""
function cal_eασ(e_fn,nασ,symmetry)
    N_spin_orbital=length(nασ)
    eασ=Vector{Any}(undef,N_spin_orbital)
    for (idx,term) in enumerate(symmetry)
        es=gene_ϵs(e_fn,nασ[term[1]])
        for i in term
            eασ[i]=es
        end
    end
    eασ
    # eασ=[ gene_ϵs(e_fn,nασ[i]) for _ in 1:N_spin_orbital]
end

# the next step is that we need to parametrize the w
# we first generate the effective energy, and use p∼exp(-E), notice we are free to subtract
"""
α is the orbital idx (start form 1) and σ is the spin indx (1,2) for spin up and own
"""
function get_idx(α,σ)
    (α-1)*2+σ
end

"""
# U1, U2, U3 are effective U for different kinds of density density interaction
N_orbital=trunc(Int,N_spin_orbital/2)
Γ=[0,1,…],  is a atomic configuration
# this allow us to parametrize the degenerate case
when Δμ=0, the generate w will give the half-filling case
cal_E_eff(U1,U2,U3,Δμ,[0,0,0,0],4)


"""
function cal_E_eff(U1,U2,U3,Δμ,Γ,N_spin_orbital)
    N_orbital=trunc(Int,N_spin_orbital/2)
    s=0
    for orb in 1:(N_orbital)
        s+=U1*Γ[get_idx(orb,1)]*Γ[get_idx(orb,2)]
    end    
    for orb1 in 1:(N_orbital-1)
        for orb2 in (orb1+1):(N_orbital)
            s+=U2*Γ[get_idx(orb1,1)]*Γ[get_idx(orb2,2)]
            s+=U2*Γ[get_idx(orb1,2)]*Γ[get_idx(orb2,1)]
        end
    end
    for orb1 in 1:(N_orbital-1)
        for orb2 in (orb1+1):(N_orbital)
            s+=U3*Γ[get_idx(orb1,1)]*Γ[get_idx(orb2,1)]
            s+=U3*Γ[get_idx(orb1,2)]*Γ[get_idx(orb2,2)]
        end
    end
    # μ=(U1+4*U2+4*U3)/2
    μ=(U1+(N_orbital-1)*U2+(N_orbital-1)*U3)/2
    # exp(-s+μ*N)
    s-(μ+Δμ)*sum(Γ)
end

"""
This follows from cal_E_eff
Remeber to update the global dictionary first
cal_Γασ(2,10)
w=cal_w(1.0,0.8,0.7,0,10)
U1=U2=U3=1.0
Δμ=0.0
"""
function cal_w(U1,U2,U3,Δμ,N_spin_orbital)
    E_eff_for_w=[cal_E_eff(U1,U2,U3,Δμ,cal_Γασ(i,N_spin_orbital),N_spin_orbital) for i in 1:2^N_spin_orbital]
    E_eff_min=minimum(E_eff_for_w)
    w2=exp.(-(E_eff_for_w.-E_eff_min))
    w2=w2./sum(w2)
    w=sqrt.(w2)
end


"""
generate interaction, without spin flip term
interaction=gene_interaction(1.0,0.1,2)
interaction=gene_interaction(1.0,0.1,4)
"""
function gene_interaction(U,J,N_spin_orbital)
    N_orbital=trunc(Int,N_spin_orbital/2)
    interaction=[]
    # introband
    for orb in 1:N_orbital
        push!(interaction,(get_idx(orb,1),get_idx(orb,2),U))
    end
    # interband, opposite spin
    for orb1 in 1:(N_orbital-1)
        for orb2 in (orb1+1):(N_orbital)
            push!(interaction,(get_idx(orb1,1),get_idx(orb2,2),U-2*J))
            push!(interaction,(get_idx(orb1,2),get_idx(orb2,1),U-2*J))
        end
    end
    # interband, same spin
    for orb1 in 1:(N_orbital-1)
        for orb2 in (orb1+1):(N_orbital)
            push!(interaction,(get_idx(orb1,1),get_idx(orb2,1),U-3*J))
            push!(interaction,(get_idx(orb1,2),get_idx(orb2,2),U-3*J))
        end
    end
    interaction
end

"""
we can assume the symmetry (i.e, the degeneracy of the band)
symmetry=collect.([1:4,5:10])
# same size as symmetry
G12_para=[0.4,0.41]
β_below_para=[3.2,4.2]
β_above_para=[3.0,4.0]
w=cal_w(1.0,0.8,0.7,0.3,N_spin_orbital)
w should be consistent with symmetry
cutoff_n=1e-4,cutoff_Δ=1e04
to control the density and charge fluctuation
interaction (we can incoorpate symmetry into the form)
"""
function cal_energy_with_symmetry(G12_para,β_below_para,β_above_para,w,e_fn,interaction,symmetry,N_spin_orbital;cutoff_n=1e-4,cutoff_Δ=1e-5,cutoff_Sloc=1e-5)
    G12ασ=cal_G(G12_para,symmetry,N_spin_orbital)
    βασ=cal_β(β_below_para,β_above_para,symmetry,N_spin_orbital)
    pmatwασ,g12matwSασ=cal_p_g12_mat(G12ασ)
    g11matwSασ=cal_g11_mat(G12ασ)
    # g12ασ=cal_Xασ(w,pmatwασ,g12matwSασ)
    # nασ=cal_Xασ(w,pmatwασ,g11matwSασ)
    g12ασ=cal_Xασ(w,pmatwασ,g12matwSασ,symmetry)
    nασ=cal_Xασ(w,pmatwασ,g11matwSασ,symmetry)
    nασ=restrict_nασ(nασ;cutoff=cutoff_n)
    # this approach is accurate, but not that efficient
    eασ=cal_eασ(e_fn,nασ,symmetry)
    # @time eασ=[ gene_ϵs(e_fn,nασ[i]) for i in 1:N_spin_orbital]
    Slocασ=cal_Slocασ(nασ,G12ασ,g12ασ)
    Slocασ=restrict_Slocασ(Slocασ,cutoff=cutoff_Sloc)
    # Slocασ=[max.(Slocασ[i],1e-5) for i in 1:N_spin_orbital]
    # print("check G12 $(G12ασ)\n ")
    Δασ=cal_Δασ(g12ασ,Slocασ,nασ)
    Δασ=restrict_Δασ(Δασ,nασ; cutoff=cutoff_Δ)
    Aασ_below,Aασ_above,αασ,nk=[Array{Any}(undef,N_spin_orbital) for _ in 1:4]
    K0=0
    # i=1
    # for i in 1:N_spin_orbital
    #     Aασ_below_,Aασ_above_,Kbelow_,Kabove_,αασ_,nk_=cal_Abelow_Aabove_Kbelow_Kabove_αασ_nk_(nασ[i],Δασ[i],βασ[i],eασ[i])
    #     # K0ασ_=mean(nk_[1].*eασ[i][1])*nασ[i]+mean(nk_[2].*eασ[i][2])*(1-nασ[i])
    #     K0ασ_=Kbelow_+Kabove_
    #     K0+=K0ασ_
    #     push!(Aασ_below,Aασ_below_)
    #     push!(Aασ_above,Aασ_above_)
    #     push!(αασ,αασ_)
    #     push!(nk,nk_)
    # end
    # we update with symmetry
    # term=symmetry[1]
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
    # Eloc=sum([coefficient*expt(w,cal_Xmatfull(pmatwασ,g33matwασ,idx1,idx2)) for (idx1,idx2,coefficient) in interaction ])
    Eloc+K0,Eloc,K0,nασ,nn,αασ,βασ,eασ,Slocασ,Δασ,Aασ_below,Aασ_above,G12ασ,w,nk
end

"""
corresponding function to save the result get form cal_energy...
save result
save_result(result,interaction,data_dir,file_base)
"""
function save_result(result,interaction,data_dir,file_base)
    filename_tag="$(data_dir)/$(file_base)"
    Etotal,Eloc,K0,nασ,nn,αασ,βασ,eασ,Slocασ,Δασ,Aασ_below,Aασ_above,G12ασ,w,nk=result
    saveData([Etotal,Eloc,K0],"$(filename_tag)_Etotal_Eloc_Ek.dat")
    saveData(nασ,"$(filename_tag)_density.dat")
    saveData(nn,"$(filename_tag)_nn.dat")
    saveData(interaction,"$(filename_tag)_interaction.dat")
    saveData(αασ,"$(filename_tag)_alpha.dat")
    saveData(βασ,"$(filename_tag)_beta.dat")
    saveData(G12ασ,"$(filename_tag)_G12.dat")
    saveData(Slocασ,"$(filename_tag)_Sloc.dat")
    saveData(Δασ,"$(filename_tag)_Delta.dat")
    saveData(Aασ_below,"$(filename_tag)_A_below.dat")
    saveData(Aασ_above,"$(filename_tag)_A_above.dat")
    saveAsSpinOrbital(nk,filename_tag,"nk")
    saveAsSpinOrbital(eασ,filename_tag,"ek")
end

function saveAsSpinOrbital(result,filename_base,qauntity_name)
    N_spin_orbital=length(result)
    for i in 1:N_spin_orbital
        saveData(result[i],"$(filename_base)_$(qauntity_name)_spin_orb_$(i).dat")
    end    
end




# update para, for a given
# we only need to udpate Γ
# function set_Γασ_VΓη_ηToIdx(N_spin_orbital)
#     update__global__Γασ__(N_spin_orbital)
#     update__global_VΓη_ηToIdx__(N_spin_orbital)
# end

"""
we can simplify the number of calculation when all orbital are same.
(We manually encode the information for each case)
Assuming N_oribtal>1
U=1.0
J=0.1
gene_interaction_degenerate(U,J,N_spin_orbital)
"""
function gene_interaction_degenerate(U,J,N_spin_orbital)
    N_orbital=trunc(Int,N_spin_orbital/2)
    interaction=Vector{Any}(undef,3)
    # introband
    # # for orb in 1:N_orbital
    #     push!(interaction,(get_idx(orb,1),get_idx(orb,2),U))
    # end
    interaction[1]=(get_idx(1,1),get_idx(1,2),U*N_orbital)
    # interband, opposite spin
    # for orb1 in 1:(N_orbital-1)
    #     for orb2 in (orb1+1):(N_orbital)
    #         push!(interaction,(get_idx(orb1,1),get_idx(orb2,2),U-2*J))
    #         push!(interaction,(get_idx(orb1,2),get_idx(orb2,1),U-2*J))
    #     end
    # end
    interaction[2]=(get_idx(1,1),get_idx(2,2),(U-2*J)*(N_orbital-1)*N_orbital)
    # interband, same spin
    # for orb1 in 1:(N_orbital-1)
    #     for orb2 in (orb1+1):(N_orbital)
    #         push!(interaction,(get_idx(orb1,1),get_idx(orb2,1),U-3*J))
    #         push!(interaction,(get_idx(orb1,2),get_idx(orb2,2),U-3*J))
    #     end
    # end
    interaction[3]=(get_idx(1,1),get_idx(2,1),(U-3*J)*(N_orbital-1)*N_orbital)
    interaction
end


function cal_energy_half_SU_N(para)
    w=cal_w(para[3],para[3],para[3],0,N_spin_orbital)
    result=cal_energy_with_symmetry(para[1:1],para[2:2],para[2:2],w,e_fn,interaction,symmetry,N_spin_orbital)
    print("call with $(para), get energy $(result[1])\n")
    result[1]
end
# we move this to specific files
# we now need to dressup some code run the minimization and input interface
