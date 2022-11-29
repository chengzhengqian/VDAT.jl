# this contains some function to parametrize G,β,w

function cal_G(G)
    [G for _ in 1:N_spin_orbital]
end
function cal_β(β)
    [[β,β] for _ in 1:N_spin_orbital]
end

"""
for SU_N, half-filling case
"""
function p_eff(U_eff,N)
    exp(-U_eff*N*(N-N_spin_orbital)/2)
end


para=[0.4,2.0,1.0]
function para_half_SU_N(para)
    G12ασ=cal_G(para[1])
    βασ=cal_β(para[2])
    U_eff=para[3]
    w2=[p_eff(U_eff,sum(cal_Γασ(i,10))) for i in 1:2^10]
    w2=w2/sum(w2)
    w=sqrt.(w2)
    G12ασ,w,βασ
end

function gene_interaction(U,J,N_orbital)
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
Etotal,Eloc,K0,nασ,nn,αασ,βασ,eασ,Slocασ,Δασ,Aασ_below,Aασ_above=cal_energy_projective(G12ασ,w,βασ,e_fn,interaction)
result=cal_energy_projective(G12ασ,w,βασ,e_fn,interaction)
para=[0.43,2.0,1.0]
result=cal_energy_projective(para_half_SU_N(para)...,e_fn,interaction)


"""
function cal_energy_half_SU_N(para)
    result=cal_energy_projective(para_half_SU_N(para)...,e_fn,interaction)
    print("call with $(para), get energy $(result[1])\n")
    result[1]
end

data_dir="/home/chengzhengqian/Documents/research/vdat/five_band_result"
mkdir(data_dir)
data_dir="/home/chengzhengqian/Documents/research/vdat/five_band_result/SU_N_half"
mkdir(data_dir)
U=2.0
for U in 3.0:30.0
    global para,interaction
    J=0.0
    interaction=gene_interaction(U,J,5)
    cal_energy_half_SU_N(para)
    res=optimize(cal_energy_half_SU_N,para,Optim.Options(iterations=100))
    para=res.minimizer
    result=cal_energy_projective(para_half_SU_N(para)...,e_fn,interaction)
    file_base="U_$(U)_J_$(J)"
    save_result(result,interaction,data_dir,file_base)
end




"""
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

# now, we check the case with chemcial potential
"""
with parameters constrol the total energy
Δμ for N/2 filling
"""
function p_eff(U_eff,Δμ_eff,N)
    exp(-U_eff*N*(N-N_spin_orbital)/2+Δμ_eff*(N-N_spin_orbital/2))
end

para=[0.4,2.0,1.0,0.0]
# we need to first compute energy , and subtract the normalization to get the probabliity
function para_doped_SU_N(para)
    G12ασ=cal_G(para[1])
    βασ=cal_β(para[2])
    U_eff=para[3]
    Δμ_eff=para[4]
    w2=[p_eff(U_eff,Δμ_eff,sum(cal_Γασ(i,10))) for i in 1:2^10]
    w2=w2/sum(w2)
    w=sqrt.(w2)
    G12ασ,w,βασ
end

"""
one also need to specify μ
μ=9*U/2
para=[0.75,32,21,17]
N_spin_orbital=10
G12ασ,w,βασ=para_doped_SU_N(para)
problem parameters
para=[0.6374835684796012, 36.62049407300821, 16.393631118306082, 82.9907878904388]
para=[0.49166501647418115, 26.14175317461315, 50.93943795018737, 87.70697327669006]
"""
function cal_energy_doped_SU_N(para)
    try
        result=cal_energy_projective(para_doped_SU_N(para)...,e_fn,interaction)
        nασ=result[4]
        E_chemical=-μ*sum(nασ)
        print("call with $(para), and μ $(μ), get energy $(result[1]), density $(nασ[1])\n")
        result[1]+E_chemical
    catch
        error("call with $(para)\n")
    end    
end


data_dir_doped="/home/chengzhengqian/Documents/research/vdat/five_band_result/SU_N_doped"
mkdir(data_dir_doped)
U=2.0
para=[0.43,2.7,0.77,0.0]
U=8.0
para=[0.46,8.0,2.24,0.0]
U=32.0
para=[0.49,32,10,0.0]
# Δμ=21
U=32.0
para=[0.49,32,21,17.0]
J=0.0
interaction=gene_interaction(U,J,5)
μ=9*U/2
res=optimize(cal_energy_doped_SU_N,para,Optim.Options(iterations=100))
# U=2.0
# for Δμ in 0.0:0.1:1.5
# for Δμ in 1.6:0.1:3.0
# for Δμ in 3.0:0.5:5.0
# for Δμ in 5.5:0.5:10.0
# U=8.0
# for Δμ in 0.0:0.5:5.0
# for Δμ in 0.0:0.5:100.0
for Δμ in 71.5:0.5:100.0
    global para,interaction,μ,paras
    J=0.0
    μ=9*U/2+Δμ
    interaction=gene_interaction(U,J,5)
    res=optimize(cal_energy_doped_SU_N,para,Optim.Options(iterations=100))
    para=res.minimizer
    result=cal_energy_projective(para_doped_SU_N(para)...,e_fn,interaction)
    file_base="U_$(U)_J_$(J)_Δmu_$(Δμ)"
    save_result(result,interaction,data_dir_doped,file_base)
end


# we now process that data
using Glob

files=glob("U_$(U)_J_$(J)_Δmu*_density.dat","$(data_dir_doped)")
file_=files[1]
parse(Float64,"1.0")

data_dir_doped_processed="/home/chengzhengqian/Documents/research/vdat/five_band_result/SU_N_doped_processed"
mkdir(data_dir_doped_processed)
U=2.0
U=8.0
for U in [2.0,8.0,32]
    files=glob("U_$(U)_J_$(J)_Δmu*_density.dat","$(data_dir_doped)")
    saveData([[parse(Float64,split(files[i],"_")[end-1]),loadData(files[i])[1]] for i in 1:length(files)],"$(data_dir_doped_processed)/dmu_n_U_$(U)_J_$(J).dat")
end

# we can also generate HF result
# n=0.5
function cal_energy_HF(U,n)
    eασ= gene_ϵs(e_fn,n)
    K=mean(eασ[1])*n*10
    Eloc=U*45*n*(n-1)
    K+Eloc
end

U=32.0
n=0.6
dn=1e-5
for U in [2.0,8.0,32.0]
    saveData([[(cal_energy_HF(U,n+dn)-cal_energy_HF(U,n-dn))/(2*dn)/10,n] for n in linspace(0.5,0.99,100)],"$(data_dir_doped_processed)/dmu_n_U_$(U)_J_$(J)_HF.dat")
end

function cal_Δμ_n_atomic(U)
    N=5:9
    Δμc_fn=N->U*(2*N-9)/2
    Δμcs=[0.0,Δμc_fn.(N)...]
    result=[]
    for i in 1:length(N)
        push!(result,(Δμcs[i],N[i]/10))
        push!(result,(Δμcs[i+1],(N[i])/10))
    end
    result
end

for U in [2.0,8.0,32.0]
    saveData(cal_Δμ_n_atomic(U),"$(data_dir_doped_processed)/dmu_n_U_$(U)_J_$(J)_atomic.dat")
end


# now, we start to examine the case with J
# we first examine the half-filling.
# we first compute the effective energy, then compute the probablity, (we can substract a constant to make the procedure more stable)
"""
p_eff=exp(-E_eff), but we can freely substract a constant, i.e the lowest energy,
Δμ=0 corresponds the half-filling case
"""
function cal_E_eff(U1,U2,U3,Δμ,Γ)
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
    μ=(U1+4*U2+4*U3)/2
    N=sum(Γ)
    # exp(-s+μ*N)
    s-(μ+Δμ)*N
end

"""
para=[0.45,2.0,1.0,0.5,0.3]
"""
function para_half_SU_N_with_J(para)
    G12ασ=cal_G(para[1])
    βασ=cal_β(para[2])
    U1,U2,U3=para[3:end]
    E_eff_for_w=[cal_E_eff(U1,U2,U3,0,cal_Γασ(i,10)) for i in 1:2^10]
    E_eff_min=minimum(E_eff_for_w)
    w2=exp.(-(E_eff_for_w.-E_eff_min))
    w2=w2./sum(w2)
    w=sqrt.(w2)
    G12ασ,w,βασ
end


function cal_energy_half_SU_N_with_J(para)
    result=cal_energy_projective(para_half_SU_N_with_J(para)...,e_fn,interaction)
    print("call with $(para), get energy $(result[1])\n")
    result[1]
end

data_dir_half_with_J="/home/chengzhengqian/Documents/research/vdat/five_band_result/SU_N_half_with_J"
mkdir(data_dir_half_with_J)

J_U=0.05
para=[0.44,2.48,0.89,0.73,0.64]
U=2.0
# for J_U, the transition is around U =5 ~ 6,
U=5.0
U=5.4
para=loadData("$(data_dir_half_with_J)/U_$(U)_J_$(U*J_U)_para.dat")
# for U in 2.0:30.0
# for U in 5.1:0.1:5.9
for U in 5.41:0.01:5.49
    global para,interaction
    J=U*J_U
    interaction=gene_interaction(U,J,5)
    res=optimize(cal_energy_half_SU_N_with_J,para,Optim.Options(iterations=200))
    para=res.minimizer
    result=cal_energy_projective(para_half_SU_N_with_J(para)...,e_fn,interaction)
    file_base="U_$(U)_J_$(J)"
    save_result(result,interaction,data_dir_half_with_J,file_base)
    saveData(para,"$(data_dir_half_with_J)/$(file_base)_para.dat")
end

    
# we first process the density-density interaction
U=2.0
J=U*J_U
data_dir_half_with_J_processed="$(data_dir_half_with_J)_processed"
mkdir(data_dir_half_with_J_processed)
Us=sort([2.0:30.0...,5.1:0.1:5.9...,5.41:0.01:5.49...])
saveData([(U,loadData("$(data_dir_half_with_J)/U_$(U)_J_$(U*J_U)_nn.dat")[[1,6,end]]...) for U in Us ],"$(data_dir_half_with_J_processed)/nn_vs_U_J_U_$(J_U).dat")


saveData([(U,loadData("$(data_dir_half_with_J)/U_$(U)_J_$(U*J_U)_G12.dat")[1]) for U in Us ],"$(data_dir_half_with_J_processed)/G12_vs_U_J_U_$(J_U).dat")


saveData([(U,loadData("$(data_dir_half_with_J)/U_$(U)_J_$(U*J_U)_para.dat")...) for U in Us ],"$(data_dir_half_with_J_processed)/para_vs_U_J_U_$(J_U).dat")

# now, we consider the doped case. Thus, we add one more parameter to control the density.

"""
para=[0.45,2.0,1.0,0.5,0.3,0.0]
"""
function para_doped_SU_N_with_J(para)
    G12ασ=cal_G(para[1])
    βασ=cal_β(para[2])
    U1,U2,U3,μeff=para[3:end]
    E_eff_for_w=[cal_E_eff(U1,U2,U3,μeff,cal_Γασ(i,10)) for i in 1:2^10]
    E_eff_min=minimum(E_eff_for_w)
    w2=exp.(-(E_eff_for_w.-E_eff_min))
    w2=w2./sum(w2)
    w=sqrt.(w2)
    G12ασ,w,βασ
end


function cal_energy_doped_SU_N_with_J(para)
    result=cal_energy_projective(para_doped_SU_N_with_J(para)...,e_fn,interaction)
    nασ=result[4]
    E_chemical=-μ*sum(nασ)
    print("call with $(para), get energy $(result[1]), density $(mean(nασ[1]))\n")
    result[1]+E_chemical
end

J_U=0.05
U=2.0
U=6.0
U=12.0
J=U*J_U
para_half=loadData("$(data_dir_half_with_J)/U_$(U)_J_$(U*J_U)_para.dat")

para=[para_half...,0.0]

data_dir_doped_with_J="/home/chengzhengqian/Documents/research/vdat/five_band_result/SU_N_doped_with_J"
mkdir(data_dir_doped_with_J)
para=loadData("$(data_dir_doped_with_J)/U_$(U)_J_$(U*J_U)_Δmu_16.0_para.dat")
para=loadData("$(data_dir_doped_with_J)/U_$(U)_J_$(U*J_U)_Δmu_24.0_para.dat")
para=loadData("$(data_dir_doped_with_J)/U_$(U)_J_$(U*J_U)_Δmu_26.0_para.dat")
# we first test Δμ=0, to check the expression for the chemcial potential
# for Δμ in 0.1:0.1:2.0
# for Δμ in 2.5:0.5:8.0
# for Δμ in 0.5:0.5:8.0
# for Δμ in 16.0:2.0:40.0
# for Δμ in 16.1:0.1:17.9
# for Δμ in 2.0:2.0:30.0
for Δμ in 26.0:1.0:50.0
    global para,interaction,μ
    μ=9*U/2-10*J+Δμ
    interaction=gene_interaction(U,J,5)
    res=optimize(cal_energy_doped_SU_N_with_J,para,Optim.Options(iterations=200))
    para=res.minimizer
    result=cal_energy_projective(para_doped_SU_N_with_J(para)...,e_fn,interaction)
    file_base="U_$(U)_J_$(J)_Δmu_$(Δμ)"
    save_result(result,interaction,data_dir_doped_with_J,file_base)
    saveData(para,"$(data_dir_doped_with_J)/$(file_base)_para.dat")
end

# check for U=6.0,  for doped case to unpolarized state
U=6.0
J=J_U*U
para=loadData("$(data_dir_doped_with_J)/U_$(U)_J_$(U*J_U)_Δmu_16.1_para.dat")
U=12.0
J=J_U*U
para=loadData("$(data_dir_doped_with_J)/U_$(U)_J_$(U*J_U)_Δmu_26.0_para.dat")
data_dir_doped_with_J_rev="/home/chengzhengqian/Documents/research/vdat/five_band_result/SU_N_doped_with_J_reversed"
mkdir(data_dir_doped_with_J_rev)
# indeed, we see that the metal phase extend to small chemical potential
# for Δμ in 16.0:-0.5:0.5
for Δμ in 25.0:-1.0:1.0
    global para,interaction,μ
    μ=9*U/2-10*J+Δμ
    interaction=gene_interaction(U,J,5)
    res=optimize(cal_energy_doped_SU_N_with_J,para,Optim.Options(iterations=200))
    para=res.minimizer
    result=cal_energy_projective(para_doped_SU_N_with_J(para)...,e_fn,interaction)
    file_base="U_$(U)_J_$(J)_Δmu_$(Δμ)"
    save_result(result,interaction,data_dir_doped_with_J_rev,file_base)
    saveData(para,"$(data_dir_doped_with_J_rev)/$(file_base)_para.dat")
end


data_dir_doped_with_J_processed="/home/chengzhengqian/Documents/research/vdat/five_band_result/SU_N_doped_with_J_processed"
mkdir(data_dir_doped_with_J_processed)
data_dir_doped_with_J_processed_rev="/home/chengzhengqian/Documents/research/vdat/five_band_result/SU_N_doped_with_J_reversed_processed"
mkdir(data_dir_doped_with_J_processed_rev)

# for U in [2.0,6.0]
for U in [12.0]
    files=glob("U_$(U)_J_$(U*J_U)_Δmu*_density.dat","$(data_dir_doped_with_J)")
    saveData([[parse(Float64,split(files[i],"_")[end-1]),loadData(files[i])[1]] for i in 1:length(files)],"$(data_dir_doped_with_J_processed)/dmu_n_U_$(U)_J_U_$(J_U).dat")
end


# for U in [6.0]
for U in [12.0]
    files=glob("U_$(U)_J_$(U*J_U)_Δmu*_density.dat","$(data_dir_doped_with_J_rev)")
    saveData([[parse(Float64,split(files[i],"_")[end-1]),loadData(files[i])[1]] for i in 1:length(files)],"$(data_dir_doped_with_J_processed_rev)/dmu_n_U_$(U)_J_U_$(J_U).dat")
end

function cal_energy_HF(U,J,n)
    eασ= gene_ϵs(e_fn,n)
    K=mean(eασ[1])*n*10
    Eloc=(45*U-100*J)*n*(n-1)
    K+Eloc
end
    

# for U in [2.0,6.0]
for U in [12.0]
    J=U*J_U
    saveData([[(cal_energy_HF(U,J,n+dn)-cal_energy_HF(U,J,n-dn))/(2*dn)/10,n] for n in linspace(0.5,0.99,100)],"$(data_dir_doped_with_J_processed)/dmu_n_U_$(U)_J_U_$(J_U)_HF.dat")
end

function Eloc_atomic(U,J,N)
    if(N==5)
        (U-3*J)*10
    else
        # assuming N>5
        U*(N-5)+(U-2*J)*4*(N-5)+(U-3*J)*(10+0.5*(N-5)*(N-6))
    end
end


function Δμc_with_J(U,J,N)
    μ=(9/2*U-10*J)
    Eloc_atomic(U,J,N+1)-Eloc_atomic(U,J,N)-μ
end
U=6.0
J=J_U*U
function cal_Δμ_n_atomic(U,J)
    N=5:9
    Δμcs=[0.0,Δμc_with_J.(U,J,N)...]
    result=[]
    for i in 1:length(N)
        push!(result,(Δμcs[i],N[i]/10))
        push!(result,(Δμcs[i+1],(N[i])/10))
    end
    result
end

# for U in [2.0,6.0]
for U in [12.0]
    saveData(cal_Δμ_n_atomic(U,J_U*U),"$(data_dir_doped_with_J_processed)/dmu_n_U_$(U)_J_U_$(J_U)_atomic.dat")
end

