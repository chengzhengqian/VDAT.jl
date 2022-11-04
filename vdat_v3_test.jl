# we test the v3


U=2.0
momentum_info=cal_momentum_derivatives(nασ,G12ασ,x,βασ,eασ,regulate_knorm_1,regulate_G12_1,regulate_Δασ_1)
interaction=[(1,2,U),(1,3,U),(1,4,U),(2,3,U),(2,4,U),(3,4,U)]
E=cal_energy(G12ασ,x,βασ,momentum_info,interaction,regulate_knorm_1,regulate_G12_1,regulate_Δασ_1)
E_check,_,_=cal_energy_direct(G12ασ,x,βασ,nασ,eασ,interaction,regulate_knorm_1,regulate_G12_1,regulate_Δασ_1)
momentum_info=cal_momentum_derivatives(nασ,G12ασ,x,βασ,eασ,regulate_knorm_1,regulate_G12_1,regulate_Δασ_1)
E=cal_energy(G12ασ,x,βασ,momentum_info,interaction,regulate_knorm_1,regulate_G12_1,regulate_Δασ_1)
∂E∂G12,∂E∂x,∂E∂βασ=cal_gradient(G12ασ,x,βασ,nασ,eασ,interaction,regulate_knorm_1,regulate_G12_1,regulate_Δασ_1)

# we first check derivatives
# we load some parameters

function loadAsSpinOrbital(filename_base,qauntity_name,N_spin_orbital)
    [loadData("$(filename_base)_$(qauntity_name)_spin_orb_$(i).dat") for i in 1:N_spin_orbital]
end

function load_old_para(U)
    old_data_dir="./two_band_degenerate_inf_half"
    old_filename_base=replace("$(old_data_dir)/U_$(U)_n_$(nασ)",","=>"_"," "=>"","["=>"","]"=>"")
    G12ασ=[i[1] for i in loadAsSpinOrbital(old_filename_base,"G12",N_spin_orbital)]
    βασ=[reshape(i,2) for i in loadAsSpinOrbital(old_filename_base,"beta",N_spin_orbital)]
    w= reshape(loadData("$(old_filename_base)_w.dat"),:)
    w_check= reshape(loadData("$(old_filename_base)_w.dat"),:)
    w02=cal_w02(cal_neffασ(nασ,G12ασ))
    VΓη,ηToIdx=__global_VΓη_ηToIdx__[N_spin_orbital]
    x=pinv(VΓη)*(w.^2-w02)
    # check=cal_w_scaled(x,nασ,G12ασ)-w_check
    return G12ασ,x,βασ
end

# some converting function
function convert_to_para(G12ασ,x,βασ)
    [G12ασ...,x...,reduce(vcat,βασ)...]
end

function convert_para_to_Gxβ(para,N_G,N_x,N_β)
    G12ασ=para[1:N_G]
    x=para[(N_G+1):(N_G+N_x)]
    βασ=[para[(N_G+N_x+1):(N_G+N_x+N_β)][(i*2-1):(i*2)] for i in 1:N_G]
    G12ασ,x,βασ
end


# test derivatives
U=4.0
interaction=[(1,2,U),(1,3,U),(1,4,U),(2,3,U),(2,4,U),(3,4,U)]
G12ασ,x,βασe=load_old_para(U)
para=convert_to_para(G12ασ,x,βασ)
para=para+rand(length(para))*1e-1
N_spin_orbital=4
N_G=N_spin_orbital
N_x=2^N_spin_orbital-(1+N_spin_orbital)
N_β=N_spin_orbital*2

# G12ασ_t,x_t,βασ_t=convert_para_to_Gxβ(para,N_G,N_x,N_β)
δpara=rand(length(para))*1e-6
para_1=para+δpara
G12ασ,x,βασ=convert_para_to_Gxβ(para,N_G,N_x,N_β)
G12ασ_1,x_1,βασ_1=convert_para_to_Gxβ(para_1,N_G,N_x,N_β)
E_0,_,_=cal_energy_direct(G12ασ,x,βασ,nασ,eασ,interaction,regulate_knorm_1,regulate_G12_1,regulate_Δασ_1)
E_1,_,_=cal_energy_direct(G12ασ_1,x_1,βασ_1,nασ,eασ,interaction,regulate_knorm_1,regulate_G12_1,regulate_Δασ_1)
# numerical version
δE=E_1-E_0


# use the momentum gradients and AD
momentum_info=cal_momentum_derivatives(nασ,G12ασ,x,βασ,eασ,regulate_knorm_1,regulate_G12_1,regulate_Δασ_1)
E_0_check=cal_energy(G12ασ,x,βασ,momentum_info,interaction,regulate_knorm_1,regulate_G12_1,regulate_Δασ_1)
E_0_check-E_0
∂E∂G12,∂E∂x,∂E∂βασ=cal_gradient(G12ασ,x,βασ,nασ,eασ,interaction,regulate_knorm_1,regulate_G12_1,regulate_Δασ_1)
∂E∂para=convert_to_para(∂E∂G12,∂E∂x,∂E∂βασ)
δE_check=dot(δpara,∂E∂para)
δE/δE_check

# we now chekc the numerical minimization first
using Optim
@time res=optimize(para->cal_energy_direct(convert_para_to_Gxβ(para,N_G,N_x,N_β)...,nασ,eασ,interaction,regulate_knorm_1,regulate_G12_1,regulate_Δασ_1)[1],para,LBFGS())
# 2 seconds
res.minimum
res.minimizer

# with user supplied gradients
function target(para)
    print("call with $((para[1],para[end],interaction[1]))\n")
    G12ασ,x,βασ=convert_para_to_Gxβ(para,N_G,N_x,N_β)
    cal_energy_direct(G12ασ,x,βασ,nασ,eασ,interaction,regulate_knorm_1,regulate_G12_1,regulate_Δασ_1)[1]
end

function target_grad!(G, para)
    print("call derivatives $((para[1],para[end],interaction[1]))\n")
    G12ασ,x,βασ=convert_para_to_Gxβ(para,N_G,N_x,N_β)
    ∂E∂G12,∂E∂x,∂E∂βασ=cal_gradient(G12ασ,x,βασ,nασ,eασ,interaction,regulate_knorm_1,regulate_G12_1,regulate_Δασ_1)
    ∂E∂para=convert_to_para(∂E∂G12,∂E∂x,∂E∂βασ)
    G[:]=∂E∂para
end
using LineSearches
LineSearches.BackTracking(order=1)

U=4.0
interaction=[(1,2,U),(1,3,U),(1,4,U),(2,3,U),(2,4,U),(3,4,U)]
G12ασ,x,βασe=load_old_para(U)
para=convert_to_para(G12ασ,x,βασ)

@time res2=optimize(target,target_grad!, para, LBFGS())
res2.minimum
res2.minimizer
result_LBFGS_v2_diff=Dict()
for U in 8.0:0.1:15.0
    print("$(U)\n")
    global interaction,para
    interaction=[(1,2,U),(1,3,U),(1,4,U),(2,3,U),(2,4,U),(3,4,U)]
    # res=optimize(target,target_grad!, para, LBFGS(),Optim.Options(time_limit=1.0))
    res=optimize(target,target_grad!, para, LBFGS(linesearch=LineSearches.MoreThuente()),
                 Optim.Options(time_limit=1.0))
    para=res.minimizer
    result_LBFGS_v2_diff[U]=res
end

U=8.0
para=result_LBFGS_v2_diff[U].minimizer

result_LBFGS_v2_diff[U].minimizer,result_LBFGS_v2_diff[U].minimum
