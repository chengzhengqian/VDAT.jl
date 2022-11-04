# we test the new scheme, direct compute derivative from G,x,β


U=4.0
U=1.0
# U=10.0
nασ=[0.5 for _ in 1:N_spin_orbital]


old_data_dir="./two_band_degenerate_inf_half"
old_filename_base=replace("$(old_data_dir)/U_$(U)_n_$(nασ)",","=>"_"," "=>"","["=>"","]"=>"")

G12ασ=[i[1] for i in loadAsSpinOrbital(old_filename_base,"G12",N_spin_orbital)]
βασ=[reshape(i,2) for i in loadAsSpinOrbital(old_filename_base,"beta",N_spin_orbital)]
w= reshape(loadData("$(old_filename_base)_w.dat"),:)
w_check= reshape(loadData("$(old_filename_base)_w.dat"),:)
w02=cal_w02(cal_neffασ(nασ,G12ασ))
VΓη,ηToIdx=__global_VΓη_ηToIdx__[N_spin_orbital]
x=pinv(VΓη)*(w.^2-w02)
cal_w_scaled(x,nασ,G12ασ)-w_check
U=1.5
U=2.0
U=3.0
U=5.0
U=9.0
interaction=[(1,2,U),(1,3,U),(1,4,U),(2,3,U),(2,4,U),(3,4,U)]

for i in 1:100
    E,Eloc,K=cal_total_energy_direct(G12ασ,x,βασ,nασ,eασ,interaction)
    ∂E∂G12,∂E∂x,∂E∂βασ=cal_gradient(G12ασ,x,βασ,nασ,eασ,interaction)
    step=1*1e-2
    function E_step(step)
        G12ασ_new=G12ασ-∂E∂G12*step
        x_new=x-∂E∂x*step
        βασ_new=βασ.-∂E∂βασ*step
        E_new,_,_=cal_total_energy_direct(G12ασ_new,x_new,βασ_new,nασ,eασ,interaction)
        return E_new,G12ασ_new,x_new,βασ_new
    end
    res=optimize(step->E_step(step[1])[1],[0.0])
    step=res.minimizer[1]
    E_new,G12ασ_new,x_new,βασ_new=E_step(step)
    print("step:$(step),ΔE: $(E_new-E),E:$(E_new)\n")
    G12ασ,x,βασ=G12ασ_new,x_new,βασ_new
end
# three steps
function E_G_step(step)
    G12ασ_new=G12ασ-∂E∂G12*step
    # x_new=x-∂E∂x*step
    # βασ_new=βασ.-∂E∂βασ*step
    E_new,_,_=cal_total_energy_direct(G12ασ_new,x,βασ,nασ,eασ,interaction)
    return E_new,G12ασ_new,x,βασ
end
function E_β_step(step)
    # G12ασ_new=G12ασ-∂E∂G12*step
    # x_new=x-∂E∂x*step
    βασ_new=βασ.-∂E∂βασ*step
    E_new,_,_=cal_total_energy_direct(G12ασ,x,βασ_new,nασ,eασ,interaction)
    return E_new,G12ασ,x,βασ_new
end
function E_x_step(step)
    # G12ασ_new=G12ασ-∂E∂G12*step
    x_new=x-∂E∂x*step
    # βασ_new=βασ.-∂E∂βασ*step
    E_new,_,_=cal_total_energy_direct(G12ασ,x_new,βασ,nασ,eασ,interaction)
    return E_new,G12ασ,x_new,βασ
end

# then we check if
res_BFGS.minimizer

for i in 1:10
    # E,Eloc,K=cal_total_energy_direct(G12ασ,x,βασ,nασ,eασ,interaction)
    ∂E∂G12,∂E∂x,∂E∂βασ=cal_gradient(G12ασ,x,βασ,nασ,eασ,interaction)
    # res=optimize(step->E_G_step(step[1])[1],[0.0])
    res=optimize(step->E_G_step(step[1])[1],[0.0],LBFGS())
    step=res.minimizer[1]
    E_new,G12ασ_new,x_new,βασ_new=E_G_step(step)
    print("step:$(step),ΔE: $(E_new-E),E:$(E_new)\n")
    E,G12ασ,x,βασ=E_new,G12ασ_new,x_new,βασ_new

    ∂E∂G12,∂E∂x,∂E∂βασ=cal_gradient(G12ασ,x,βασ,nασ,eασ,interaction)
    res=optimize(step->E_x_step(step[1])[1],[0.0])
    # res=optimize(step->E_x_step(step[1])[1],[0.0],LBFGS())
    step=res.minimizer[1]
    E_new,G12ασ_new,x_new,βασ_new=E_x_step(step)
    print("step:$(step),ΔE: $(E_new-E),E:$(E_new)\n")
    E,G12ασ,x,βασ=E_new,G12ασ_new,x_new,βασ_new

    ∂E∂G12,∂E∂x,∂E∂βασ=cal_gradient(G12ασ,x,βασ,nασ,eασ,interaction)
    res=optimize(step->E_β_step(step[1])[1],[0.0])
    # res=optimize(step->E_x_step(step[1])[1],[0.0],LBFGS())
    step=res.minimizer[1]
    E_new,G12ασ_new,x_new,βασ_new=E_β_step(step)
    print("step:$(step),ΔE: $(E_new-E),E:$(E_new)\n")
    E,G12ασ,x,βασ=E_new,G12ασ_new,x_new,βασ_new
end



# there are some problems, the first step to check is that if we numerical minimze the reuslt , we will get the same result

using Optim

res=optimize(para->cal_total_energy_direct([para[1] for _ in 1:4],[[para[2] for _ in 1:6]...,[para[3] for _ in 1:4]...,para[4]],[[para[5],para[5]] for _ in 1:4],nασ,eασ,interaction)[1],[0.45,-1.2,0.0,8,3])
para=res.minimizer
cal_total_energy_direct([para[1] for _ in 1:4],[[para[2] for _ in 1:6]...,[para[3] for _ in 1:4]...,para[4]],[[para[5],para[5]] for _ in 1:4],nασ,eασ,interaction)
G12ασ=[para[1] for _ in 1:4]
x=[[para[2] for _ in 1:6]...,[para[3] for _ in 1:4]...,para[4]]
βασ=[[para[5],para[5]] for _ in 1:4]
momentum_info=cal_momentum_part_in_nβ(nασ,G12ασ,x,βασ,eασ)
cal_total_energy(G12ασ,x,βασ,momentum_info,interaction)


# there are some problems in direclty using the first order, we try BGFS algorithm
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
    check=cal_w_scaled(x,nασ,G12ασ)-w_check
    return G12ασ,x,βασ,check
end

U=4.0
G12ασ,x,βασ,check=load_old_para(U)
para_init=[G12ασ[1],x[1],x[7],x[end],βασ[1][1]]
interaction=[(1,2,U),(1,3,U),(1,4,U),(2,3,U),(2,4,U),(3,4,U)]
# Nelder-Mead, use some simplex to narrow down the minimum, no gradient is required
res_NM=optimize(para->cal_total_energy_direct([para[1] for _ in 1:4],[[para[2] for _ in 1:6]...,[para[3] for _ in 1:4]...,para[4]],[[para[5],para[5]] for _ in 1:4],nασ,eασ,interaction)[1],para_init)
para_init_NM=res_NM.minimizer
para_init_NM-para_init

res_BFGS=optimize(para->cal_total_energy_direct([para[1] for _ in 1:4],[[para[2] for _ in 1:6]...,[para[3] for _ in 1:4]...,para[4]],[[para[5],para[5]] for _ in 1:4],nασ,eασ,interaction)[1],para_init,BFGS())
para_init_BFGS=res_BFGS.minimizer
para_init_BFGS-para_init
para_init_BFGS-para_init_NM

res_BFGS.minimum-res_NM.minimum

# now, consider U=5
U=5.0
interaction=[(1,2,U),(1,3,U),(1,4,U),(2,3,U),(2,4,U),(3,4,U)]

res_NM=optimize(para->cal_total_energy_direct([para[1] for _ in 1:4],[[para[2] for _ in 1:6]...,[para[3] for _ in 1:4]...,para[4]],[[para[5],para[5]] for _ in 1:4],nασ,eασ,interaction)[1],para_init)
para_new_NM=res_NM.minimizer
para_new_NM-para_init

res_BFGS=optimize(para->cal_total_energy_direct([para[1] for _ in 1:4],[[para[2] for _ in 1:6]...,[para[3] for _ in 1:4]...,para[4]],[[para[5],para[5]] for _ in 1:4],nασ,eασ,interaction)[1],para_init,BFGS())
para_new_BFGS=res_BFGS.minimizer
para_new_BFGS-para_init

res_BFGS.minimum
res_NM.minimum

# then we see if we could use BFGS algorithm in the large U region
U=5.0
interaction=[(1,2,U),(1,3,U),(1,4,U),(2,3,U),(2,4,U),(3,4,U)]
para=res_BFGS.minimizer
result=Dict()
for U in 5.0:0.1:10.0
    interaction=[(1,2,U),(1,3,U),(1,4,U),(2,3,U),(2,4,U),(3,4,U)]
    res_BFGS=optimize(para->cal_total_energy_direct([para[1] for _ in 1:4],[[para[2] for _ in 1:6]...,[para[3] for _ in 1:4]...,para[4]],[[para[5],para[5]] for _ in 1:4],nασ,eασ,interaction)[1],para,BFGS())
    para=res_BFGS.minimizer
    result[U]=res_BFGS
    print("$(U)\n")
end

U=5.0
result[5.1].minimizer

# for U=4.0 , we the non-linear form
U=4.0
G12ασ,x,βασ,check=load_old_para(U)
para_init=[G12ασ[1],x[1],x[7],x[end],βασ[1][1]]
interaction=[(1,2,U),(1,3,U),(1,4,U),(2,3,U),(2,4,U),(3,4,U)]
# Nelder-Mead, use some simplex to narrow down the minimum, no gradient is required
res_NM_v2=optimize(para->cal_total_energy_direct_v2([para[1] for _ in 1:4],[[para[2] for _ in 1:6]...,[para[3] for _ in 1:4]...,para[4]],[[para[5],para[5]] for _ in 1:4],nασ,eασ,interaction)[1],para_init)
para_init_NM_v2=res_NM_v2.minimizer
res_NM_v2.minimum,res_NM_v2.minimizer
result_NM_v2=Dict()
U=10.0
U=8.0
result_NM_v2[U].minimum,result_NM_v2[U].minimizer
for U in 4.0:0.1:10.0
    interaction=[(1,2,U),(1,3,U),(1,4,U),(2,3,U),(2,4,U),(3,4,U)]
    res_NM_v2=optimize(para->cal_total_energy_direct_v2([para[1] for _ in 1:4],[[para[2] for _ in 1:6]...,[para[3] for _ in 1:4]...,para[4]],[[para[5],para[5]] for _ in 1:4],nασ,eασ,interaction)[1],para_init_NM_v2)
    para_init_NM_v2=res_NM_v2.minimizer
    result_NM_v2[U]=res_NM_v2
    print("$(U)\n")
end

(para->cal_total_energy_direct_v2([para[1] for _ in 1:4],[[para[2] for _ in 1:6]...,[para[3] for _ in 1:4]...,para[4]],[[para[5],para[5]] for _ in 1:4],nασ,eασ,interaction))(result_NM_v2[U].minimizer)

# we check for BFGS

U=4.0
G12ασ,x,βασ,check=load_old_para(U)
para_init=[G12ασ[1],x[1],x[7],x[end],βασ[1][1]]
interaction=[(1,2,U),(1,3,U),(1,4,U),(2,3,U),(2,4,U),(3,4,U)]
res_BFGS_v2=optimize(para->cal_total_energy_direct_v2([para[1] for _ in 1:4],[[para[2] for _ in 1:6]...,[para[3] for _ in 1:4]...,para[4]],[[para[5],para[5]] for _ in 1:4],nασ,eασ,interaction)[1],para_init,BFGS())
para_init_BFGS_v2=res_BFGS_v2.minimizer
res_BFGS_v2.minimum,res_BFGS_v2.minimizer

result_BFGS_v2=Dict()
U=8.0
U=10.0
result_BFGS_v2[U].minimum,result_BFGS_v2[U].minimizer
sort(collect(keys(result_BFGS_v2)))
para_init_BFGS_v2=result_BFGS_v2[8.8].minimizer
for U in 8.8:0.1:10.0
    interaction=[(1,2,U),(1,3,U),(1,4,U),(2,3,U),(2,4,U),(3,4,U)]
    res_BFGS_v2=optimize(para->cal_total_energy_direct_v2([para[1] for _ in 1:4],[[para[2] for _ in 1:6]...,[para[3] for _ in 1:4]...,para[4]],[[para[5],para[5]] for _ in 1:4],nασ,eασ,interaction)[1],para_init_BFGS_v2,BFGS(),Optim.Options(time_limit=1.0))
    para_init_BFGS_v2=res_BFGS_v2.minimizer
    result_BFGS_v2[U]=res_BFGS_v2
    print("$(U)\n")
end

U=10.0
(para->cal_total_energy_direct_v2([para[1] for _ in 1:4],[[para[2] for _ in 1:6]...,[para[3] for _ in 1:4]...,para[4]],[[para[5],para[5]] for _ in 1:4],nασ,eασ,interaction))(result_BFGS_v2[U].minimizer)


# now, we try gradient descent

function E_G_step_v2(step)
    G12ασ_new=G12ασ-∂E∂G12*step
    # x_new=x-∂E∂x*step
    # βασ_new=βασ.-∂E∂βασ*step
    E_new,_,_=cal_total_energy_direct_v2(G12ασ_new,x,βασ,nασ,eασ,interaction)
    return E_new,G12ασ_new,x,βασ
end
function E_β_step_v2(step)
    # G12ασ_new=G12ασ-∂E∂G12*step
    # x_new=x-∂E∂x*step
    βασ_new=βασ.-∂E∂βασ*step
    E_new,_,_=cal_total_energy_direct_v2(G12ασ,x,βασ_new,nασ,eασ,interaction)
    return E_new,G12ασ,x,βασ_new
end
function E_x_step_v2(step)
    # G12ασ_new=G12ασ-∂E∂G12*step
    x_new=x-∂E∂x*step
    # βασ_new=βασ.-∂E∂βασ*step
    E_new,_,_=cal_total_energy_direct_v2(G12ασ,x_new,βασ,nασ,eασ,interaction)
    return E_new,G12ασ,x_new,βασ
end

U=8.0
U=10.0
interaction=[(1,2,U),(1,3,U),(1,4,U),(2,3,U),(2,4,U),(3,4,U)]

U=4.0
G12ασ,x,βασ,check=load_old_para(U)
para_init=[G12ασ[1],x[1],x[7],x[end],βασ[1][1]]
U=8.0
interaction=[(1,2,U),(1,3,U),(1,4,U),(2,3,U),(2,4,U),(3,4,U)]

for i in 1:10
    # E,Eloc,K=cal_total_energy_direct(G12ασ,x,βασ,nασ,eασ,interaction)
    ∂E∂G12,∂E∂x,∂E∂βασ=cal_gradient_v2(G12ασ,x,βασ,nασ,eασ,interaction)
    # res=optimize(step->E_G_step(step[1])[1],[0.0])
    res=optimize(step->E_G_step_v2(step[1])[1],[0.0])
    step=res.minimizer[1]
    E_new,G12ασ_new,x_new,βασ_new=E_G_step_v2(step)
    print("step:$(step),ΔE: $(E_new-E),E:$(E_new)\n")
    E,G12ασ,x,βασ=E_new,G12ασ_new,x_new,βασ_new
    ∂E∂G12,∂E∂x,∂E∂βασ=cal_gradient_v2(G12ασ,x,βασ,nασ,eασ,interaction)
    res=optimize(step->E_x_step_v2(step[1])[1],[0.0])
    # res=optimize(step->E_x_step(step[1])[1],[0.0],LBFGS())
    step=res.minimizer[1]
    E_new,G12ασ_new,x_new,βασ_new=E_x_step_v2(step)
    print("step:$(step),ΔE: $(E_new-E),E:$(E_new)\n")
    E,G12ασ,x,βασ=E_new,G12ασ_new,x_new,βασ_new
    ∂E∂G12,∂E∂x,∂E∂βασ=cal_gradient_v2(G12ασ,x,βασ,nασ,eασ,interaction)
    res=optimize(step->E_β_step_v2(step[1])[1],[0.0])
    # res=optimize(step->E_x_step(step[1])[1],[0.0],LBFGS())
    step=res.minimizer[1]
    E_new,G12ασ_new,x_new,βασ_new=E_β_step_v2(step)
    print("step:$(step),ΔE: $(E_new-E),E:$(E_new)\n")
    E,G12ασ,x,βασ=E_new,G12ασ_new,x_new,βασ_new
end

# check
U=8.0
para=result_NM_v2[U].minimizer
G12ασ=[para[1] for _ in 1:4]
x=[[para[2] for _ in 1:6]...,[para[3] for _ in 1:4]...,para[4]]
βασ=[[para[5],para[5]] for _ in 1:4]

U=4.0
interaction=[(1,2,U),(1,3,U),(1,4,U),(2,3,U),(2,4,U),(3,4,U)]
cal_total_energy_direct_v2(G12ασ,x,βασ,nασ,eασ,interaction)
momentum_info=cal_momentum_part_in_nβ_v2(nασ,G12ασ,x,βασ,eασ)
cal_total_energy_v2(G12ασ,x,βασ,momentum_info,interaction)
∂E∂G12,∂E∂x,∂E∂βασ=gradient((G12ασ,x,βασ)->cal_total_energy_v2(G12ασ,x,βασ,momentum_info,interaction),G12ασ,x,βασ)

# Finally, we check LBFGS, a low memory version

U=4.0
G12ασ,x,βασ,check=load_old_para(U)
para_init=[G12ασ[1],x[1],x[7],x[end],βασ[1][1]]
interaction=[(1,2,U),(1,3,U),(1,4,U),(2,3,U),(2,4,U),(3,4,U)]
res_LBFGS_v2=optimize(para->cal_total_energy_direct_v2([para[1] for _ in 1:4],[[para[2] for _ in 1:6]...,[para[3] for _ in 1:4]...,para[4]],[[para[5],para[5]] for _ in 1:4],nασ,eασ,interaction)[1],para_init,LBFGS())
para_init_LBFGS_v2=res_LBFGS_v2.minimizer
res_LBFGS_v2.minimum,res_LBFGS_v2.minimizer

result_LBFGS_v2=Dict()
U=8.0
U=10.0
U=4.0
result_LBFGS_v2[U].minimum,result_LBFGS_v2[U].minimizer
sort(collect(keys(result_LBFGS_v2)))
para_init_LBFGS_v2=result_LBFGS_v2[4.0].minimizer
for U in 4.0:0.1:10.0
    interaction=[(1,2,U),(1,3,U),(1,4,U),(2,3,U),(2,4,U),(3,4,U)]
    res_LBFGS_v2=optimize(para->cal_total_energy_direct_v2([para[1] for _ in 1:4],[[para[2] for _ in 1:6]...,[para[3] for _ in 1:4]...,para[4]],[[para[5],para[5]] for _ in 1:4],nασ,eασ,interaction)[1],para_init_LBFGS_v2,LBFGS(),Optim.Options(time_limit=0.4))
    para_init_LBFGS_v2=res_LBFGS_v2.minimizer
    result_LBFGS_v2[U]=res_LBFGS_v2
    print("$(U)\n")
end

para=result_LBFGS_v2[10.0].minimizer
G12ασ=[para[1] for _ in 1:4]
x=[[para[2] for _ in 1:6]...,[para[3] for _ in 1:4]...,para[4]]
βασ=[[para[5],para[5]] for _ in 1:4]
momentum_info=cal_momentum_part_in_nβ_v2(nασ,G12ασ,x,βασ,eασ)
(nασ,Δασ,nασ_below,nασ_above,αασ_below,αασ_above,βασ_below,βασ_above,nkασ_below,nkασ_above,Aασ_below,Aασ_above,Kασ_below,Kασ_above,∂Kασ∂nX_below,∂Kασ∂nX_above,∂Kασ∂βX_below,∂Kασ∂βX_above,∂Aασ∂nX_below,∂Aασ∂nX_above,∂Aασ∂βX_below,∂Aασ∂βX_above)=momentum_info

(para->cal_total_energy_direct_v2([para[1] for _ in 1:4],[[para[2] for _ in 1:6]...,[para[3] for _ in 1:4]...,para[4]],[[para[5],para[5]] for _ in 1:4],nασ,eασ,interaction))(result_LBFGS_v2[10.0].minimizer)

# now, we examine the optimization use user defined gradients
#  we first follow the example

function target(x)
    print("call target with $(x)\n")
    (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

# we define the gradient
function target_grad!(G, x)
    print("call target grad with $(x)\n")
    G[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    G[2] = 200.0 * (x[2] - x[1]^2)
end


res=optimize(target,target_grad!, [0.0,0.0], LBFGS())

# we we check with our problem
# we first load the parameters

U=4.0
# this is for v1, but should be a good starting point for v2
G12ασ,x,βασ,check=load_old_para(U)

function convert_to_vec(G12ασ,x,βασ)
    [G12ασ...,x...,reduce(vcat,βασ)...]
end

para=convert_to_vec(G12ασ,x,βασ)
"""
N_spin_orbital=4
N_G=N_spin_orbital
N_x=2^N_spin_orbital-(1+N_spin_orbital)
N_β=N_spin_orbital*2
"""
function convert_para_to_Gxβ(para,N_G,N_x,N_β)
    G12ασ=para[1:N_G]
    x=para[(N_G+1):(N_G+N_x)]
    βασ=[para[(N_G+N_x+1):(N_G+N_x+N_β)][(i*2-1):(i*2)] for i in 1:N_G]
    G12ασ,x,βασ
end

para_t=rand(N_G+N_x+N_β)
G12ασ_t,x_t,βασ_t=convert_para_to_Gxβ(para_t,N_G,N_x,N_β)
para_t_check=convert_to_vec(G12ασ_t,x_t,βασ_t)
para_t_check-para_t

# we now check the derivatives.
interaction=[(1,2,U),(1,3,U),(1,4,U),(2,3,U),(2,4,U),(3,4,U)]
G12ασ,x,βασ,check=load_old_para(U)
para_0=convert_to_vec(G12ασ,x,βασ)+rand(length(para_0))*1e-1

G12ασ_0,x_0,βασ_0=convert_para_to_Gxβ(para_0,N_G,N_x,N_β)
E0,_,_=cal_total_energy_direct_v2(G12ασ_0,x_0,βασ_0,nασ,eασ,interaction)
δpara=rand(length(para_0))*1e-4
para_1=para_0+δpara
G12ασ_1,x_1,βασ_1=convert_para_to_Gxβ(para_1,N_G,N_x,N_β)
E1,_,_=cal_total_energy_direct_v2(G12ασ_1,x_1,βασ_1,nασ,eασ,interaction)
δE=E1-E0

∂E∂G12,∂E∂x,∂E∂βασ=cal_gradient_v2(G12ασ_0,x_0,βασ_0,nασ,eασ,interaction)
∂E∂para=convert_to_vec(∂E∂G12,∂E∂x,∂E∂βασ)
δE_check=dot(δpara,∂E∂para)
δE/δE_check

# 
interaction=[(1,2,U),(1,3,U),(1,4,U),(2,3,U),(2,4,U),(3,4,U)]
U=4.0
G12ασ,x,βασ,check=load_old_para(U)
para_init=convert_to_vec(G12ασ,x,βασ)


function target(para)
    G12ασ,x,βασ=convert_para_to_Gxβ(para,N_G,N_x,N_β)
    print("call target with $([G12ασ[1],βασ[1]])\n")
    cal_total_energy_direct_v2(G12ασ,x,βασ,nασ,eασ,interaction)[1]
end

# we define the gradient
function target_grad!(G, para)
    G12ασ,x,βασ=convert_para_to_Gxβ(para,N_G,N_x,N_β)
    ∂E∂G12,∂E∂x,∂E∂βασ=cal_gradient_v2(G12ασ,x,βασ,nασ,eασ,interaction)
    ∂E∂para=convert_to_vec(∂E∂G12,∂E∂x,∂E∂βασ)
    print("call target_grad with $([G12ασ[1],βασ[1]])\n")
    G[:]=∂E∂para
end


res=optimize(target,target_grad!, para_init, LBFGS())
U=5.0
interaction=[(1,2,U+0.2),(1,3,U-0.1),(1,4,U-0.3),(2,3,U+0.3),(2,4,U-0.1),(3,4,U+0.2)]
result_LBFGS_v2_diff=Dict()
sort(collect(keys(result_LBFGS_v2_diff)))
U=8.0
U=9.5
result_LBFGS_v2_diff[U].minimum,result_LBFGS_v2_diff[U].minimizer
para=result_LBFGS_v2_diff[U].minimizer
U=4.0
G12ασ,x,βασ,check=load_old_para(U)
para=convert_to_vec(G12ασ,x,βασ)
for U in 4.0:0.1:10.0
    interaction=[(1,2,U),(1,3,U),(1,4,U),(2,3,U),(2,4,U),(3,4,U)]
    res=optimize(target,target_grad!, para, LBFGS(),Optim.Options(time_limit=1.0))
    para=res.minimizer
    result_LBFGS_v2_diff[U]=res
end

# res.minimum
# res.minimizer
