# we first load parameters
U=1.0
U=4.0
U=10.0
nασ=[0.5 for _ in 1:N_spin_orbital]
N_spin_orbital=4

old_data_dir="./two_band_degenerate_inf_half"
old_filename_base=replace("$(old_data_dir)/U_$(U)_n_$(nασ)",","=>"_"," "=>"","["=>"","]"=>"")

# load old data (by numerical minimization)
G12ασ=[i[1] for i in loadAsSpinOrbital(old_filename_base,"G12",N_spin_orbital)]
βασ=[reshape(i,2) for i in loadAsSpinOrbital(old_filename_base,"beta",N_spin_orbital)]
w= reshape(loadData("$(old_filename_base)_w.dat"),:)
w_check= reshape(loadData("$(old_filename_base)_w.dat"),:)
w02=cal_w02(cal_neffασ(nασ,G12ασ))
VΓη,ηToIdx=__global_VΓη_ηToIdx__[N_spin_orbital]
x=pinv(VΓη)*(w.^2-w02)
w_check2=sqrt.(w02+VΓη*x)
w_check2-w

data_dir="./two_band_inf_diff/"
for U in 8.2:0.1:10.0
    tag="U_$(U)"
    interaction=[(1,2,U),(1,3,U),(1,4,U),(2,3,U),(2,4,U),(3,4,U)]
    G12ασ,x,βασ,∂E∂G12ασ,∂E∂x,∂E∂Aασ=solve_N3(G12ασ,x,βασ,nασ,eασ,interaction;λG=0.00,λx=0.0000,λβ=0.0,N_iter=1)
    G12ασ,x,βασ,∂E∂G12ασ,∂E∂x,∂E∂Aασ=solve_N3(G12ασ,x,βασ,nασ,eασ,interaction;λG=0.001,λx=0.00001,λβ=0.5,N_iter=1000)
    obs=cal_obs(G12ασ,x,βασ,nασ,eασ,interaction)
    print("tag is $(tag)\n")
    saveObs(obs,data_dir,tag,∂E∂G12ασ,∂E∂x,∂E∂Aασ)
end

# there are some problem in linear minimization,
# we check U=5.5

# check objs
E0=cal_obs(G12ασ,x,βασ,nασ,eασ,interaction)[1]
δG=1E-7
δG=1E-5
δG=1E-4
δEδG=(cal_obs(G12ασ.+δG,x,βασ,nασ,eασ,interaction)[1]-cal_obs(G12ασ.-δG,x,βασ,nασ,eασ,interaction)[1])/δG/8
E,Eloc,Ek,local_info,αασ,βασ,Δασ,Aασ,Slocασ,G12ασ,x,w,nk,nασ,eασ=obs

