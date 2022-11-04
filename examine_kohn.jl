
data_dir="1d_kohn_examine"
ϕ=0.0
nσ=[0.5,0.5]

for ϕ=-0.1:0.01:0.1
    subdir="$(data_dir)/phi_$(ϕ)"
    if(!isdir(subdir))
        mkdir(subdir)
    end
    ϵfσ=[0.0,0.0]
    k_below=linspace(-pi/2,pi/2,20).+ϕ
    k_above=linspace(pi/2,3*pi/2,20).+ϕ
    ϵsσ_=[-2*cos.(k_below),-2*cos.(k_above)]
    ϵsσ=[ϵsσ_,ϵsσ_]
    para=[-0.2,0.7,0.7,0.38]
    for U in 1.0:1.0:20.0
        para,result=solve_vdat(U,ϵsσ,nσ,para;option="no-magnetization")
        save_result(U,ϵsσ,nσ,ϵfσ,result,subdir)
    end
end
U=1.0
using Dierckx
for U in 1.0:1.0:20.0
    E0=load_result(U,nσ,"$(data_dir)/phi_$(0.0)")[1]
    data=[(ϕ, load_result(U,nσ,"$(data_dir)/phi_$(ϕ)")[1]-E0) for ϕ in -0.1:0.01:0.1]
    saveData(data,"$(data_dir)/E_phi_U_$(U).dat")
end

result=[]
for U in 1.0:1.0:20.0
    data=loadData("$(data_dir)/E_phi_U_$(U).dat")
    fn=Spline1D(data[:,1],data[:,2])
    push!(result,[U,derivative(fn,0,nu=2)])
end

saveData(result,"$(data_dir)/d2E_dphi2_1d.dat")

data=[(0.0,1.0,1.0),[(U,load_result(U,nσ,"$(data_dir)/phi_$(0.0)")[(end-1):end]...) for U in 1.0:1.0:20.0]...]
saveData(data,"$(data_dir)/Z_U_1d.dat")



