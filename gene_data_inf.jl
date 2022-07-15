# import Pkg
# Pkg.add("Optim")
include("./vdat.jl")
data_dir="/home/chengzhengqian/Documents/research/vdat/hubbard_one_band_analytic/data_inf/"
mkdir(data_dir)

e_fn=gene_spline_band("./es_inf.dat")
nσ=[0.5,0.5]
ϵsσ=[gene_ϵs(e_fn,nσ[1]),gene_ϵs(e_fn,nσ[2])]

para=[-0.02,0.6,0.38]
for U in 0.1:0.1:10.0
    para,result=solve_vdat(U,ϵsσ,nσ,para;option="half-filling")
    save_result(U,ϵsσ,nσ,result,data_dir)
end

# magnetization
nσs=0.5:-0.01:0.01
function gene_density_list(ns_low;digits=2)
    trunc_fn=n->round(n,digits=digits)
    [trunc_fn.([n_,1-n_]) for n_ in ns_low]
end
nσ_list=gene_density_list(nσs)

for nσ in nσ_list
    print("nσ is $(nσ)\n")
    ϵsσ=[gene_ϵs(e_fn,nσ[1]),gene_ϵs(e_fn,nσ[2])]
    para=[-0.02,0.6,0.6,0.38]
    for U in 0.1:0.1:10.0
        para,result=solve_vdat(U,ϵsσ,nσ,para;option="half-filling-magnetization")
        save_result(U,ϵsσ,nσ,result,data_dir)
    end
end

# doped case


nσs=0.5:-0.01:0.01
nσs=0.5:-0.0001:0.49
nσ_list=[[nσ_,nσ_] for nσ_ in nσs]

for nσ in nσ_list
    print("nσ is $(nσ)\n")
    ϵsσ=[gene_ϵs(e_fn,nσ[1]),gene_ϵs(e_fn,nσ[2])]
    para=[-0.02,0.6,0.6,0.38]
    for U in 0.1:0.1:20.0
        para,result=solve_vdat(U,ϵsσ,nσ,para;option="no-magnetization")
        save_result(U,ϵsσ,nσ,result,data_dir)
    end
end
# there are some problems for some intermediate dnesity

for nσ in nσ_list
    data=[(0.0,nσ[1]*nσ[2]),[(U,load_result(U,nσ,data_dir)[end]) for U in 0.1:0.1:10.0]...]
    saveData(data,"$(data_dir)/U_d_ns_$(nσ[1])_$(nσ[2]).dat")
end



# collect data
data_dir="/home/chengzhengqian/Documents/research/vdat/hubbard_one_band_analytic/data_inf/"
nσ=[0.5,0.5]
nσs=0.5:-0.01:0.01
nσ_list=[[nσ_,nσ_] for nσ_ in nσs]
nσs=0.5:-0.01:0.01
nσ_list=gene_density_list(nσs)
for nσ in nσ_list
    data=[(0.0,nσ[1]*nσ[2]),[(U,load_result(U,nσ,data_dir)[end]) for U in 0.1:0.1:10.0]...]
    saveData(data,"$(data_dir)/U_d_ns_$(nσ[1])_$(nσ[2]).dat")
end


