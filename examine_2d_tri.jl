include("./vdat.jl")
data_dir="./data_2d_tri_with_Z"
mkdir(data_dir)

e_fn=gene_spline_band("./es_2d_tri.dat")
nσ=[0.4,0.6]
nσs=0.5:-0.01:0.01
function gene_density_list(ns_low;digits=2)
    trunc_fn=n->round(n,digits=digits)
    [trunc_fn.([n_,1-n_]) for n_ in ns_low]
end
nσ_list=gene_density_list(nσs)


for nσ in nσ_list
    print("nσ is $(nσ)\n")
    ϵsσ=[gene_ϵs(e_fn,nσ[1]),gene_ϵs(e_fn,nσ[2])]
    ϵfσ=e_fn.(nσ)
    para=[-0.02,0.6,0.6,0.38]
    for U in 0.1:0.1:10.0
        para,result=solve_vdat(U,ϵsσ,nσ,para;option="half-filling-magnetization")
        save_result(U,ϵsσ,nσ,ϵfσ,result,data_dir)
    end
end

U=1.0
for U in 1.0:1.0:10.0
    data=[[(nσ[1],load_result(U,nσ,data_dir)...) for nσ in nσ_list]...]
    saveData(data,"$(data_dir)/E_etc_vs_ns_U_$(U).dat")
end
U=7.0
using Dierckx
for U in 1.0:1.0:10.0
    data=loadData("$(data_dir)/E_etc_vs_ns_U_$(U).dat")
    n=data[:,1]
    E=data[:,2]
    M=1.0 .-2.0*n
    fn_E_M=Spline1D(M,E)
    saveData(hcat(derivative(fn_E_M,M),M),"$(data_dir)/M_B_U_$(U).dat")
end

