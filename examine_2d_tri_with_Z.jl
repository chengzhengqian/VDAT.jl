include("./vdat.jl")
data_dir="./data_2d_tri_with_Z"
mkdir(data_dir)


e_fn=gene_spline_band("./es_2d_tri.dat")
# nσs=0.5:-0.01:0.01
# function gene_density_list(ns_low;digits=2)
#     trunc_fn=n->round(n,digits=digits)
#     [trunc_fn.([n_,1-n_]) for n_ in ns_low]
# end
# nσ_list=gene_density_list(nσs)
nσ_list=[[0.2,0.8],[0.4,0.6],[0.3,0.7],[0.5,0.5]]
# for nσ in nσ_list#
for nσ in [[0.1,0.9]]
    print("nσ is $(nσ)\n")
    ϵsσ=[gene_ϵs(e_fn,nσ[1]),gene_ϵs(e_fn,nσ[2])]
    ϵfσ=e_fn.(nσ)
    para=[-0.02,0.6,0.6,0.6,0.6,0.38,0.38]
    for U in 0.1:0.1:20.0
        para,result=solve_vdat(U,ϵsσ,nσ,para;option="general")
        save_result(U,ϵsσ,nσ,ϵfσ,result,data_dir)
    end
end


# for nσ in nσ_list
for nσ in  [[0.1,0.9]]
    data=[(0.0,1.0,1.0),[(U,load_result(U,nσ,data_dir)[(end-1):end]...)
                         for U in 0.1:0.1:20.0]...]
    saveData(data,"$(data_dir)/Z_up_Z_dn_ns_$(nσ[1])_$(nσ[2]).dat")
end

