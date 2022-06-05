include("./vdat.jl")
data_dir="./data_inf"
mkdir(data_dir)
e_fn=gene_spline_band("./es_inf.dat")
nσ=[0.5,0.5]
ϵsσ=[gene_ϵs(e_fn,nσ[1]),gene_ϵs(e_fn,nσ[2])]
para=[-0.17,0.7,0.38]
# U=1.0
# @time   para,result=solve_vdat(U,ϵsσ,nσ,para;option="half-filling")
for U in 1.0:1.0:10.0
    para,result=solve_vdat(U,ϵsσ,nσ,para;option="half-filling")
    save_result(U,ϵsσ,nσ,result,data_dir)
end

data_dir="./data_inf"
nσ=[0.5,0.5]
data=[(0.0,0.25),[(U,load_result(U,nσ,data_dir)[end]) for U in 1.0:0.1:10.0]...]
saveData(data,"$(data_dir)/U_d_ns_$(nσ[1])_$(nσ[2]).dat")


# specify other density for given magnetization
# nσ=[0.3,0.7]
# nσ=[0.1,0.9]
# nσ=[0.2,0.8]
nσ=[0.4,0.6]
ϵsσ=[gene_ϵs(e_fn,nσ[1]),gene_ϵs(e_fn,nσ[2])]
para=[-0.17,0.7,0.6,0.37]
for U in 1.0:1.0:10.0
    print("U $(U)\n")
    para,result=solve_vdat(U,ϵsσ,nσ,para;option="half-filling-magnetization")
    save_result(U,ϵsσ,nσ,result,data_dir)
end

# general case
e_fn=gene_spline_band("./es_inf.dat")
nσ=[0.3,0.4]
ϵsσ=[gene_ϵs(e_fn,nσ[1]),gene_ϵs(e_fn,nσ[2])]
para=[-0.17,0.7,0.7,0.7,0.7,0.38,0.38]
for U in 1.0:1.0:10.0
    para,result=solve_vdat(U,ϵsσ,nσ,para;option="general")
    save_result(U,ϵsσ,nσ,result,data_dir)
end

# no-magnetization cased
e_fn=gene_spline_band("./es_inf.dat")
nσ=[0.3,0.3]
ϵsσ=[gene_ϵs(e_fn,nσ[1]),gene_ϵs(e_fn,nσ[2])]
para=[-0.17,0.7,0.7,0.38]
for U in 1.0:1.0:10.0
    para,result=solve_vdat(U,ϵsσ,nσ,para;option="no-magnetization")
    save_result(U,ϵsσ,nσ,result,data_dir)
end



