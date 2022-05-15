data_dir="./data_2d"
mkdir(data_dir)
nσ=[0.5,0.5]
e_fn=gene_spline_band("./es_2d.dat")
ϵsσ=[gene_ϵs(e_fn,nσ[1]),gene_ϵs(e_fn,nσ[2])]
para=[-0.17,0.7,0.38]
for U in 1.0:1.0:16.0
    para,result=solve_vdat(U,ϵsσ,nσ,para;option="half-filling")
    save_result(U,ϵsσ,nσ,result,data_dir)
end
data_dir="./data_2d"
nσ=[0.5,0.5]
data=[(0.0,0.25),[(U,load_result(U,nσ,data_dir)[end]) for U in 1.0:1.0:16.0]...]
saveData(data,"$(data_dir)/U_d_ns_$(nσ[1])_$(nσ[2]).dat")

nσ=[0.4,0.6]
# specify other density
# nσ=[0.3,0.7]
# nσ=[0.1,0.9]
# nσ=[0.2,0.8]
ϵsσ=[gene_ϵs(e_fn,nσ[1]),gene_ϵs(e_fn,nσ[2])]
para=[-0.17,0.7,0.6,0.37]
for U in 1.0:0.1:16.0
    para,result=solve_vdat(U,ϵsσ,nσ,para;option="half-filling-magnetization")
    save_result(U,ϵsσ,nσ,result,data_dir)
end

