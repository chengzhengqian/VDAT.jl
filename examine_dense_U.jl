# check whether these is any issue when we turn ΔU small

include("./vdat.jl")
data_dir="./data_inf_dense"
mkdir(data_dir)

e_fn=gene_spline_band("./es_inf.dat")
nσ=[0.5,0.5]
ϵsσ=[gene_ϵs(e_fn,nσ[1]),gene_ϵs(e_fn,nσ[2])]
para=[-0.17,0.7,0.38]

# we create a list to store all the intermediate parameters to check 
para_list=[]

ΔU=0.05
for U in 1.0:ΔU:10.0
    para,result=solve_vdat(U,ϵsσ,nσ,para;option="half-filling")
    E,Ek,Eloc,nkσ,d=result
    # we store the U, E, para, and use E to check any suspicious parameters
    push!(para_list,[U,E,para...])
    save_result(U,ϵsσ,nσ,result,data_dir)
end

saveData(para_list,"$(data_dir)/para_list_check.dat")
data=[(0.0,0.25),[(U,load_result(U,nσ,data_dir)[end]) for U in 1.0:ΔU:10.0]...]
saveData(data,"$(data_dir)/U_d_ns_$(nσ[1])_$(nσ[2]).dat")

# so we find there is problem when U=4.65 ->U=4.70
# we can plot the energy, which we could see the problem (the energy is spuriously low after U>=4.70).
# we now check how the parameters go so we can identify the problem
# we can see the problem is g012
# we change the cosntraint and see




# we do it again

ΔU=0.05
para=[-0.17,0.7,0.38]
para_list=[]
for U in 1.0:ΔU:10.0
    para,result=solve_vdat(U,ϵsσ,nσ,para;option="half-filling")
    E,Ek,Eloc,nkσ,d=result
    # we store the U, E, para, and use E to check any suspicious parameters
    push!(para_list,[U,E,para...])
    save_result(U,ϵsσ,nσ,result,data_dir)
end

saveData(para_list,"$(data_dir)/para_list_check.dat")
data=[(0.0,0.25),[(U,load_result(U,nσ,data_dir)[end]) for U in 1.0:ΔU:10.0]...]
saveData(data,"$(data_dir)/U_d_ns_$(nσ[1])_$(nσ[2]).dat")

# now, it seems that there is no problem
