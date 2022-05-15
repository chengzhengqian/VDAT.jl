# using CZQUtils
using Dierckx
# from CZQUtils, so one does not need to download that package
using DelimitedFiles
linspace(start,stop,length)=range(start,stop=stop,length=length)
saveData(data,filename)= open(filename,"w") do io writedlm(io,data) end
loadData(filename)=readdlm(filename)


"""
inf_fn=gene_spline_band("./es_inf.dat")
d2_fn=gene_spline_band("./es_2d.dat")
"""
function gene_spline_band(filename)
    data=reshape(loadData(filename),:)
    N_sample=size(data)[1]
    index=collect(linspace(0,1,N_sample))
    return Spline1D(index,data)
end

"""
e_fn=inf_fn
n=0.4
ϵs=gene_ϵs(e_fn,nσ[1])
nσ
ϵsσ=[gene_ϵs(e_fn,nσ[1]),gene_ϵs(e_fn,nσ[2])]
"""
function gene_ϵs(e_fn,n;N_samples=40,N_minimal=4)
    index_points_below=max(trunc(Int64,n*N_samples),N_minimal)
    index_points_above=max(trunc(Int64,(1-n)*N_samples),N_minimal)
    index_below=collect(linspace(0,n,index_points_below+1))
    energy_below=[(integrate(e_fn,index_below[i],index_below[i+1]))/(index_below[i+1]-index_below[i]) for i in 1:index_points_below]
    index_above=collect(linspace(n,1,index_points_above+1))
    energy_above=[(integrate(e_fn,index_above[i],index_above[i+1]))/(index_above[i+1]-index_above[i]) for i in 1:index_points_above]
    [energy_below,energy_above]
end
