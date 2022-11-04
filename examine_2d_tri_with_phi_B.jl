include("./vdat.jl")

phi_B_data="./2d_phi_B_data/"
mkdir("./data_2d_tri_with_phi_B")
function gene_density_list(ns_low;digits=2)
    trunc_fn=n->round(n,digits=digits)
    [trunc_fn.([n_,1-n_]) for n_ in ns_low]
end

tag="1_3"
tag="1_2"
tag="1_5"
tag="1_4"
tag="1_10"
data_dir="./data_2d_tri_with_phi_B/phi_B_$(tag)"
mkdir(data_dir)

saveData(sort((x->convert(Float64,x)).(loadData("$(phi_B_data)/phi_B_$(tag).txt")[:,2][2:end])),"$(phi_B_data)/phi_B_$(tag)_process.dat")
e_fn=gene_spline_band("$(phi_B_data)/phi_B_$(tag)_process.dat")

nσs=0.5:-0.001:0.01
nσ_list=gene_density_list(nσs;digits=3)

for tag in ["1_3","1_2","1_5","1_4","1_10"]
    e_fn=gene_spline_band("$(phi_B_data)/phi_B_$(tag)_process.dat")
    data_dir="./data_2d_tri_with_phi_B/phi_B_$(tag)"
    try
        for nσ in nσ_list
            print("nσ is $(nσ) tag is $(tag)\n")
            ϵsσ=[gene_ϵs(e_fn,nσ[1]),gene_ϵs(e_fn,nσ[2])]
            ϵfσ=e_fn.(nσ)
            para=[-0.02,0.6,0.6,0.38]
            for U in 0.1:0.1:20.0
                para,result=solve_vdat(U,ϵsσ,nσ,para;option="half-filling-magnetization")
                save_result(U,ϵsσ,nσ,ϵfσ,result,data_dir)
            end
        end
    catch
    end    
end


# there are some problem for large polarization
nσ=[0.5,0.5]
nσ=[0.4,0.6]
nσ=[0.3,0.7]
nσ=[0.2,0.8]
U=0.1
process_data_dir="./data_2d_tri_with_phi_B_processed"
mkdir(process_data_dir)
for tag  in [ "1_10", "1_2","1_5","1_4","1_3"]
    data_dir="./data_2d_tri_with_phi_B/phi_B_$(tag)"
    saveData([(U,load_result(U,nσ,data_dir)...) for U in 0.1:0.1:20.0],"$(process_data_dir)/U_E_etc_ns_$(nσ[1])_$(nσ[2])_tag_$(tag).dat")
end

U=1.0
tag="1_2"
tag="1_3"
tag="1_4"
tag="1_5"
tag="1_10"
for tag in [ "1_10", "1_2","1_5","1_4","1_3"]
    data_dir="./data_2d_tri_with_phi_B/phi_B_$(tag)"
    for U in 1.0:1.0:20.0
        result=[]
        for nσ in nσ_list
            try
                data=load_result(U,nσ,data_dir)
                push!(result, [nσ[1],data...])
            catch
            end
        end
        saveData(result,"$(process_data_dir)/E_etc_vs_ns_U_$(U)_tag_$(tag).dat")
    end
end

U=1.0
tag="1_3"

for tag in [ "1_10", "1_2","1_5","1_4","1_3"]
    for U in 1.0:1.0:20.0
        data=loadData("$(process_data_dir)/E_etc_vs_ns_U_$(U)_tag_$(tag).dat")
        n=data[:,1]
        E=data[:,2]
        M=1.0 .-2.0*n
        fn_E_M=Spline1D(M,E,s=1e-4)
        saveData(hcat(derivative(fn_E_M,M),M),"$(process_data_dir)/M_B_U_$(U)_tag_$(tag).dat")
    end
end

# we can also try use the Maxwell construction

# here, we generate the density of state
tag="1_10"    
e_p_fn=gene_spline_band("$(phi_B_data)/phi_B_$(tag)_process.dat")

# ps=linspace(0.0,1.0,100)
# [(e_p_fn(p_),1/derivative(e_p_fn,p_)) for p_ in ps]

# we compute the K1 and K3


c0=0.5-sqrt(3)/3
K1,K2,K3,K4=[mean(es_half.^n)*0.5 for n in 1:4]
# so we need to change the sign?
beta=K1/c0-c0*K3/2/K1^2
beta*(1-3*K2^2/(3*(K2^2+K4)-2*K2*beta^2))
#
#
"""
β=12
"""
function cal_c0(β,es_half)
    mean(es_half./sqrt.(es_half.^2 .+β^2))*0.5
end


function cal_K(β,es_half)
    -mean(es_half.^2 ./sqrt.(es_half.^2 .+β^2))
end

function cal_A(β,es_half)
    mean(β ./sqrt.(es_half.^2 .+β^2))
end

function cal_d(β,es_half)
    A=cal_A(β,es_half)
    0.25*(1-A^4)
end

function cal_Uc(βstar,es_half)
    -gradient(β->cal_K(β,es_half),βstar)[1]/gradient(β->cal_d(β,es_half),βstar)[1]
end

using Flux
tag="1_10"    
tag="1_5"    
tag="1_4"    
tag="1_3"    
tag="1_2"    

# this is the correct version
e_fn=gene_spline_band("$(phi_B_data)/phi_B_$(tag)_process.dat")
es_half=gene_ϵs(e_fn,0.5)[1]
K1,K2,K3,K4,K5,K6=[mean(es_half.^n)*0.5 for n in 1:6]
Uc_0=βstar0=K1/c0
βstar1=K1/c0-c0*K3/2/K1^2
Uc_1=K1/c0-c0*(K3-6*K1*K2)/2/K1^2
βstar=find_zero(β_->cal_c0(β_,es_half)-c0,12,Order0())
Uc=cal_Uc(βstar,es_half)
K1,K2,K3,βstar0,Uc_0,βstar1,Uc_1,βstar,Uc

# so the first step of expansion is right
# but cal_Uc is more challenging
beta=12
cal_K(beta,es_half)
cal_K_1(beta)
# this form is good enough
cal_K_1(beta)=-2*K2/beta+K4/beta^3-3*K6/beta^5/4
cal_K_1(beta)=-2*K2/beta+K4/beta^3
cal_A(beta,es_half)
cal_A_1(beta)
cal_A_1(beta)=1-K2/beta^2+3/4*K4/beta^4-5/8*K6/beta^6
cal_A_1(beta)=1-K2/beta^2+3/4*K4/beta^4

cal_d(beta,es_half)
cal_d_1(beta)
cal_d_1(beta)=0.25*(1-cal_A_1(beta)^4)
cal_d_1(beta)=K2/beta^2-3/4*(K4+2*K2^2)/beta^4
beta=12
cal_Uc(beta,es_half)
cal_Uc_1(beta)
cal_Uc_1_alter(beta)
cal_Uc_1_alter(beta)=beta*(1-6*K2^2/(-2*K2*beta^2+6*K2^2+3*K4))
cal_Uc_1(beta)=-gradient(β->cal_K_1(β),beta)[1]/gradient(β->cal_d_1(β),beta)[1]


