# use SCDA to solve the one band Hubbard model
using Statistics
using Roots
using LinearAlgebra
using Optim
include("./load_band.jl")

# first, we start with local part
# non-interacting reference value for w
#density for spin up and down
# nσ=[0.4,0.6]
# # variable parameter, G012σ
# g012σ=[0.3,0.4]
"""
non-interacting value for w^2*4
cal_ω(nσ,g012σ)
For this two parameters
nσ=[0.4,0.6]
g012σ=[0.3,0.4]
Mathematic expresion gives
0.98,0.66,1.40,0.94
# we check they agrees
ω=cal_ω(nσ,g012σ)
"""
function cal_ω(nσ,g012σ)
    nup,ndn=nσ
    g012up,g012dn=g012σ
    Δnup=2*nup-1
    Δndn=2*ndn-1
    tup=4*g012up^2+1
    tdn=4*g012dn^2+1
    ω1=(1+4*Δnup*g012up/tup)*(1+4*Δndn*g012dn/tdn)
    ω2=(1+4*Δnup*g012up/tup)*(1-4*Δndn*g012dn/tdn)
    ω3=(1-4*Δnup*g012up/tup)*(1+4*Δndn*g012dn/tdn)
    ω4=(1-4*Δnup*g012up/tup)*(1-4*Δndn*g012dn/tdn)
    [ω1,ω2,ω3,ω4]
end

# we could check the range of Δu
"""
Δu=-0.1
we get w from Mathematica as
0.47,0.436,0.613,0.460
w=cal_w(ω,Δu)
one should ensure Δu is real, 
so Δu in 
max(-ω1,-ω4)<Δu<min(ω2,ω3)
this is constraint by l, which is in [-1,1]
"""
function cal_w(ω,Δu)
    0.5*sqrt.(ω+[Δu,-Δu,-Δu,Δu])
end

"""
compute the 12 blocks for local green function
with the previous values
0.28803, 0.387357
g12σ=cal_g12σ(w,g012σ)
# and we verify the expression
"""
function cal_g12σ(w,g012σ)
    f12=w[1]*w[2]
    f13=w[1]*w[3]
    f24=w[2]*w[4]
    f34=w[3]*w[4]
    tσ=[2*(f13+f24),2*(f12+f34)]
    (4*(tσ.+1).*(g012σ.^2) .+tσ.-1)./(8*g012σ)
end


"""
compute the self-energy, for g012σ (g011σ=1/2..)and g12σ, n1σ (i.e g11σ)
In calculations, g11σ is just n1σ by construction
s11σ
1.35019,0.779547
s12σ
0.012429,0.00535501
checked
sσ=s11σ,s12σ=cal_sσ(nσ,g012σ,g12σ);
"""
function cal_sσ(nσ,g012σ,g12σ)
    Aσ=(1.0 .-nσ).*(nσ)
    factor=(4*g012σ.^2 .+ 1.0).*(g12σ.^2 + nσ.^2)
    s11σ=4*g012σ.^2 .* (g12σ.^2 .- Aσ) + 4*g12σ.*g012σ - g12σ.^2 + Aσ
    s12σ=g12σ.*(4*g012σ.*(g012σ.-g12σ).-1.0)+4*Aσ.*g012σ
    if(s12σ[1]<0 || s12σ[2]<0 )
        print("g012σ: $(g012σ) g12σ: $(g12σ)")
        error("s12σ has negative element")
    end    
    [s11σ./factor,s12σ./factor]
end

"""
Δσ=cal_Δσ(sσ,g12σ)
"""
function cal_Δσ(sσ,g12σ)
    s11σ,s12σ=sσ
    g12σ.*s12σ./s11σ
end

"""
A single band is parametrize by 4 numbers
α<,β<,α>,β>
α could be solved from 
So we need to sample the k points
check how the ..
we parametrize Δu as l
l=-0.99999
m=0.49999
check_Δs([0.2,0.8],[m,m],l)
"""
function check_Δs(nσ,g012σ,l)
    ω=cal_ω(nσ,g012σ)
    if(l>0)
        Δu=l*min(ω[2],ω[3])
    else
        Δu=-l*max(-ω[1],-ω[4])
    end
    w=cal_w(ω,Δu)
    g12σ=cal_g12σ(w,g012σ)
    sσ=cal_sσ(nσ,g012σ,g12σ)
    cal_Δσ(sσ,g12σ),Δu,w,g12σ,sσ
end

"""
this is for either below and above the fermi surface
we test for the region below the fermi surface first.
nk_mean=1-Δσ[1]/nσ[1]
β=0.1
ϵs_=ϵs[1]
α=0.3
for a particular spin orbital and region
"""
function cal_nk_A(nk_mean,β_,ϵs_)
    function compute_nk(α)
        nk_fn=ϵ->0.5*(1+ (α-ϵ)/sqrt((α-ϵ)^2+β_^2))
        nk_fn.(ϵs_)
    end
    function compute_nk_mean(α)
        mean(compute_nk(α))
    end
    local α
    try        
        α=find_zero(α->(compute_nk_mean(α)-nk_mean),0.0,Order0())
    catch
        print("$(nk_mean),$(β_),$(ϵs_)")
        error("can't find root!")
    end    
    nk=compute_nk(α)
    A=mean(sqrt.(nk.*(1 .- nk)))
    nk,A
end

"""
for both below and above for a given spin
nσ_=nσ[1]
Δσ_=Δσ[1]
β=[0.1,0.1]
nk,A=cal_nk_A_full(nσ_,Δσ_,β,ϵs)
"""
function cal_nk_A_full(nσ_,Δσ_,β,ϵs)
    Δσ_max=(1-nσ_)*nσ_
    if(Δσ_>Δσ_max)
        Δσ_=Δσ_max
    end
    if(Δσ_<0)
        Δσ_=0.0001
    end    
    nk_mean_below=1-Δσ_/nσ_
    β_below=β[1]
    ϵs_below=ϵs[1]
    nk_below,A_below=cal_nk_A(nk_mean_below,β_below,ϵs_below)
    nk_mean_above=Δσ_/(1-nσ_)
    β_above=β[2]
    ϵs_above=ϵs[2]
    nk_above,A_above=cal_nk_A(nk_mean_above,β_above,ϵs_above)
    [nk_below,nk_above],[A_below,A_above]
end

"""
# frist, for each spin, second, for each region
βσ=[β,β]
ϵsσ has the same structure
nkσ,A_below_σ,A_above_σ=cal_nk_A_σ(nσ,Δσ,βσ,ϵsσ)
"""
function cal_nk_A_σ(nσ,Δσ,βσ,ϵsσ)
    nk_up,A_up=cal_nk_A_full(nσ[1],Δσ[1],βσ[1],ϵsσ[1])
    nk_dn,A_dn=cal_nk_A_full(nσ[2],Δσ[2],βσ[2],ϵsσ[2])
    nkσ=[nk_up,nk_dn]
    A_below_σ=[A_up[1],A_dn[1]]
    A_above_σ=[A_up[2],A_dn[2]]
    nkσ,A_below_σ,A_above_σ
end

"""
check for some test input
cal_g_other_components([[0.8],[0.6]],[0.1],[0.2],[0.4])
we should get 0.1239, 0.144591,-0.041318,-0.185903
g13σ,g23σ,g31σ,g32σ=cal_g_other_components(sσ,A_below_σ,A_above_σ,nσ)
"""
function cal_g_other_components(sσ,A_below_σ,A_above_σ,nσ)
    s11σ,s12σ=sσ
    ssqrtdetσ=sqrt.(( s11σ.^2)+(s12σ.^2))
    local s12sqrtσ
    try
        s12sqrtσ=sqrt.(s12σ)
    catch
        print("s12σ is $(s12σ)")
    end
    g13σ=(1.0 .- nσ) .* s11σ .* A_above_σ  ./s12sqrtσ  ./ ssqrtdetσ
    g23σ= nσ.* A_below_σ ./ s12sqrtσ +(1.0 .- nσ) .* A_above_σ  .* s12sqrtσ  ./ ssqrtdetσ
    g31σ=-nσ.*s11σ./s12sqrtσ.*A_below_σ
    g32σ=-(1.0 .- nσ).*ssqrtdetσ ./ s12sqrtσ .* A_above_σ - nσ.*s12sqrtσ.*A_below_σ
    g13σ,g23σ,g31σ,g32σ
end

"""
for the code generated from Mathematica
"""
function Power(x,y)
    x^y
end

"""
we generate them from mathematica
g11,g22,g33 just n, g21=-g12
g012=g012σ[1]
g11=nσ[1]
g12=g12σ[1]
g13=g13σ[1]
g23=g23σ[1]
g31=g31σ[1]
g32=g32σ[1]
s11=s11σ[1]
s12=s12σ[1]
g013,g023,g031,g032,g033=cal_g0_other_components_given_spin(g012,g11,g12,g13,g23,g31,g32)
g0=[ 0.5 g012 g013 ;
         (-g012) 0.5 g023;
        g031  g032 g033
    ]
g=[ g11 g12 g13 ;
     (-g12) g11 g23;
    g31   g32  g11
 ]
sloc=[ s11 s12 0.0;
            (-s12) s11 0.0;
            0.0   0.0 1.0
         ]
id3=Matrix(I,3,3)
(inv(g0)-id3)*sloc-(inv(g)-id3)
"""
function cal_g0_other_components_given_spin(g012,g11,g12,g13,g23,g31,g32)
    g013=(-(g12*(-2*g012*g13+g23))+g11*(g13+2*g012*g23))/(2.0*(Power(g11,2)+Power(g12,2)))
    g023=(g11*(-2*g012*g13+g23)+g12*(g13+2*g012*g23))/(2.0*(Power(g11,2)+Power(g12,2)))
    g031=((1-g11+2*g012*g12)*g31-(2*g012*(-1+g11)+g12)*g32)/(2.0*(Power(-1+g11,2)+Power(g12,2)))
    g032=(-((-1+g11)*(-2*g012*g31+g32))+g12*(g31+2*g012*g32))/(2.0*(Power(-1+g11,2)+Power(g12,2)))
    g033=(-4*Power(g11,4)+2*Power(g11,5)+Power(g11,3)*(2+4*Power(g12,2)-2*g13*g31-2*g23*g32)+Power(g11,2)*(-4*Power(g12,2)+g23*(2*g012*g31+3*g32)+g13*(3*g31-2*g012*g32)+2*g12*(g23*g31-g13*g32))+g11*(2*Power(g12,4)-g23*(2*g012*g31+g32)-g13*(g31-2*g012*g32)-2*Power(g12,2)*(-1+g13*g31+g23*g32)+g12*(2*g13*(2*g012*g31+g32)-2*g23*(g31-2*g012*g32)))+g12*(-(g13*(2*g012*g31+g32))+g23*(g31-2*g012*g32)+2*Power(g12,2)*(g23*g31-g13*g32)+g12*(g23*(-2*g012*g31+g32)+g13*(g31+2*g012*g32))))/(2.0*(Power(-1+g11,2)+Power(g12,2))*(Power(g11,2)+Power(g12,2)))
    g013,g023,g031,g032,g033
end

# finally, we just need to compute double occupancy
"""
g013up,g023up,g031up,g032up,g033up=cal_g0_other_components_given_spin(g012σ[1],nσ[1],g12σ[1],g13σ[1],g23σ[1],g31σ[1],g32σ[1])
g013dn,g023dn,g031dn,g032dn,g033dn=cal_g0_other_components_given_spin(g012σ[2],nσ[2],g12σ[2],g13σ[2],g23σ[2],g31σ[2],g32σ[2])
g0othersup=cal_g0_other_components_given_spin(g012σ[1],nσ[1],g12σ[1],g13σ[1],g23σ[1],g31σ[1],g32σ[1])
g0othersdn=cal_g0_other_components_given_spin(g012σ[2],nσ[2],g12σ[2],g13σ[2],g23σ[2],g31σ[2],g32σ[2])
dq=cal_dq(g012σ,g0othersup...,g0othersdn...)
matrix in q, we need to use W to transform it to w
"""
function cal_dq(g012σ,g013up,g023up,g031up,g032up,g033up,g013dn,g023dn,g031dn,g032dn,g033dn)
    g012up,g012dn=g012σ
    #d11
    d11=g033dn*g033up
    #d12
    d12=-((g013dn*g031dn+g023dn*g032dn)*g033up)/(2.0*g012dn)
    #d13
    d13=-((g013up*g031up+g023up*g032up)*g033dn)/(2.0*g012up)
    #d14
    d14=(g013dn*g013up*g031dn*g031up+g023dn*g023up*g032dn*g032up)/(2.0*g012dn*g012up)
    #d22
    d22=((g023dn*g031dn-g013dn*g032dn+g012dn*g033dn)*g033up)/g012dn
    #d23
    d23=(g013up*g023dn*g031up*g032dn+g013dn*g023up*g031dn*g032up)/(2.0*g012dn*g012up)
    #d24
    d24=-((g013up*g031up+g023up*g032up)*(g023dn*g031dn-g013dn*g032dn+g012dn*g033dn))/(2.0*g012dn*g012up)
    #d33
    d33=(g033dn*(g023up*g031up-g013up*g032up+g012up*g033up))/g012up
    #d34
    d34=-((g013dn*g031dn+g023dn*g032dn)*(g023up*g031up-g013up*g032up+g012up*g033up))/(2.0*g012dn*g012up)
    #d44
    d44=((g023dn*g031dn-g013dn*g032dn+g012dn*g033dn)*(g023up*g031up-g013up*g032up+g012up*g033up))/(g012dn*g012up)
    dq=[
        d11 d12 d13 d14;
        d12 d22 d23 d24;
        d13 d23 d33 d34;
        d14 d24 d34 d44
    ]
end

# to connect with the form in w, we need to construct W matrix
"""
dw=cal_dw(g012σ,g0othersup...,g0othersdn...)
"""
function cal_dw(g012σ,g013up,g023up,g031up,g032up,g033up,g013dn,g023dn,g031dn,g032dn,g033dn)
    dq=cal_dq(g012σ,g013up,g023up,g031up,g032up,g033up,g013dn,g023dn,g031dn,g032dn,g033dn)
    W=[
        0.5 0.5 0.5 0.5 ;
        0.5 (-0.5) 0.5 (-0.5);
        0.5 0.5  (-0.5) (-0.5);
        0.5 (-0.5) (-0.5) (0.5)
    ]
    W*dq*W
end

"""
U=1.0
l is the effective parameter in [-1,1], -1  corresponding the fulling projetion of d , 1 is the opposite limit, 0 is the non-interacting case.
l=-0.1
"""
function compute_GN3(ϵsσ,U,nσ,l,βσ,g012σ)
    ω=cal_ω(nσ,g012σ)
    if(l>0)
        Δu=l*min(ω[2],ω[3])
    else
        Δu=-l*max(-ω[1],-ω[4])
    end
    w=cal_w(ω,Δu)
    g12σ=cal_g12σ(w,g012σ)
    sσ=cal_sσ(nσ,g012σ,g12σ)
    Δσ=cal_Δσ(sσ,g12σ)
    nkσ,A_below_σ,A_above_σ=cal_nk_A_σ(nσ,Δσ,βσ,ϵsσ)
    g13σ,g23σ,g31σ,g32σ=cal_g_other_components(sσ,A_below_σ,A_above_σ,nσ)
    g0othersup=cal_g0_other_components_given_spin(g012σ[1],nσ[1],g12σ[1],g13σ[1],g23σ[1],g31σ[1],g32σ[1])
    g0othersdn=cal_g0_other_components_given_spin(g012σ[2],nσ[2],g12σ[2],g13σ[2],g23σ[2],g31σ[2],g32σ[2])
    dw=cal_dw(g012σ,g0othersup...,g0othersdn...)
    d=transpose(w)*dw*w
    i=1
    Ek=sum([( nσ[i]*mean(nkσ[i][1].*ϵsσ[i][1]) +(1-nσ[i])*mean(nkσ[i][2].*ϵsσ[i][2]) ) for i in 1:2  ])
    Eloc=U*d
    E=Ek+Eloc
    E,Ek,Eloc,nkσ,d
end

# nσ=[0.5,0.5]
# ϵsσ=[gene_ϵs(e_fn,nσ[1]),gene_ϵs(e_fn,nσ[2])]
# E,Ek,Eloc,nkσ,d=compute_GN3(ϵsσ,U,nσ,-0.1,[[0.0001,0.0001],[0.1,0.1]],[0.5,0.5]);d,Ek

# we first test for half-filling
# nσhalf=[0.5,0.5]
# ϵsσhafl=[gene_ϵs(e_fn,nσ[1]),gene_ϵs(e_fn,nσ[2])]
function cal_energy_half_filling(U,ϵsσ,nσ,l,β,g012)
    l=constraint_l(l)
    g012=constraint_g012(g012)
    compute_GN3(ϵsσ,U,nσ,l,[[β,β],[β,β]],[g012,g012])
end


    

# U=1.0
# para=[-0.1,0.1,0.4]
# U=1.0
# res=optimize(x->cal_energy_half_filling(U,x[1],x[2],x[3])[1],para)
# para=Optim.minimizer(res)
# U=10.0
# res=optimize(x->cal_energy_half_filling(U,x[1],x[2],x[3])[1],para)
# para=Optim.minimizer(res)
# E,Ek,Eloc,nkσ,d=cal_energy_half_filling(U,para...)
# nkσ[1]

# nσ=[0.4,0.6]
# ϵsσ=[gene_ϵs(e_fn,nσ[1]),gene_ϵs(e_fn,nσ[2])]
# E,Ek,Eloc,nkσ,d=compute_GN3(ϵsσ,U,nσ,-0.99,[[0.2,0.2],[0.2,0.2]],[0.3,0.3]);d,Ek
# mean(nkσ[1][1])*nσ[1]+mean(nkσ[1][2])*(1-nσ[1])
function constraint_l(l)
    clamp(l,-0.9999,0.9999)
end
function constraint_g012(g012)
    if(g012>0.2)
        return g012
    else
        g012=0.2*exp(-(0.2-g012))
    end    
end



function cal_energy_half_filling_magnetization(U,ϵsσ,nσ,l,β1,β2,g012)
    l=constraint_l(l)
    g012=constraint_g012(g012)
    E,Ek,Eloc,nkσ,d=compute_GN3(ϵsσ,U,nσ,l,[[β1,β2],[β2,β1]],[g012,g012])
end

# para=[-0.2,0.1,0.1,0.4]
# U=1.0
# U=2.0
# U=10.0
# @time res=optimize(x->cal_energy_half_filling_magnetization(U,x...)[1],para)
# @time res=optimize(x->cal_energy_half_filling_magnetization(U,x...)[1],para, LBFGS())

# para=Optim.minimizer(res)
# E,Ek,Eloc,nkσ,d=cal_energy_half_filling_magnetization(U,para...);d

# nkσ[1]

"""
option can be
half-filling
hafl-filling-magnetization
# todo add more situation
"""
function solve_vdat(U,ϵsσ,nσ,para;option="half-filling")
    if(option=="half-filling")
        cal_energy=cal_energy_half_filling
    elseif(option=="half-filling-magnetization")
        cal_energy=cal_energy_half_filling_magnetization
    else
        error("unsurpported option $(option) ")
    end
    res=optimize(x->cal_energy(U,ϵsσ,nσ,x...)[1],para)
    para=Optim.minimizer(res)
    result=cal_energy(U,ϵsσ,nσ,para...)
    para,result
end



# mkdir("./data")


function save_result(U,ϵsσ,nσ,result,data_dir)
    E,Ek,Eloc,nkσ,d=result
    filename_base="$(data_dir)/U_$(U)_ns_$(nσ[1])_$(nσ[2])"
    filename_energy="$(filename_base)_E_Ek_Eloc_d.dat"
    filename_nk_up="$(filename_base)_nk_spin_up.dat"
    filename_nk_dn="$(filename_base)_nk_spin_dn.dat"
    saveData([E,Ek,Eloc,d],filename_energy)
    nk_up=[nkσ[1][1]...,nkσ[1][2]...]
    ϵs_up=[ϵsσ[1][1]...,ϵsσ[1][2]...]
    saveData(hcat(ϵs_up,nk_up),filename_nk_up)
    nk_dn=[nkσ[2][1]...,nkσ[2][2]...]
    ϵs_dn=[ϵsσ[2][1]...,ϵsσ[2][2]...]
    saveData(hcat(ϵs_dn,nk_dn),filename_nk_dn)
end

function load_result(U,nσ,data_dir)
    filename_base="$(data_dir)/U_$(U)_ns_$(nσ[1])_$(nσ[2])"
    filename_energy="$(filename_base)_E_Ek_Eloc_d.dat"
    loadData(filename_energy)
end


