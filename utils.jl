# for v3, some auxillary function to compute Γασ and Vη

const __global__Γασ__=Dict()

"""
we run
update__global__Γασ__(2)
update__global__Γασ__(4)
as default
# now, we try the five band model
update__global__Γασ__(10)
"""
function update__global__Γασ__(N_spin_orbital)
    global __global__Γασ__
    __global__Γασ__[N_spin_orbital]=[reverse(digits(Γ-1,base=2,pad=N_spin_orbital)) for Γ in 1:2^N_spin_orbital]
end

"""
one should call the update version first
"""
function cal_Γασ(Γ,N_spin_orbital)
    global __global__Γασ__
    __global__Γασ__[N_spin_orbital][Γ]
end

# ensure we compute some common N_spin_orbital
# update__global__Γασ__(2)
# update__global__Γασ__(4)

"""
now, we need a conventon to generate all idx (combination with n>=2)
i.e for N_spin_orbital=4
we have 
[1,2] ...
[1,2,3] ...
[1,2,3,4]..
The order does not matter, as long as we are consistent through the calculation
ηToIdx=gene_idx(N_spin_orbital)

"""
function gene_idx(N_spin_orbital)
    [idx for idx in combinations(1:N_spin_orbital) if length(idx)>=2]
end

"""
idx=[i,j,..k] the collection of sites the density correlation evolves
"""
function cal_Vη(N_spin_orbital,idx)
    N_Γ=2^N_spin_orbital
    # Γ=3                         # some test
    # we should reverse the order to agree with our definition and direct product
    Vη=zeros(N_Γ)
    for Γ in 1:N_Γ
        # Γασ=reverse(digits(Γ-1,base=2,pad=N_spin_orbital))
        Γασ=cal_Γασ(Γ,N_spin_orbital)
        Vη[Γ]=prod((Γασ.-0.5)[idx])
    end
    Vη
end


function gene_VΓη_ηToIdx(N_spin_orbital)
    ηToIdx=gene_idx(N_spin_orbital)
    N_Γ=2^N_spin_orbital
    N_η=length(ηToIdx)
    VΓη=zeros(N_Γ,N_η)
    for i in 1:N_η
        VΓη[:,i]=cal_Vη(N_spin_orbital,ηToIdx[i])
    end
    VΓη,ηToIdx
end

const __global_VΓη_ηToIdx__=Dict()

"""
update__global_VΓη_ηToIdx__(10)
"""
function update__global_VΓη_ηToIdx__(N_spin_orbital)
    __global_VΓη_ηToIdx__[N_spin_orbital]=gene_VΓη_ηToIdx(N_spin_orbital)
end

function cal_VΓη_ηToIdx(N_spin_orbital)
    VΓη,ηToIdx=__global_VΓη_ηToIdx__[N_spin_orbital]
end


# update__global_VΓη_ηToIdx__(2)
# update__global_VΓη_ηToIdx__(4)

# function to extend operator to full space, pmatwασ is normally idenity matrix, here, we assume it is general, so pmatwασ,Xmatασ are vectors of matrices

"""
extend Xmatασ[idx] to the full space, by assume it is in idx spin orbital
"""
function cal_Xmatfull(pmatwασ,Xmatασ,idx)
    kron(pmatwασ[1:(idx-1)]...,Xmatασ[idx],pmatwασ[(idx+1):end]...)
end

"""
extend Xmatασ[idx1]*Xmatασ[idx2] to the full space, assuming idx1<idx2
"""
function cal_Xmatfull(pmatwασ,Xmatασ,idx1,idx2)
    kron(pmatwασ[1:(idx1-1)]...,Xmatασ[idx1],pmatwασ[(idx1+1):(idx2-1)]...,Xmatασ[idx2],pmatwασ[(idx2+1):end]...)
end


# these are functions for constraint k

function norm1(k)
    sum(abs.(k))
end

# for derivaties
Base.convert(::Type{Int64},x::Nothing)=0

"""
k0 is the unit vector, knorm is the norm of k
"""
function cal_k0_knorm(k)
    knorm=norm1(k)
    k0=k/knorm
    k0,knorm
end

"""
used to constraint k
# it is acutally the scale, and with the new scheme, this is the only function we need
"""
function cal_maxnorm(w02,k0)
    kmax=minimum([-w02[i]/k0[i] for i in 1:length(w02)  if k0[i]<0])
end

"""
the main point is to ensure knorm'∈[0,kmax]; Explore other possibilities
there may be problem as it is a periodic function in knorm
"""
function regulate_knorm_1(knorm,kmax)
    kmax*(1-1e-4)*abs(sin(knorm/kmax))
end


"""
for the easy of benchmarking, we set the direct cutoff
"""
function regulate_knorm_2(knorm,kmax)
    if(knorm>kmax-1e-4)
        kmax-1e-4
    else
        knorm
    end    
end

"""
regulate_knorm_linear(0.0,0.2)
"""
function regulate_knorm_linear(knorm,kmax)
    if(knorm>kmax*(1-1e-4))
        kmax*(1-1e-4)*exp(-(knorm-kmax*(1-1e-4))/kmax)
    else
        knorm
    end    
end

"""
tanh(10)
regulate_knorm_3(2,1)
"""
function regulate_knorm_3(knorm,kmax)
    if(knorm<kmax)
        kmax*(1-1e-4)*(1-(knorm-kmax)^2/kmax^2)
    else
        kmax*(1-1e-4)/(1+(knorm-kmax)^2)
    end    
end

"""
compute for g11, g12 etc
"""
function cal_Xασ(w,pmatwασ,Xmatwασ)
    N_spin_orbital=length(pmatwασ)
    [expt(w,cal_Xmatfull(pmatwασ,Xmatwασ,i)) for i in 1:N_spin_orbital]
end

"""
using symmetry to speed up the calculation
"""
function cal_Xασ(w,pmatwασ,Xmatwασ,symmetry)
    N_spin_orbital=length(pmatwασ)
    result=zeros(N_spin_orbital)
    for (idx,term) in enumerate(symmetry)
        result[term].=expt(w,cal_Xmatfull(pmatwασ,Xmatwασ,symmetry[idx][1]))
    end
    result                      
    # [expt(w,cal_Xmatfull(pmatwασ,Xmatwασ,i)) for i in 1:N_spin_orbital]
end



"""
compute expetation values, w^2 is assumed to be normalizd,
i.e, compute from cal_w
"""
function expt(w,Xmatfull)
    dot(w,Xmatfull*w)
end

function cal_Slocασ(nασ,G12ασ,g12ασ)
    cal_sloc11sloc12.(nασ,G12ασ,g12ασ)
end

# some function to update G12 using the fixed point method

function cal_rασ(Slocασ)
    [cal_rIns11s12(sloc11,sloc12)  for (sloc11,sloc12) in Slocασ]
end

"""
cασ=cal_cασ(nασ,Δασ,rασ)
"""
function cal_cασ(nασ,Δασ,rασ)
    cal_chalf.(nασ,Δασ,rασ)
end


function cal_G12ασ(nασ,Δασ,rασ,cασ)
    cal_g012Incr.(nασ,Δασ,rασ,cασ)
end

"""
restrict the range of Δασ
# 
"""
function restrict_Δασ(Δασ,nασ;cutoff=1e-4)
    # Δασ=min.(Δασ,nασ .- 1e-4)
    # Δασ=max.(Δασ, 1e-4)    
    clamp.(Δασ,cutoff,min.(nασ .- cutoff,1.0 .- nασ .- cutoff))
end

function restrict_nασ(nασ;cutoff=1e-4)
    # Δασ=min.(Δασ,nασ .- 1e-4)
    # Δασ=max.(Δασ, 1e-4)    
    clamp.(nασ,cutoff,1.0 - cutoff)
end

function restrict_Slocασ(Slocασ;cutoff=1e-5)
    N_spin_orbital=length(Slocασ)
    Slocασ=[max.(Slocασ[i],cutoff) for i in 1:N_spin_orbital]
end


"""
there is regulation
"""
function cal_Δασ(g12ασ,Slocασ,nασ)
    N_spin_orbital=length(g12ασ)
    # [ cal_delta(g12ασ[idx],Slocασ[idx]...) for idx in 1:N_spin_orbital]
    # to fix the AD problem
    [ cal_delta(g12ασ[idx],Slocασ[idx][1],Slocασ[idx][2]) for idx in 1:N_spin_orbital]
end

function loadAsSpinOrbital(filename_base,qauntity_name,N_spin_orbital)
    [loadData("$(filename_base)_$(qauntity_name)_spin_orb_$(i).dat") for i in 1:N_spin_orbital]
end


# functions to benchmark
"""
U=2.0
"""
function load_para_two_band_half(U)
    old_data_dir="./two_band_degenerate_inf_half"
    old_filename_base=replace("$(old_data_dir)/U_$(U)_n_$(nασ)",","=>"_"," "=>"","["=>"","]"=>"")
    G12ασ=[i[1] for i in loadAsSpinOrbital(old_filename_base,"G12",N_spin_orbital)]
    βασ=[reshape(i,2) for i in loadAsSpinOrbital(old_filename_base,"beta",N_spin_orbital)]
    αασ=[reshape(i,2) for i in loadAsSpinOrbital(old_filename_base,"alpha",N_spin_orbital)]
    Δαασ=[ (αασ_[1]-αασ_[2]) for αασ_ in αασ]
    w= reshape(loadData("$(old_filename_base)_w.dat"),:)
    w_check= reshape(loadData("$(old_filename_base)_w.dat"),:)
    w02=cal_w02(cal_neffασ(nασ,G12ασ))
    VΓη,ηToIdx=__global_VΓη_ηToIdx__[N_spin_orbital]
    x=pinv(VΓη)*(w.^2-w02)
    # check=cal_w_scaled(x,nασ,G12ασ)-w_check
    return G12ασ,x,Δαασ,βασ
end
