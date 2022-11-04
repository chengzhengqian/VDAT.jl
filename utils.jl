# for v3, some auxillary function to compute Γασ and Vη

const __global__Γασ__=Dict()

"""
we run
update__global__Γασ__(2)
update__global__Γασ__(4)
as default
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
update__global__Γασ__(2)
update__global__Γασ__(4)

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

function update__global_VΓη_ηToIdx__(N_spin_orbital)
    __global_VΓη_ηToIdx__[N_spin_orbital]=gene_VΓη_ηToIdx(N_spin_orbital)
end

function cal_VΓη_ηToIdx(N_spin_orbital)
    VΓη,ηToIdx=__global_VΓη_ηToIdx__[N_spin_orbital]
end


update__global_VΓη_ηToIdx__(2)
update__global_VΓη_ηToIdx__(4)

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
"""
function cal_maxnorm(w02,k0)
    kmax=minimum([-w02[i]/k0[i] for i in 1:length(w02)  if k0[i]<0])
end

"""
the main point is to ensure knorm'∈[0,kmax]; Explore other possibilities
"""
function regulate_knorm_1(knorm,kmax)
    kmax*(1-1e-4)*abs(sin(knorm/kmax))
end

"""
compute expetation values, w^2 is assumed to be normalizd,
i.e, compute from cal_w
"""
function expt(w,Xmatfull)
    dot(w,Xmatfull*w)
end

