# this implements the single particle density matrix formalism of SCDA

using LinearAlgebra
using Statistics
using Combinatorics
using Roots
using Zygote

include("./include_gene_code.jl")
# processing the band structure
include("./load_band.jl")
include("./utils.jl")

# we first process momentum part, we write them to a seperate file
# the main function is
# Δασ,Aασ_below,Aασ_above,K=cal_Δ_Abelow_Aabove_K(Δαασ,βασ,eασ,nασ) 
# which computes the change transfer and change flutuation for a given set of parameters
include("./vdat_sdm_momentum.jl")

# then we need to deal with local part, and self-consistency of G12



