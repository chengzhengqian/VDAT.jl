include("./vdat_v1.jl")
data_dir="./two_band_degenerate_inf_half"
mkdir(data_dir)
N_spin_orbital=4
nloc=0.5
nloc=0.4
nloc=0.3
U=1.0
para=[0.4,-0.01,0.0,0.3,0.2]
e_fn=gene_spline_band("./es_inf.dat")
es=gene_ϵs(e_fn,nloc)
eασ=[es for _ in 1:N_spin_orbital]
# there are some problem for large U, check the reason
# two band case
for U in 1.0:0.1:10.0
    Ueff=6*U
    res=optimize(para->compute_degenerate_case(nloc,para[1],para[2:4],eασ,Ueff,[para[5],para[5]],N_spin_orbital)[1],para)
    para=res.minimizer
    print("U is $(U) with para $(para)\n")
    βασ=[[para[5],para[5]] for _ in 1:N_spin_orbital]
    g012ασ=para[1]
    nασ=[nloc for _ in 1:N_spin_orbital]
    G12ασ=[g012ασ for _ in 1:N_spin_orbital]
    # x_reduce=para[2:4]
    # compute results
    Etotal,Eloc,Ek,αασ,nk,Δασ,Slocασ,nn12expt,x,w=compute_degenerate_case(nloc,para[1],para[2:4],eασ,Ueff,[para[5],para[5]],N_spin_orbital)
    savdData(Etotal,Eloc,Ek,αασ,nk,Δασ,Slocασ,nn12expt,x,w,nασ,G12ασ,βασ,eασ,U,data_dir)
end

# now, we check for three band
data_dir="./three_band_degenerate_inf_half"
mkdir(data_dir)
N_spin_orbital=6
nloc=0.5
para=[0.4,-0.01,0.0,0.1,0.0,0.03,1.0]
e_fn=gene_spline_band("./es_inf.dat")
es=gene_ϵs(e_fn,nloc)
eασ=[es for _ in 1:N_spin_orbital]
Ueff=15*U
res=optimize(para->compute_degenerate_case(nloc,para[1],para[2:(end-1)],eασ,Ueff,[para[end],para[end]],N_spin_orbital)[1],para)
para=res.minimizer

# for U in 1.0:0.1:10.0
for U in 10.1:0.1:20.0
    # Ueff=6*U
    Ueff=15*U
    res=optimize(para->compute_degenerate_case(nloc,para[1],para[2:(end-1)],eασ,Ueff,[para[end],para[end]],N_spin_orbital)[1],para)
    para=res.minimizer
    print("U is $(U) with para $(para)\n")
    βασ=[[para[5],para[5]] for _ in 1:N_spin_orbital]
    g012ασ=para[1]
    nασ=[nloc for _ in 1:N_spin_orbital]
    G12ασ=[g012ασ for _ in 1:N_spin_orbital]
    # x_reduce=para[2:4]
    # compute results
    Etotal,Eloc,Ek,αασ,nk,Δασ,Slocασ,nn12expt,x,w=compute_degenerate_case(nloc,para[1],para[2:(end-1)],eασ,Ueff,[para[end],para[end]],N_spin_orbital)
    savdData(Etotal,Eloc,Ek,αασ,nk,Δασ,Slocασ,nn12expt,x,w,nασ,G12ασ,βασ,eασ,U,data_dir)
end

# check for 4 band
data_dir="./four_band_degenerate_inf_half"
mkdir(data_dir)
N_spin_orbital=8
nloc=0.5
e_fn=gene_spline_band("./es_inf.dat")
es=gene_ϵs(e_fn,nloc)
eασ=[es for _ in 1:N_spin_orbital]
para=[0.4,-0.01,0.1,-0.002,0.01,1.2]
# we take the zero out
U=1.0
Ueff=28*U
res=optimize(para->compute_degenerate_case(nloc,para[1],[para[2],0.0,para[3],0.0,para[4],0.0,para[5]],eασ,Ueff,[para[end],para[end]],N_spin_orbital)[1],para)
# for U in 1.0:0.1:10.0
# for U in 10.1:0.1:20.0
for U in 20.1:0.1:30.0
    # Ueff=6*U
    # Ueff=15*U
    Ueff=28*U
    res=optimize(para->compute_degenerate_case(nloc,para[1],[para[2],0.0,para[3],0.0,para[4],0.0,para[5]],eασ,Ueff,[para[end],para[end]],N_spin_orbital)[1],para)
    para=res.minimizer
    print("U is $(U) with para $(para)\n")
    βασ=[[para[5],para[5]] for _ in 1:N_spin_orbital]
    g012ασ=para[1]
    nασ=[nloc for _ in 1:N_spin_orbital]
    G12ασ=[g012ασ for _ in 1:N_spin_orbital]
    # x_reduce=para[2:4]
    # compute results
    Etotal,Eloc,Ek,αασ,nk,Δασ,Slocασ,nn12expt,x,w=compute_degenerate_case(nloc,para[1],[para[2],0.0,para[3],0.0,para[4],0.0,para[5]],eασ,Ueff,[para[end],para[end]],N_spin_orbital)
    savdData(Etotal,Eloc,Ek,αασ,nk,Δασ,Slocασ,nn12expt,x,w,nασ,G12ασ,βασ,eασ,U,data_dir)
end

# we check one band finaly
data_dir="./one_band_degenerate_inf_half"
mkdir(data_dir)
N_spin_orbital=2
nloc=0.5
nloc=0.4
nloc=0.3
U=1.0
# para=[0.4,-0.01,0.0,0.3,0.2]
para=[0.4,-0.1,1.0]
e_fn=gene_spline_band("./es_inf.dat")
es=gene_ϵs(e_fn,nloc)
eασ=[es for _ in 1:N_spin_orbital]
# there are some problem for large U, check the reason
# two band case
for U in 1.0:0.1:10.0
    Ueff=U
    res=optimize(para->compute_degenerate_case(nloc,para[1],para[2],eασ,Ueff,[para[end],para[end]],N_spin_orbital)[1],para)
    para=res.minimizer
    print("U is $(U) with para $(para)\n")
    βασ=[[para[end],para[end]] for _ in 1:N_spin_orbital]
    g012ασ=para[1]
    nασ=[nloc for _ in 1:N_spin_orbital]
    G12ασ=[g012ασ for _ in 1:N_spin_orbital]
    # x_reduce=para[2:4]
    # compute results
    Etotal,Eloc,Ek,αασ,nk,Δασ,Slocασ,nn12expt,x,w=compute_degenerate_case(nloc,para[1],para[2],eασ,Ueff,[para[end],para[end]],N_spin_orbital)
    savdData(Etotal,Eloc,Ek,αασ,nk,Δασ,Slocασ,nn12expt,x,w,nασ,G12ασ,βασ,eασ,U,data_dir)
end


"""
save the data
"""
function savdData(Etotal,Eloc,Ek,αασ,nk,Δασ,Slocασ,nn12expt,x,w,nασ,G12ασ,βασ,eασ,U,data_dir)
    filename_base=replace("$(data_dir)/U_$(U)_n_$(nασ)",","=>"_"," "=>"","["=>"","]"=>"")
    saveData([Etotal,Eloc,Ek],"$(filename_base)_Etotal_Eloc_Ek.dat")
    saveAsSpinOrbital(nk,filename_base,"nk")
    saveAsSpinOrbital(eασ,filename_base,"ek")
    saveAsSpinOrbital(αασ,filename_base,"alpha")
    saveAsSpinOrbital(βασ,filename_base,"beta")
    saveAsSpinOrbital(nασ,filename_base,"n")
    saveAsSpinOrbital(G12ασ,filename_base,"G12")
    saveAsSpinOrbital(G12ασ,filename_base,"G12")
    saveAsSpinOrbital(Slocασ,filename_base,"S")
    saveData(x,"$(filename_base)_x.dat")
    saveData(w,"$(filename_base)_w.dat")
    saveData(nn12expt,"$(filename_base)_nn.dat")
end

"""
take nk as example
quantity_name="nk"
nk[2]
"""
function saveAsSpinOrbital(nk,filename_base,qauntity_name)
    N_spin_orbital=length(nk)
    for i in 1:N_spin_orbital
        saveData(nk[i],"$(filename_base)_$(qauntity_name)_spin_orb_$(i).dat")
    end    
end

