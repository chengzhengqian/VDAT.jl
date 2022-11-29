function cal_energy_half_SU_N(para)
    w=cal_w(para[3],para[3],para[3],0,N_spin_orbital)
    result=cal_energy_with_symmetry(para[1:1],para[2:2],para[2:2],w,e_fn,interaction,symmetry,N_spin_orbital)
    print("call with $(para), get energy $(result[1])\n")
    result[1]
end
