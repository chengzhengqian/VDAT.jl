function save_result(result,interaction,data_dir,file_base)
    filename_tag="$(data_dir)/$(file_base)"
    Etotal,Eloc,K0,nασ,nn,αασ,βασ,eασ,Slocασ,Δασ,Aασ_below,Aασ_above,G12ασ,w,nk=result
    saveData([Etotal,Eloc,K0],"$(filename_tag)_Etotal_Eloc_Ek.dat")
    saveData(nασ,"$(filename_tag)_density.dat")
    saveData(nn,"$(filename_tag)_nn.dat")
    saveData(interaction,"$(filename_tag)_interaction.dat")
    saveData(αασ,"$(filename_tag)_alpha.dat")
    saveData(βασ,"$(filename_tag)_beta.dat")
    saveData(G12ασ,"$(filename_tag)_G12.dat")
    saveData(Slocασ,"$(filename_tag)_Sloc.dat")
    saveData(Δασ,"$(filename_tag)_Delta.dat")
    saveData(Aασ_below,"$(filename_tag)_A_below.dat")
    saveData(Aασ_above,"$(filename_tag)_A_above.dat")
    saveAsSpinOrbital(nk,filename_tag,"nk")
    saveAsSpinOrbital(eασ,filename_tag,"ek")
end

function saveAsSpinOrbital(result,filename_base,qauntity_name)
    N_spin_orbital=length(result)
    for i in 1:N_spin_orbital
        saveData(result[i],"$(filename_base)_$(qauntity_name)_spin_orb_$(i).dat")
    end    
end
