# there are some difference in cal_total_energy and direct version, we try to nail down the problem here.
# we first copy the two relevant function here and return the intermediate variables.
"""
we store all the intermediate results
"""
function cal_total_energy_check(G12ασ,x,βασ,momentum_info,interaction)
    (nασ,Δασ,nασ_below,nασ_above,αασ_below,αασ_above,βασ_below,βασ_above,nkασ_below,nkασ_above,Aασ_below,Aασ_above,Kασ_below,Kασ_above,∂Kασ∂nX_below,∂Kασ∂nX_above,∂Kασ∂βX_below,∂Kασ∂βX_above,∂Aασ∂nX_below,∂Aασ∂nX_above,∂Aασ∂βX_below,∂Aασ∂βX_above)=momentum_info
    # this is the fixed value, as denoted as star in notes, here, we just keep the name, but add _track (if we will use the same variable) to the value (zeroth order are same) computed in the procedure, so the automatically differentialion then can properly backpropagate teh derivatives from the momentum info
    N_spin_orbital=length(nασ)
    w=cal_w_scaled(x,nασ,G12ασ)
    pmatwασ,g12matwSασ=cal_p_g12_mat(G12ασ)
    g12ασ=[expt(w,cal_Xmatfull(pmatwασ,g12matwSασ,i)) for i in 1:N_spin_orbital]
    Slocασ=cal_Slocασ(nασ,G12ασ,g12ασ)
    K0=sum(Kασ_below)+sum(Kασ_above)
    Δασ_track=cal_Δασ(g12ασ,Slocασ) # Δασ_track-Δασ
    nασ_below_track=nασ-Δασ_track # nασ_below_track-nασ_below
    nασ_above_track=Δασ_track   #  nασ_above_track-nασ_above
    δnασ_below=nασ_below_track-nασ_below
    δnασ_above=nασ_above_track-nασ_above
    βασ_below_track=[βασ_[1] for βασ_ in βασ]
    βασ_above_track=[βασ_[2] for βασ_ in βασ]
    δβασ_below=βασ_below_track-βασ_below
    δβασ_above=βασ_above_track-βασ_above
    # we then compute the change of δβ
    # we use dot to sum, so we drop ασ
    δK_below=dot(∂Kασ∂nX_below,δnασ_below)+dot(∂Kασ∂βX_below,δβασ_below)
    δK_above=dot(∂Kασ∂nX_above,δnασ_above)+dot(∂Kασ∂βX_above,δβασ_above)
    δK=δK_below+δK_above
    K_track=K0+δK
    # now, we need to track A  part, Aασ_below
    Aασ_below_track=Aασ_below + ∂Aασ∂nX_below.*δnασ_below + ∂Aασ∂βX_below.*δβασ_below
    Aασ_above_track=Aασ_above + ∂Aασ∂nX_above.*δnασ_above + ∂Aασ∂βX_above.*δβασ_above
    g33matwασ=[cal_g33_mat_(cal_Gfull(nασ[i],G12ασ[i],g12ασ[i],Aασ_below_track[i],Aασ_above_track[i],Slocασ[i])) for i in 1:N_spin_orbital]
    Eloc=sum([coefficient*expt(w,cal_Xmatfull(pmatwασ,g33matwασ,idx1,idx2)) for (idx1,idx2,coefficient) in interaction ])
    E=Eloc+K_track
    w,g12ασ,Slocασ,Δασ_track,δK,Aασ_below_track,Aασ_above_track,Eloc,E
end

"""
it seems the problem is we 
"""
function cal_total_energy_direct(G12ασ,x,βασ,eασ,interaction)
    # we first take the cal_momemtum_part(nασ,G12ασ,x,βασ,eασ), but with the scaled cal_w_scaled
    N_spin_orbital=length(nασ)
    w=cal_w_scaled(x,nασ,G12ασ)
    pmatwασ,g12matwSασ=cal_p_g12_mat(G12ασ)
    g12ασ=[expt(w,cal_Xmatfull(pmatwασ,g12matwSασ,i)) for i in 1:N_spin_orbital]
    # we hav enot update S
    #!! fixed this
    Slocασ=cal_Slocασ(nασ,G12ασ,g12ασ)

    Δασ=cal_Δασ(g12ασ,Slocασ)
    Aασ=[]
    αασ=[]
    nk=[]
    # we don't need derivatives
    K0=0                        # kinecit energy
    for i in 1:N_spin_orbital
        Aασ_,αασ_,nk_=cal_Aασ_αασ_nk_(nασ[i],Δασ[i],βασ[i],eασ[i])
        K0ασ_=mean(nk_[1].*eασ[i][1])*nασ[i]+mean(nk_[2].*eασ[i][2])*(1-nασ[i])
        K0+=K0ασ_
        push!(Aασ,Aασ_)
        push!(αασ,αασ_)
        push!(nk,nk_)
    end
    # now, we move to cal_energy_for_diff_old
    g33matwασ=[cal_g33_mat_(cal_Gfull(nασ[i],G12ασ[i],g12ασ[i],Aασ[i],Slocασ[i])) for i in 1:N_spin_orbital]
    Eloc=sum([coefficient*expt(w,cal_Xmatfull(pmatwασ,g33matwασ,idx1,idx2)) for (idx1,idx2,coefficient) in interaction ])
    Eloc+K0
    w,g12ασ
end



w_v1_0,g12ασ_v1_0,Slocασ_v1_0,Δασ_track_v1_0,δK_v1_0,Aασ_below_track_v1_0,Aασ_above_track_v1_0,Eloc_v1_0,E_v1_0=cal_total_energy_check(G12ασ,x,βασ,momentum_info,interaction)
# now ,consider a random change
δG12=rand(4)*1e-5
G12ασ_new=G12ασ+δG12
w_v1_new,g12ασ_v1_new,Slocασ_v1_new,Δασ_track_v1_new,δK_v1_new,Aασ_below_track_v1_new,Aασ_above_track_v1_new,Eloc_v1_new,E_v1_new=cal_total_energy_check(G12ασ_new,x,βασ,momentum_info,interaction)

# and we do the same thing for direct
