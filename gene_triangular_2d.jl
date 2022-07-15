"""
in cartesian coordiantes (kx,ky)
k=[0.0,0.0]
"""
function ek(k)
    kx,ky=k
    -2*(cos(kx)+2*cos(kx/2)*cos(sqrt(3)/2*ky))
end
