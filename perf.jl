z = rand(800,10000);
a = rand(800,10000);
w = randn(800,800);
atmp = zeros(800,500);

function mv(z,a,w)
    av = view(a,:,1:500)
    z[:,1:500] = w * av
end

function mcopy(z,a,w)
    av = view(a,:,1:500)
    copyto!(atmp, av)
    z[:,1:500] = w * atmp
end

function v1(z,a,w)
    av = view(a,:,1:500)
    z[:,1:500] = w * av   
    av[:,1:500] = tanh.(z[:,1:500])
    av .== a[:,1:500]
end

function v2(z,a,w)
    av = view(a,:,1:500)
    zv = view(z,:,1:500)
    zv[:] = w * av 
    av[:] = tanh.(zv)
    av .== a[:,1:500]
end

function v3(z,a,w)
    av = view(a,:,1:500)
    zv = view(z,:,1:500)
    zv[:] = w * av 
    av[:] = tanh.(zv)
    av .== a[:,1:500]
end

