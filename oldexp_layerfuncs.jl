# closures and layer_funcs collections

function docalcs(x,y,z)
    println("x: ",x)
    println("y: ", y)
    println("z: ", z)
    x[:] = x .* 2.0
    y[:] = y .* 2.0
    z[:] = z .* 2.0    
end

function docalcs(x,y)
    println("x: ",x)
    println("y: ", y)
    x[:] = x .* 2.0
    y[:] = y .* 2.0
end

function preploops()
    a = [1,2,3]
    b = [21,22,23]
    c = [31,32,33]
    funcs1 = [(docalcs,(a,b)), (docalcs,(b,c))]
    funcs2 = [(docalcs,(a,b,c)), (docalcs,(c,b,a))]
    looper(funcs1)
    looper(funcs2)
end

function looper(fs)
    for i = 1:length(fs)
        fs[i][1](fs[i][2]...)
    end
end

