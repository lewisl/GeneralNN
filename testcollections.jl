struct foo
    x::Int
end

mutable struct bar
    x::Int
end

function makedict()
    return Dict("x"=> 5)
end

function makestruct()
    return foo(5)
end

function makemutable()
    return bar(5)
end

function retrievedict(n::Int, dd)
    z = Array{Int,1}(undef,n)
    for i=1:n
        z[i] = dd["x"]
    end
end

function retrievestruct(n::Int, ss)
    z = Array{Int,1}(undef,n)
    for i=1:n
        z[i] = ss.x
    end
end

function retrievemutable(n::Int, mm)
    z = Array{Int,1}(undef,n)
    for i=1:n
        z[i] = mm.x
    end
end