function bycomp(x)
    return [j >= 0.0 ? 1.0 : 0.0 for j in x]
end


function byfor(x)
    choices = zeros(size(x))
    for i = 1:size(choices, 1)
        for j = 1:size(choices,2)
            choices[i,j] =  x[i,j] >= 0.0 ? x[i,j] : 0.01 * x[i,j]
        end
    end
    return choices
end


function byformax(x)
    choices = zeros(size(x))
    for j = 1:size(choices,2)
        for i = 1:size(choices, 1)
            choices[i,j] =  max(0.0, x[i,j]) 
        end
    end
    return choices
end


function bigmax(x)
    choices = zeros(size(x))
    choices[:] = max.(0.0, x)
    return choices
end

function splitmax(x)
    choices = zeros(size(x))
    pos = map(x -> x>0.0, x)
    neg = map(x -> x<0.0, x)
    choices[pos] = max.(0.0, x[pos])
    choices[neg] *= 0.01
    return choices
end

function doif(a,b,choice)
    byif(a,b,choice)
end

function byif(a,b,choice)
    if choice
        return a + b
    else
        return a - b
    end
end

function bydisp(a,b, choice)
    simple(a,b,choice)
end

function simple(a,b, choice::Void)
    return a+b
end

function simple(a,b,split::Int)
    return a - b
end
