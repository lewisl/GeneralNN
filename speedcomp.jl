function bycomp(x)
    return [j >= .5 ? 1.0 : 0.0 for j in x]
end


function byfor(x)
    choices = zeros(size(x))
    for i = 1:size(choices, 1)
        for j = 1:size(choices,2)
            choices[i,j] =  x[i,j] >= 0.5 ? 1.0 : 0.0
        end
    end
    return choices
end