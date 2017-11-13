
function outer(m::Int64, mbsize::Int64)
    if mbsize > m; mbsize = m; end
    if mbsize < 1; mbsize = 1; end
    x = rand(1,m)
    used = 0
    start = 1
    fin = start + mbsize -1
    while used < m
        thisbatch = size(x[start:fin],1)
        println(thisbatch)

        used += fin - start + 1
        start = fin + 1
        fin = min(start + mbsize -1, m)
        if m - fin < mbsize
            fin = m
        end
    end

end
