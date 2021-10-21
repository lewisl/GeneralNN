function colswap(m, sv)
    mm = copy(m)
    for (i,j) in enumerate(sv)
        mm[:,i] = m[:,j]
    end
    return mm
end