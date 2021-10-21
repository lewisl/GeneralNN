struct MBrng
    cnt::Int
    incr::Int
end

# Base.iterate(mb::MBrng, state=1) = ( state < mb.cnt ?
#                 ((state : (state + mb.incr - 1 < mb.cnt ? state + mb.incr - 1 : mb.cnt)), state + mb.incr)  
#                   : nothing)

for colrng in MBrng(500,100)
       println(colrng)
       end          


# alternative approach to iterate function

Base.iterate(mb::MBrng, state=1) = mbiter(mb::MBrng, state)

function mbiter(mb::MBrng, state)
    up = state + mb.incr
    hi = if up - 1 < mb.cnt
            up - 1
        else
            mb.cnt
        end
    ret = if state < mb.cnt
            (state:hi, up)
        else
            nothing
        end
    return ret
end


# slightly more obvious way using explicit loop: same calcs, just not in the iterator
for (b,l) in zip(Iterators.countfrom(1,mbzi),Iterators.countfrom(mbzi,mbzi))
                    b > train.n && break
                    colrng = b:(l < train.n ? l : train.n)
                    hp.mb_size = colrng[end] - b + 1     


    # stuff

end