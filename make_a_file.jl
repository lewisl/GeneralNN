# open a CSV file and save it as a matlab file
# CSV file should have one header line and use unix line endings

using MAT

function mkfile(csvfname, matfname)
    # load the csv file
    data, hdr = readcsv(csvfname, header=true)

    if size(data,2) < 2
        error("Data must be 2 columns or more. Ending.")
    end

    if size(data,2) != size(hdr,2)
        error("Number of columns of data must equal number of headers.")
    end
    
    # check if output file exists and ask permission to overwrite
    if isfile(matfname)
        print("Output file $matfname exists. OK to overwrite? ")
        resp = readline()
        if contains(lowercase(resp), "y")
            rm(matfname)
        else
            error("File exists. Response said no to overwrite.")
        end
    end

    # how many data columns are inputs
    nx = 0
    ny = 0
    for (index, col) in enumerate(hdr)
        col = lowercase(String(col)) # julia makes col type SubString{String}
        if contains(col, "x")
            nx += 1
        end
        if contains(col, "y")
            ny += 1
        end
    end

    # loop through columns and write to matlab file
    outfile = matopen(matfname, "w")
    write(outfile, "x", data[:,1:nx])
    write(outfile, "y", data[:,nx+1:end])
    close(outfile) 

end

