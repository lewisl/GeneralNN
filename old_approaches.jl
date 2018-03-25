function dopad_cat(img, pad)  # use array catenation
    if ndims(img) == 3
        imgx, imgy, imgd = size(img)
    elseif ndims(img) == 2
        imgx, imgy = size(img)
        imgd = 1
    else
        error("Wrong number of dimensions for image.")
    end

    t = eltype(img)

    if imgd > 1
        hold = [vcat(zeros(t, pad,imgy+2*pad),
                     hcat(zeros(t, imgx,pad),
                     img[:,:,i],
                     zeros(t, imgx,pad)),
                     zeros(t, pad,imgy+2*pad)) 
                for i = 1:imgd]
        return cat(3,hold...)
    else
        return vcat(zeros(t, pad,imgy+2*pad),
                    hcat(zeros(t, imgx,pad),
                    img,
                    zeros(t, imgx,pad)),
                    zeros(t, pad,imgy+2*pad))
    end
end


# fc basically reimplements reshape:  same # of allocations, same speed, uses more memory though because it makes a copy
"""
function fc(img)

    flatten a 2d or 3d image for a conv net to use as a fully connected layer.
    follow the convention that features are rows and examples are columns.
    each column of an image plane is stacked below the prior column.  the next
    plan (or channel) comes below the prior channel.
    The size of img must have 4 dimensions even if each image has only 1 channel.
    For example:  7 6x6 images each with 1 channel should have dims 6,6,1,7.
"""
function fc2(imgstack)
    if ndims(imgstack) != 4
        error("imgstack must have 4 dimensions even if img is 2d (use 1 for number of channels)")
    end
    t = eltype(imgstack)
    m,n,c,z = size(imgstack)  # m x n image with c channels => z images like this
    ret = Array{t}(prod(size(imgstack)[1:end-1]), z)  # total values (per image) rows x k (images) columns
    for l = 1:z
        for k = 1:c
            for j = 1:n
                for i = 1:m
                    ret[i + ((j-1)*m) + (k-1)*(m*n), l] =  imgstack[i, j, k, l]
                end
            end
        end
    end

    return ret
end


"""
function stack2(fc, imgdims)

    Convert a flattened image stack back to an image stack.
    imgdims must provide 3 integer values:  m (rows or height) x n (columns or width) x c (number of channels).
    c, number of channels, must be provided for 2D images:  just use 1.

    returns: an imagestack that is m x n x c x z where z is the number of images (or examples)
    Note: this makes a copy of the original data
"""
function stack2(fc, imgdims::Array{Int,1})
    if length(imgdims) != 3
        error("imgdims must contain 3 integer values")
    end

    if ndims(fc) != 2
        error("fc must a 2d array with rows for image data and a column for each image")
    end

    m,n,c = imgdims
    fcx, z = size(fc)

    if fcx != prod(imgdims)
        error("number of rows--number of elements for each image--does not match the image dimensions")
    end

    ret = Array{Float64,4}(m,n,c,z)
    for l = 1:z
        for k = 1:c
            for j = 1:n
                for i = 1:m
                    ret[i,j,k,l] = fc[i + ((j-1)*m) + (k-1)*(m*n) ,l]
                end
            end
        end
    end

    return ret
end