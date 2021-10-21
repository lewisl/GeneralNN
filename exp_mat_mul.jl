using LinearAlgebra

function exp_mat_mul!(res_arr, arr1, arr2)
    x1, y1 = size(arr1)
    x2, y2 = size(arr2)
    x_res, y_res = size(res_arr)

    @assert y1 == x2 "arr1 cols not equal arr2 rows"
    @assert x_res == x1 "result rows not equal arr1 rows"
    @assert y_res == y2 "result columns not equal arr2 columns"

    for row = 1:x1 # rows of first array and result
        for col = 1:y2 # columns of second array and result
            bb = 0.0
            for incol = 1:y1 # columns of first array and rows of second array
                bb += arr1[row, incol] * arr2[incol, col]
            end
            res_arr[row, col] = bb
        end
    end

end

function exp2!(res_arr, arr1, arr2)
    x1, y1 = size(arr1)
    x2, y2 = size(arr2)
    x_res, y_res = size(res_arr)

    hold = zeros(y1)

    @assert y1 == x2 "arr1 cols not equal arr2 rows"
    @assert x_res == x1 "result rows not equal arr1 rows"
    @assert y_res == y2 "result columns not equal arr2 columns"

    for row = axes(arr1,1)
        for col = axes(arr2,2)
            for i = axes(arr1,2)
                hold[i] = arr1[row,i] * arr2[i,col]
            end
            res_arr[row, col] = sum(hold)
        end
    end
end


function exp3!(res_arr, arr1, arr2)
    res_arr[:] = mul!(res_arr, arr1, arr2)
end

function exp4!(res_arr, arr1, arr2, bias)
    # this is really fast with NO allocations!
    res_arr[:] = mul!(res_arr, arr1, arr2)
    for j = axes(res_arr,2)
        for i = axes(bias, 1)
            res_arr[i,j] += bias[i]
        end
    end
end