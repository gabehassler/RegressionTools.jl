function r_squared(y::Vector{Float64},
                   y_hat::Vector{Float64};
                   adjusted::Bool = false,
                   p::Int = -1)

    if adjusted && p == -1
        error("must set 'n' and 'p' keyword arguments for adjusted r squared")
    end
    n = length(y)
    @assert length(y_hat) == n
    vy = var(y)
    vy_hat = var(y_hat)
    r2 = vy_hat / vy
    if adjusted
        r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    end

    r2
end