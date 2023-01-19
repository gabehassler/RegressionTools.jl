
function lasso(X::Matrix{Float64}, y::Vector{Float64}; adaptive::Bool = false)
    n, p = size(X)
    @rput X
    @rput y

    weights = ones(p)

    R"""
    library(glmnet)
    """

    if adaptive
        R"""
        fit.ridge <- cv.glmnet(X, y, alpha = 0)
        ridge_coefs <- as.numeric(coef(fit.ridge, s = "lambda.min"))
        """
        @rget ridge_coefs

        weights .= 1 ./ abs.(ridge_coefs[2:end]) # first is intercept
    end

    @rput weights

    R"""
    fit.lasso <- cv.glmnet(X, y, alpha = 1, penalty.factor = weights)
    beta <- as.numeric(coef(fit.lasso, s = "lambda.min"))
    """

    @rget beta
end

function elastic_net(X::Matrix{Float64}, y::Vector{Float64})
    @rput X
    @rput y


    R"""
    fit <- cv.glmnet(X, y)
    beta <- as.numeric(coef(fit, s = "lambda.min"))
    """

    @rget beta
end
