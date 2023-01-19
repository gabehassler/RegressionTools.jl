
function make_raw_design_matrix(df::DataFrame;
                                vars::Vector{String} = names(df),
                                categorical_vars::Vector{String} = String[],
                                references::Dict{String, String} = Dict{String, String}(),
                                interactions::Vector{<:Vector{<:AbstractString}} = Vector{String}[])
    n = length(vars)
    sub_vars = [String[] for _ = 1:n]
    sub_mats = Vector{Matrix{Float64}}(undef, n)

    # @assert "has.ehp" in keys(references)
    # @assert "has.hmp" in keys(references)

    dict = Dict{String, Vector{String}}()

    for i = 1:n
        force_categorical = vars[i] in categorical_vars
        x = df[!, vars[i]]
        x = force_categorical ? string.(x) : x

        ref = string(x[1])
        if vars[i] in keys(references)
            ref = references[vars[i]]
            if !(ref in x)
                error("reference $ref not in $(vars[i]). Options are $(join(unique(x), ", "))")
            end
        end

        var, mat = expand(x, vars[i], ref)
        sub_vars[i] = var
        sub_mats[i] = mat
        dict[vars[i]] = var
    end

    v = vcat(sub_vars...)
    X = hcat(sub_mats...)

    for i = 1:length(interactions)
        i_vars = interactions[i]
        if length(i_vars) != 2
            error("not implemented")
        end
        v1 = i_vars[1]
        v2 = i_vars[2]
        s1 = dict[v1]
        s2 = dict[v2]
        n1 = length(s1)
        n2 = length(s2)
        m = n1 * n2
        X_int = fill(NaN, size(X, 1), m)
        v_int = fill("", m)
        ind = 1
        nm = "$v1 by $v2 interaction"
        for j = 1:n1
            p1 = parse_binary(s1[j])
            for k = 1:n2
                p2 = parse_binary(s2[k])
                nmjk = categorical_to_binary(nm,
                                             "$(p1.value) -X- $(p2.value)",
                                             "$(p1.reference) -X- $(p2.reference)")
                ind1 = findfirst(isequal(s1[j]), v)
                ind2 = findfirst(isequal(s2[k]), v)
                # @show ind1
                # @show ind2
                # @show s1
                # @show s2
                X_int[:, ind] = X[:, ind1] .* X[:, ind2]
                v_int[ind] = nmjk
                ind += 1
            end
        end
        v = [v; v_int]
        X = [X X_int]
        dict[nm] = v_int
    end

    return (v = v, X = X, expansion_dict = dict)
end

function merge_columns(v::Vector{String},
                       X::Matrix{Float64},
                       columns::Vector{String},
                       new_column::String;
                       keep_vars::Vector{String} = String[])
    inds = findall(x -> x in columns, v)
    @assert length(inds) == length(columns)
    x = sum(X[:, inds], dims = 2)
    for i = 1:length(x)
        if x[i] > 1
            x[i] = 1
        end
    end
    drop_vars = setdiff(columns, keep_vars)
    drop_inds = findall(x -> x in drop_vars, v)
    to_keep = setdiff(1:length(v), drop_inds)
    v = [v[to_keep]; new_column]
    X = [X[:, to_keep] x]
    return (v = v, X = X)
end

function process_design_matrix(v::Vector{String},
                               X::Matrix{Float64};
                               eigval_tol::Float64 = 1e-10,
                               eigvec_tol::Float64 = 1e-10,
                               add_intercept::Bool = true,
                               standardize_continuous::Bool = false,
                               categorical_drop::Dict{String, String} = Dict{String, String}(),
                               minimum_observations::Int = 0)
    x = 1
    p = size(X, 2)

    to_remove_novar = Int[]
    to_remove_lowcount = Int[]

    # remove zero variance variables & (optionally) standardize continuous
    for j = 1:p
        x = @view X[:, j]
        u = unique(x)
        if length(u) == 1
            push!(to_remove_novar, j)
        elseif standardize_continuous &&
                length(u) > 2 && #assumed it's a continuous variable
                !startswith(v[j], "conditions.")
            t = fit(ZScoreTransform, x)
            StatsBase.transform!(t, x)
        elseif count(a -> a > 0.0, x) < minimum_observations
            push!(to_remove_lowcount, j)
        end
    end

    println("removing the following variables from the model (zero variance): "
            * join(v[to_remove_novar], ", ") * "\n")
    println("removing the following variables from the model (low non-zero count): "
            * join(v[to_remove_lowcount], ", ") * "\n")

    to_remove = [to_remove_novar; to_remove_lowcount]
    to_keep = setdiff(1:p, to_remove)
    v = v[to_keep]
    X = X[:, to_keep]

    # removing other variables
    # to_remove = String[]
    # p = size(X, 2)
    # for i = 1:p
    #     val = parse_binary(v[i])
    #     if !isnothing(val) &&
    #             val.var in keys(categorical_drop) &&
    #             categorical_drop[val.var] == val.value
    #         push!(to_remove, i)
    #         @show i
    #     end
    # end
    # to_keep = setdiff(1:p, to_remove)
    # v = v[to_keep]
    # X = X[:, to_keep]


    v = ["intercept"; v]
    X = [ones(size(X, 1)) X]


    eig = eigen(X' * X)

    zero_inds = findall(x -> abs(x) < eigval_tol, eig.values)
    nz = length(zero_inds)

    if nz > 4
        println("Many colinear sets of variables detected. Not printing.\n")
    else
        for ind in zero_inds
            v_inds = findall(x -> abs(x) > eigvec_tol, eig.vectors[:, ind])
            println("The following variables are jointly co-linear: " * join(v[v_inds], ", "))
            println()
        end
    end

    to_keep = 1:length(v)
    to_remove = Int[]
    counter = 0
    @show nz
    while eig.values[1] < eigval_tol

        inds = findall(x -> abs(x) > eigvec_tol, @view eig.vectors[:, 1])
        inds = setdiff(inds, 1)
        Xi = @view X[:, to_keep[inds]]

        vs = vec(var(Xi, dims = 1))
        push!(to_remove, to_keep[inds[findmin(vs)[2]]])
        to_keep = setdiff(1:length(v), to_remove)

        X_keep = X[:, to_keep]
        eig = eigen(X_keep' * X_keep)
        counter += 1
        @show counter
        if counter > nz + 1
            error("this shouldn't happen")
        end
        # to_keep = setdiff(1:length(v), to_remove[1:i])
        # Z = X[:, to_keep]
        # d = eigen(Z' * Z)
        # @show d.values
        # a = findall(x -> abs(x) < eigval_tol, d.values)
        # b = length(a)
        # for j = 1:b
        #     @show j
        #     c = findall(x -> abs(x) > eigvec_tol, @view d.vectors[:, a[j]])
        #     @show to_keep[c]
        # end
    end

    println("removing the following variables from the model (fully co-linear): "
            * join(v[to_remove], ", ") * "\n")

    to_keep = setdiff(1:length(v), to_remove)
    v = v[to_keep]
    X = X[:, to_keep]

    if !add_intercept
        X = X[:, 2:end]
        v = v[2:end]
    end


    return (v = v, X = X)
end

function make_design_matrix(df::DataFrame;
                            vars::Vector{String} = names(df),
                            add_intercept::Bool = true,
                            standardize_continuous::Bool = false,
                            categorical_vars::Vector{String} = String[])

    @unpack v, X, expansion_dict = make_raw_design_matrix(df;
                                          vars = vars,
                                          categorical_vars = categorical_vars)
    @unpack v, X = process_design_matrix(v,
                                 X,
                                 add_intercept = add_intercept,
                                 standardize_continuous = standardize_continuous)
    return (v = v, X = X, expansion_dict = expansion_dict)

end

function expand(x::AbstractVector{<:Union{Bool, Int, Float64, Missing}},
                s::String,
                ::String)
    [s], reshape(convert.(Float64, x), length(x), 1)
end

function expand(x::AbstractVector{<:Union{<:AbstractString, Missing}},
                s::String,
                reference::String)

    u = string.(unique(x))
    p = length(u)
    n = length(x)

    if p == 1
        return [categorical_to_binary(s, u[1], "NA")], zeros(n, 1)
    end

    ref_ind = findfirst(isequal(reference), u)
    other_inds = setdiff(1:p, ref_ind)

    vars  = [categorical_to_binary(s, u[i], reference) for i in other_inds]
    X = zeros(n, p - 1)
    for i = 1:(p - 1)
        ind = other_inds[i]
        for j = 1:n
            X[j, i] = x[j] == u[ind] ? 1.0 : 0.0
        end
    end

    return vars, X
end


function categorical_to_binary(var::String, cat::String, ref::String)
    "$var: $cat (versus $ref)"
end

function parse_binary(s::AbstractString)
    re = Regex("(.*): (.*) \\(versus (.*)\\)")
    m = match(re, s)
    if isnothing(m)
        return nothing
    end
    return (var = m[1], value = m[2], reference = m[3])
end

function parse_binary_interaction(s::AbstractString)
    re = Regex(
        "(.*) by (.*) interaction: (.*) -X- (.*) \\(versus (.*) -X- (.*)\\)"
        )
    m = match(re, s)
    return (vars = [m[1], m[2]], values = [m[3], m[4]], references = [m[5], m[6]])
end
