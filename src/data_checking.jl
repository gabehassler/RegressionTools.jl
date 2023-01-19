function find_colinearities(X::Matrix{Float64};
                            α::Float64 = 0.05,
                            verbose::Bool = true,
                            vars::Vector{String} = String[])

    if verbose && length(vars) == 0
        error("must supply 'vars' keyword argument if 'verbose = true'")
    end
    C = cor(X, dims = 1)
    p = size(C, 1)
    D = Symmetric(1 .- abs.(C))
    tree = hclust(D, linkage = :complete)
    clusts = cutree(tree, h = α)
    u = unique(clusts)
    clust_ids = [findall(isequal(x), clusts) for x in u]
    clust_sizes = length.(clust_ids)
    relevant_clusts = findall(x -> x > 1, clust_sizes)
    rel_ids = clust_ids[relevant_clusts]
    if verbose
        println("Highly-correlated clusters of variables:")
        for clust in rel_ids
            println("\t" * join(vars[clust], ", "))
            for comb in combinations(1:length(clust), 2)
                i1 = clust[comb[1]]
                i2 = clust[comb[2]]
                println("\t\t correlation between $(vars[i1]) and $(vars[i2]): $(C[i1, i2])")
            end
        end
    end
    rel_ids, C
end