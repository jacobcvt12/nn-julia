using Distributions

# create composite for model
type Network
    # layers
    num_layers::Int

    # sizes
    sizes::Vector{Int}

    # biases and weights
    biases::Vector{Array{Float64, 2}}
    weights::Vector{Array{Float64, 2}}

    Network(num_layers, sizes, biases, weights) = new(num_layers, sizes, biases, weights)
end

# create outer constructor
function Network(sizes)
    num_layers = length(sizes)
    biases = [rand(Normal(), y, 1) for y in sizes[2:end]]
    weights = [rand(Normal(), y, x) for (x, y) in zip(sizes[1:(end-1)], sizes[2:end])]

    Network(num_layers, sizes, biases, weights)
end

function sigmoid(z)
    1. / (1. + exp(-z))
end

function sigmoid_prime(z)
    sigmoid(z) * (1 - sigmoid(z))
end

function feedfoward(net::Network, a)
    for (b, w) in zip(net.biases, net.weights)
        a = sigmoid(w * a + b)
    end

    return a
end
