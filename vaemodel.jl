module VAE
# import Automatic Differentiation
using Flux
using MLDatasets
using Statistics
using Logging
using Test
using Random
using Images
using Plots:scatter
using StatsFuns: log1pexp
Random.seed!(412414);

#### Probability Stuff
# log-pdf of x under Factorized or Diagonal Gaussian N(x|μ,σI)
function factorized_gaussian_log_density(mu, logsig,xs)
  """
  mu and logsig either same size as x in batch or same as whole batch
  returns a 1 x batchsize array of likelihoods
  """
  σ = exp.(logsig)
  return sum((-1/2)*log.(2π*σ.^2) .+ -1/2 * ((xs .- mu).^2)./(σ.^2),dims=1)
end

# log-pdf of x under Bernoulli
function bernoulli_log_density(logit_means,x)
  """Numerically stable log_likelihood under bernoulli by accepting μ/(1-μ)"""
  b = (x .* 2) .- 1 # {0,1} -> {-1,1}
  return - log1pexp.(-b .* logit_means)
end

## This is really bernoulli
@testset "test stable bernoulli" begin
  using Distributions
  x = rand(10,100) .> 0.5
  μ = rand(10)
  logit_μ = log.(μ./(1 .- μ))
  @test logpdf.(Bernoulli.(μ),x) ≈ bernoulli_log_density(logit_μ,x)
  # over i.i.d. batch
  @test sum(logpdf.(Bernoulli.(μ),x),dims=1) ≈ sum(bernoulli_log_density(logit_μ,x),dims=1)
end

# sample from Diagonal Gaussian x~N(μ,σI) (hint: use reparameterization trick here)
sample_diag_gaussian(μ,logσ) = (ϵ = randn(size(μ)); μ .+ exp.(logσ).*ϵ)
# sample from Bernoulli (this can just be supplied by library)
sample_bernoulli(θ) = rand.(Bernoulli.(θ))

# Load MNIST data, binarise it, split into train and test sets (10000 each) and partition train into mini-batches of M=100.
# You may use the utilities from A2, or dataloaders provided by a framework
function load_binarized_mnist(train_size=10000, test_size=10000)
  train_x, train_label = MNIST.traindata(1:train_size);
  test_x, test_label = MNIST.testdata(1:test_size);
  @info "Loaded MNIST digits with dimensionality $(size(train_x))"
  train_x = reshape(train_x, 28*28,:)
  test_x = reshape(test_x, 28*28,:)
  @info "Reshaped MNIST digits to vectors, dimensionality $(size(train_x))"
  train_x = train_x .> 0.5; #binarize
  test_x = test_x .> 0.5; #binarize
  @info "Binarized the pixels"
  return (train_x, train_label), (test_x, test_label)
end

function batch_data((x,label)::Tuple, batch_size=100)
  """
  Shuffle both data and image and put into batches
  """
  N = size(x)[end] # number of examples in set
  rand_idx = shuffle(1:N) # randomly shuffle batch elements
  batch_idx = Iterators.partition(rand_idx,batch_size) # split into batches
  batch_x = [x[:,i] for i in batch_idx]
  batch_label = [label[i] for i in batch_idx]
  return zip(batch_x, batch_label)
end
# if you only want to batch xs
batch_x(x::AbstractArray, batch_size=100) = first.(batch_data((x,zeros(size(x)[end])),batch_size))


### Implementing the model

## Load the Data

## Model Dimensionality
# #### Set up model according to Appendix C (using Bernoulli decoder for Binarized MNIST)
# Set latent dimensionality=2 and number of hidden units=500.
Dz, Dh = 2, 500
Ddata = 24*5

# ## Generative Model
# This will require implementing a simple MLP neural network
# See example_flux_model.jl for inspiration
# Further, you should read the Basics section of the Flux.jl documentation
# https://fluxml.ai/Flux.jl/stable/models/basics/
# that goes over the simple functions you will use.
# You will see that there's nothing magical going on inside these neural network libraries
# and when you implemented a neural network in previous assignments you did most of the work.
# If you want more information about how to use the functions from Flux, you can always reference
# the internal docs for each function by typing `?` into the REPL:
# ? Chain
# ? Dense
decoder = Chain(Dense(Dz,Dh,tanh), Dense(Dh,Ddata))

## Model Distributions
log_prior(z) = factorized_gaussian_log_density(0,0,z)

function log_likelihood(x,z)
  """ Compute log likelihood log_p(x|z)"""
  θ = decoder(z)
  return sum(bernoulli_log_density(θ,x), dims = 1)
end


joint_log_density(x,z) = log_likelihood(x,z) + log_prior(z)

## Amortized Inference
function unpack_gaussian_params(θ)
  μ, logσ = θ[1:2,:], θ[3:end,:]
  return  μ, logσ
end

encoder = Chain(Dense(Ddata,Dh,tanh), Dense(Dh,Dz*2),unpack_gaussian_params)

log_q(q_μ, q_logσ, z) = factorized_gaussian_log_density(q_μ, q_logσ, z)

function elbo(x)
  q_μ, q_logσ = encoder(x)
  z = sample_diag_gaussian(q_μ,q_logσ)
  joint_ll = joint_log_density(x,z)
  log_q_z = log_q(q_μ, q_logσ, z)
  elbo_estimate =  mean(joint_ll - log_q_z)
  return elbo_estimate
end

function loss(x)
  return -elbo(x)
end
##loss(train_x)

# Training with gradient optimization:
# See example_flux_model.jl for inspiration

function train_model_params!(train_x, test_x; loss = loss, encoder = encoder, decoder = decoder, nepochs=10)
  # model params
  ps = Flux.params(encoder, decoder)
  # ADAM optimizer with default parameters
  opt = ADAM()
  # over batches of the data
  for i in 1:nepochs
    for d in batch_x(train_x)
      gs = Flux.gradient(ps) do
        ls = loss(d)
        return ls
      end
      Flux.Optimise.update!(opt,ps,gs)
    end
    if i%1 == 0 # change 1 to higher number to compute and print less frequently
      @info "Test loss at epoch $i: $(loss(batch_x(test_x)[1]))"
      means = encoder(train_x)[1]
      display(scatter(means[1,:],means[2,:], markersize = 2, alpha = 0.1))
    end
  end
  @info "Parameters of encoder and decoder trained!"
end

#=
### Save the trained model!
using BSON:@save
cd(@__DIR__)
@info "Changed directory to $(@__DIR__)"
save_dir = "trained_models"
if !(isdir(save_dir))
  mkdir(save_dir)
  @info "Created save directory $save_dir"
end
@save joinpath(save_dir,"encoder_params.bson") encoder
@save joinpath(save_dir,"decoder_params.bson") decoder
@info "Saved model params in $save_dir"


## Load the trained model!
using BSON:@load
cd(@__DIR__)
@info "Changed directory to $(@__DIR__)"
load_dir = "trained_models"
@load joinpath(load_dir,"encoder_params.bson") encoder
@load joinpath(load_dir,"decoder_params.bson") decoder
@info "Load model params from $load_dir"

# Visualization
# make vector of digits into images, works on batches also
mnist_img(x) = ndims(x)==2 ? Gray.(permutedims(reshape(x,28,28,:), [2, 1, 3])) : Gray.(transpose(reshape(x,28,28)))
## Example for how to use mnist_img to plot digit from training data
plot(mnist_img(train_x[:,10]))

z = sample_diag_gaussian(zeros(2,10), zeros(2,10))
transform(m) = exp(m)/(1+exp(m))

plotsl = Any[]
for i in 1:10
  push!(plotsl, plot(mnist_img(transform.(decoder(z)[:,i]))))
end
for i in 1:10
  push!(plotsl, plot(mnist_img(sample_bernoulli(transform.(decoder(z)[:,i])))))
end
plotsl
display(plot(plotsl..., layout = grid(2,10), size = (1000,400)))
savefig("plots/3a.pdf")



means = encoder(train_x)[1]
scatter(means[1,:],means[2,:], markersize = 0.2)
savefig("plots/3b-1.pdf")
scatter(means[1,:],means[2,:], markersize = 2,markerstrokewidth = 0.01, group = train_label)
savefig("plots/3b-2.pdf")


function interpolate(z1, z2; n = 10)
  inter = Any[]
  for i in 1:n
    push!(inter, ((i-1)/n)z1 +(1-(i-1)/n)z2)
  end
  return inter
end


z1 = [0,5]
z2 = [4,0]
z3 = [-2,0]
z4 = [0,0]
z5 = [2,-3]
z6 = [0,-3]

plotinter = Any[]

for i in 1:10
  push!(plotinter, plot(mnist_img(transform.(decoder(interpolate(z1,z2)[i])))))
end
for i in 1:10
  push!(plotinter, plot(mnist_img(transform.(decoder(interpolate(z3,z4)[i])))))
end
for i in 1:10
  push!(plotinter, plot(mnist_img(transform.(decoder(interpolate(z5,z6)[i])))))
end
display(plot(plotinter..., layout = grid(3,10), size = (1000,400)))
savefig("plots/3c.pdf")


imgid = 27
plot(mnist_img(train_x[:,imgid]))
digit = train_x[:,imgid]
K = 1000
params = (randn(Dz), randn(Dz))

function half(x)
  m = reshape(reshape(x, 28,28, length(x[1,:]))[1:14,:,:],14*28,length(x[1,:]))
  return m
end

function log_likelihood_top(x,z)
  """ Compute log likelihood log_p(top half of x|z)"""
  θ = half(decoder(z))
  return sum(bernoulli_log_density(θ,x), dims = 1)
end

joint_log_density_top(x,z) = log_likelihood_top(x,z) + log_prior(z)
##takes half matrix x as evidence

function elbo_svi(params,logp,num_samples)
  sample_z = params[1].+ exp.(params[2]) .* randn(length(params[1]), num_samples)
  logp_estimate = logp(sample_z)
  logq_estimate = factorized_gaussian_log_density(params[1], params[2],sample_z)
  return mean(logp_estimate.-logq_estimate)
end

function neg_elbo_conv(params; x = half(digit), num_samples = K)
  logp(zs) = joint_log_density_top(x, zs)
  return -elbo_svi(params,logp, num_samples)
end


function skillcontour!(f; colour=nothing)
  n = 100
  x = range(-3,stop=3,length=n)
  y = range(-3,stop=3,length=n)
  z_grid = Iterators.product(x,y) # meshgrid for contour
  z_grid = reshape.(collect.(z_grid),:,1) # add single batch dim
  z = f.(z_grid)
  z = getindex.(z,1)'
  max_z = maximum(z)
  levels = [.99, 0.9, 0.8, 0.7,0.6,0.5, 0.4, 0.3, 0.2] .* max_z
  if colour==nothing
  p1 = contour!(x, y, z, fill=false, levels=levels)
  else
  p1 = contour!(x, y, z, fill=false, c=colour,levels=levels,colorbar=false)
  end
  plot!(p1)
end

function fit_variational_dist(init_params; evidence=half(digit), num_itrs=200, lr= 0.1, num_q_samples = 1000)
  params_cur = init_params
  loss(θ) = neg_elbo_conv(θ, x = evidence, num_samples = num_q_samples)
  plots_training = Any[]
  for i in 1:num_itrs
    grad_params = gradient(loss, params_cur)[1]
    params_cur =  params_cur .- grad_params.*lr
    @info "loss: $(loss(params_cur)) n: $(i)"
    p1 = plot(title="Contour Plots of p (Red) and q_phi(Blue)",
        xlabel = "z1",
        ylabel = "z2");
    p(zs) = exp(joint_log_density_top(evidence, zs))
    skillcontour!(p,colour=:red)
    qphi(zs) = exp(factorized_gaussian_log_density(params_cur[1], params_cur[2],zs))
    skillcontour!(qphi, colour=:blue)
    display(p1)
    if i%10 == 0
      push!(plots_training, p1)
    end
  end
  return (params_cur, loss(params_cur), plots_training)
end

paramsinit = [[-1,1],[0,0]]
params1 = fit_variational_dist(paramsinit, lr = 0.005, num_itrs = 50, num_q_samples= 100)

p1 = plot(title="Contour Plots of p (Red) and q_phi(Blue)",
    xlabel = "z1",
    ylabel = "z2");
p(zs) = exp(joint_log_density_top(half(digit), zs))
skillcontour!(p,colour=:red)
qphi(zs) = exp(factorized_gaussian_log_density(params1[1][1], params1[1][2],zs))
skillcontour!(qphi, colour=:blue)
savefig("plots/4a-1.pdf")
params1[3]
plot(params1[3]...,layout = grid(5,10), showaxis = false, size = (1000,1000), title = "", xlabel = "", ylabel = "")
savefig("plots/4a-2.pdf")

samplezfinal = sample_diag_gaussian(params1[1][1], params1[1][2])
frankenpart = transform.(decoder(samplezfinal))
frankenstein = reshape(hcat(reshape(train_x[:,imgid], 28,28)[:,1:14],reshape(frankenpart, 28,28)[:,15:28]),28*28)
comp = [plot(mnist_img(frankenstein)),plot(mnist_img(train_x[:,imgid]))]
plot(comp...,layout = grid(1,2))
savefig("plots/4b")
=#

end
