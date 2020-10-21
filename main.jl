using Revise
cd(@__DIR__)
@info "changed directory to $(@__DIR__)"
include("Scansion.jl")
include("vaemodel.jl")
using Images
using Plots
using .scansion: scan_lines, it_scans
using .VAE: train_model_params!, encoder, decoder, sample_diag_gaussian,sample_bernoulli

excerptlength =5

function clean_data(lines)
    for i in 1:length(lines)
        lines[i] = replace(lines[i], r"[ā]"i => s"a")
        lines[i] = replace(lines[i], r"[ē]"i => s"e")
        lines[i] = replace(lines[i], r"[ī]"i => s"i")
        lines[i] = replace(lines[i], r"[ō]"i => s"o")
        lines[i] = replace(lines[i], r"[ū]"i => s"u")
        lines[i] = replace(lines[i], r"[^a-zA-Z\s]" => s"")
        lines[i] = rstrip(lines[i])
    end
    f(x)=(x != "")
    lines = filter(f, lines)
    return lines
end

function make_24(vec)
    spacevec = Any[]
    for i in 1:length(vec)
        if vec[i] == 2
            push!(spacevec,1)
            push!(spacevec,1)
        elseif vec[i] == 1
            push!(spacevec,0)
        end
    end
    return spacevec
end

function make_data(;file="book1.txt", excerptlength = 5)
    lines = readlines(file)
    lines_cl= clean_data(lines)
    lines_sc= scan_lines(lines_cl)[1] ##take only validly scanned ones
    data = zeros(24, length(lines_sc))
    for i in 1:length(lines_sc)
        data[:,i]= make_24(lines_sc[i])
    end
    excerpts = zeros(24*excerptlength, floor(Int, length(lines_sc)/excerptlength)-1)
    for i in 1:floor(Int, length(lines_sc)/excerptlength)-1
        for j in 0:(excerptlength-1)
            excerpts[(j*24+1):((j+1)*24),i] = data[:,i*excerptlength+j]
        end
    end
    return excerpts
end




train_x, test_x = (make_data(file = "halffirst.txt"),make_data(file="halfsecond.txt"))

train_x[1,:] == ones(size(train_x,2))

## Train the model
train_model_params!(train_x,test_x, nepochs=400)

# Visualization
##Latent space
means = encoder(train_x)[1]
scatter(means[1,:],means[2,:], markersize = 2, alpha = 0.1)
##Sample picture of excerpt, works on batches of excerpts also
meter_img(x) = ndims(x)==2 ? Gray.(permutedims(reshape(x,24,excerptlength,:), [2, 1, 3])) : Gray.(transpose(reshape(x,24,excerptlength)))
transform(m) = exp(m)/(1+exp(m))
display(plot(meter_img(1 .- train_x[:,22])))
display(plot(meter_img(1 .- transform.(decoder(means)[:,22]))))

z = zeros(2,10)
z[:,1] = [0,2]'
z[:,2] = [0,2.1]'
z[:,3] = [0, 0]'
z[:,4] = [0,0.1]'
z[:,5] = [0,-0.1]'
z[:,6] = [0,0.3]'
z[:,7] = [1.5,1]'
z[:,8] = [1.5,1.1]'
z[:,9] = [1.5,0.9]'
z[:,10]= [1.5,0.5]'
plotsl = Any[]
z = z = zeros(2,10)
for i in 1:10
    z[1:2,i] = encoder(train_x[:,i])[1]'
end
for i in 1:10
    push!(plotsl, plot(meter_img((1 .- train_x[:,i]))))
end
for i in 1:10
  push!(plotsl, plot(meter_img(1 .- transform.(decoder(z)[:,i]))))
end
for i in 1:10
  push!(plotsl, plot(meter_img(1 .-sample_bernoulli(transform.(decoder(z)[:,i])))))
end
plotsl
display(plot(plotsl..., layout = grid(3,10), size = (1000,400)))
##savefig("plots/3a.pdf")
