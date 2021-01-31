module Hrgnwake

#=
Copyright (c) 2021 p1scescom

This software is released under the MIT License.
http://opensource.org/licenses/mit-license.php
=#

export main

using Images, CUDA, Flux, ETLCDBReader, FixedPointNumbers, Colors, Statistics
using Flux.Data
using Flux.Losses: crossentropy
using Flux: onecold, onehot, onehotbatch

allhrgn = ['あ', 'い', 'う', 'え', 'お', 'か', 'が', 'き', 'ぎ', 'く', 'ぐ', 'け', 'げ', 'こ', 'ご', 'さ', 'ざ', 'し', 'じ', 'す', 'ず', 'せ', 'ぜ', 'そ', 'ぞ', 'た', 'だ', 'ち', 'ぢ', 'つ', 'づ', 'て', 'で', 'と', 'ど', 'な', 'に', 'ぬ', 'ね', 'の', 'は', 'ば', 'ぱ', 'ひ', 'び', 'ぴ', 'ふ', 'ぶ', 'ぷ', 'へ', 'べ', 'ぺ', 'ほ', 'ぼ', 'ぽ', 'ま', 'み', 'む', 'め', 'も', 'や', 'ゆ', 'よ', 'ら', 'り', 'る', 'れ', 'ろ', 'わ', 'を', 'ん']
hrgnl = length(allhrgn)

imgsize = (63,64)
batchsize = 30

# ETLCDB http://etlcdb.db.aist.go.jp/specification-of-etl-9?lang=ja
function gethrgndata(;dir="$(ENV["HOME"])/Downloads/ETL9B/", testlen = 20)
    etls = ETLCDBReader.getetl9b(dir)
    hrgnetls = filter(x -> x.charcode in allhrgn, etls)
    cuetlsdata = map(x -> Float32.(x.data), hrgnetls)

    train_x = cuetlsdata[1:end-hrgnl*testlen]
    train_y = onehotbatch(map(x -> x.charcode, hrgnetls)[1:end-hrgnl*testlen], allhrgn)

    test_x = cat(permutedims(cuetlsdata[end-hrgnl*testlen+1:end])..., dims=4)
    test_y = onehotbatch(map(x -> x.charcode, hrgnetls)[end-hrgnl*testlen+1:end], allhrgn)

    return train_x,train_y, test_x,test_y
end

function generatemodel()
    model = Chain(Conv((3,3), 1 => 8, relu, pad=1, stride=1),
                      flatten,
                      Dense(imgsize[1]*imgsize[2]*8, hrgnl),
                      softmax)
    return model
end

function train(model, datas, labels, batchsize = 20)
    loss(x,y) = begin
            crossentropy(model(x), y)
        end
    accuracy(x,y) = mean(onecold(model(x)) .== onecold(y))
    opt = ADAM()

    for _ in 1:1
        trainloader = DataLoader((datas, labels), batchsize=batchsize, shuffle=true)
        for (data,label) in trainloader
            cdata = cat(permutedims(data)..., dims=4)
            d = [(cdata, label)]
            Flux.train!(loss, params(model),d , opt)
            cdata = nothing
            d = nothing
        end
    end
end

# init function
function main(model = gpu(generatemodel()))
    train_x,train_y, test_x,test_y = gpu.(gethrgndata())
    train(model, train_x,train_y)
    return model
end

end # module
