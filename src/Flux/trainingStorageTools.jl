using Statistics, LineSearches, Optim

function buildRecord(stack::stFlux, nEpochs)
    p = params(stack).order
    return [zeros(Float32, size(x)..., nEpochs+1) for x in p]
end

function buildRecord(stack, nEpochs)
    zeros(eltype(stack), size(stack)..., nEpochs)
end

function expandRecord!(record::Array{Array}, nEpochs)
    for ii in 1:length(record)
        adding = zeros(Float32, size(record[ii])[1:end-1]..., nEpochs)
        record[ii] = cat(record[ii], adding, dims=ndims(record[ii]))
        size(record[ii])
    end
end

function expandRecord!(record::Array{<:Number}, nEpochs)
    adding = zeros(eltype(record), size(record)[1:end-1]..., nEpochs)
    record[:] = cat(record, adding, dims=ndims(record))
end

function addCurrent!(record, stack::stFlux,ii)
    p = params(stack).order

    for (kk,x) in enumerate(p)
        ax = axes(x)
        record[kk][ax..., ii+1] = x
    end
end

function addCurrent!(record, result,ii)
    p = params(stack).order
    result = 3
    record[]
    for (kk,x) in enumerate(p)
        ax = axes(x)
        record[kk][ax..., ii+1] = x
    end
end

"""
    makeObjFun(target, path::pathLocs, st, normalize=st.normalize)
"""
function makeObjFun(target, path::pathLocs, st, normalize=st.normalize,λ = 1e-10)
    if normalize
        nPaths = sum([prod(size(x)[2:end-1]) for x in target.output])
        return obj(x) = sum(mse(a[1],a[2]) for a in zip(st(x)[path],target[path])) + λ*norm(x)
    else
        return cobj(x) = sum(mse(a[1],a[2]) for a in zip(st(x)[path],target[path]))
    end
end
"""
    makeObjFun(target, path::pathLocs, st, normalize=st.normalize)
"""
function makeObjFun(target, st, normalize=st.normalize,λ=1e-10)
    path = pathLocs(0,:,1,:,2,:)
    makeObjFun(target,path,st,normalize,λ)
end

function fitReverseSt(N, initGuess; opt=Momentum(), ifGpu=identity, obj = nothing, keepWithin=-1,stochastic=false)
    ongoing = copy(initGuess|> ifGpu);
    pathOfDescent = zeros(Float64, size(ongoing,1),1,size(initGuess,3),N+1)|>ifGpu;
    println("initial objective function is $(obj(ongoing))")
    pathOfDescent[:,:,:,1] = ongoing|>ifGpu;
    err = zeros(Float64, N+1); err[1] = obj(pathOfDescent[:,:,:,1])
    return justTrain(N, pathOfDescent, err, 1, ongoing, opt=opt, ifGpu=ifGpu, obj=obj, keepWithin=keepWithin,stochastic=stochastic)
end

function justTrain(N, pathOfDescent, err, prevLen, ongoing; opt=Momentum(), ifGpu=identity, obj = obj, keepWithin=-1,stochastic=false)
    for ii in (prevLen+1):(prevLen+N)
        err[ii] = obj(ongoing)
        if err[ii]>1e5 || isnan(err[ii])
            println("way too big at $(ii), aborting")
            break
        elseif err[ii]/err[1] <1e-7
            println("oh, we actually finished?")
            err= err[1:ii]
            pathOfDescent = pathOfDescent[:,:,:,1:ii-1]
            return (pathOfDescent,err)
        end
        if ii %10==0
            println("$ii rel err: $(err[ii]/err[1]), abs err: $(err[ii]), η=$(opt.eta)")
        end
        ∇ = gradient(obj, ongoing)
        # if the err is flat (less than a .1% change over the past 20 steps), change the step size
        if ii >30 && (mean(err[ii-22:ii-18]) - mean(err[ii-3:ii]))/err[ii] < .001
            # if it varies too much, decrease the step, otherwise, increase
            if count(abs.(err[ii-20:ii].- mean(err[ii-20:ii])) ./err[ii] .>.03) >5
            #if std(err[max(1,ii-20):ii])/err[ii] > 1e-4
                #if sum(abs.(diff(diff(err[max(1, ii-20):ii]))))/err[ii]
                opt.eta = opt.eta*.95
            else
                opt.eta = opt.eta/.9
            end
        end
        update!(opt, ongoing, ∇[1])
        sz = norm(ongoing)
        if keepWithin >0 && sz>keepWithin
            ongoing = ongoing / norm(ongoing) * keepWithin
        end
        if stochastic
            ongoing += opt.eta*1*norm(∇,1)*randn(size(ongoing)...)
        end
        pathOfDescent[:,:,:,ii] = ongoing
    end
    return (pathOfDescent, err,opt)
end

function maximizeSingleCoordinate(N, initGuess, p,target, st; ifGpu=identity,
                                  obj = nothing, keepWithin=-1,
                                  stochastic=false, spacing=:linear,
                                  adjustStepSize=false, allowedTime=-1,
                                  λ=1e-10, tryCatch=true,kwargs...)
    ongoing = copy(initGuess) |> ifGpu;
    if obj!=nothing
        println("initial objective function is $(obj(ongoing))")
    end
    if allowedTime>0
        totalUsed = 0
        N=500
    else
        totalUsed = NaN
    end
    paths = Array{Optim.MultivariateOptimizationResults,1}(undef, 20)
    pathErr = Array{Array{Float64,1}, 1}(undef,20)
    m = findfirst(p.indices .!=nothing) - 1 # which layer are we working in?
    
    if spacing==:exponential
        targetVals = exp.(range(0, log(prod(size(target[m])[2:end-1])), length=20))
    elseif spacing==:linear
        targetVals = range(1, prod(size(target[m])[2:end-1]), length=20)
    else
        error("no spacing type $(spacing)")
    end
   
    lastI=-1
    for (ii,t) in enumerate(targetVals)
        println("-----------------------------------------------------------------------")
        println("target value $t")
        println("-----------------------------------------------------------------------")
        target[p] = t
        divvyAllowedTime = (allowedTime - totalUsed)/(20-ii+1)
        println("allowed time = $(divvyAllowedTime)")
        tmp, λtmp, err, netTime = fitByPerturbing(N, ongoing, p, target, st;
                                    λ = λ, timeAl = divvyAllowedTime,
                                    tryCatch=tryCatch, kwargs...)
        pathErr[ii] = err
        if tmp!=nothing
            paths[ii] =tmp; λ = λtmp
        else
            println("this broke, doubtful it will converge after")
            break
        end
        totalUsed += netTime
        ongoing = paths[ii].minimizer
        lastI = ii
        if totalUsed > allowedTime
            break
        end
    end
    return paths[1:lastI], pathErr[1:lastI]
end

"""
    fitByPerturbing(N, initGuess, p,target, st; obj=nothing, allowedTime = -1,
                         rate = .1, rateFraction=.9, missesInARow=5,
                         NSamples=1000, ifGpu= identity, noiseType=:pink,
                         NAttempts=1e6, varargs...)

implementing the perturbed BFGS idea. Differences: perturbing using pink noise
whose norm is dependent on the size of the signal, whereas theirs is a uniform 
"""

function fitByPerturbing(N, initGuess, p,target, st;  timeAl = -1,
                         rate = .1f0, rateFraction=.8f0, λ=1, missesInARow=1,
                         obj=makeObjFun(target, p,st, true,λ),
                         NSamples=500, ifGpu= identity, noiseType=:pink,
                         NAttempts=1e3, varargs...)

    if timeAl>0
        totalUsed = 0
        N=10000000000000
    else
        totalUsed = NaN
    end
    pert = copy(initGuess) |> ifGpu
    tooMany = 0
    errGrouped = zeros(0)
    # the first algorithm is fixed, so don't pass it through varargs
    fixedAlgo = filter(x-> x[1]!=:algo, varargs)
    perturbedRes, λ = fitUsingOptim(N, pert, p, target, st;
                                    timeAl = timeAl - totalUsed, λ=λ,
                                    ifGpu=ifGpu, algo=BFGS, fixedAlgo...)
    fprev = perturbedRes.minimum # get the initial fit value
    errThisRun = [x.value for x in perturbedRes.trace]
    append!(errGrouped,errThisRun)
    totalUsed+= perturbedRes.time_run
    for j=1:NAttempts
        println("perturbing for the $(j)th time. rate = $(rate)")
        # generate the next example
        (pert, minObj,tooMany), timeUsed, _ = @timed perturb(perturbedRes.minimizer, obj, rate,
                                                             NSamples, noiseType, tooMany)
        println(minObj)
        totalUsed += timeUsed
        oldRes = perturbedRes
        perturbedRes, λ = fitUsingOptim(N, pert, p, target, st;
                                        timeAl = timeAl - totalUsed,
                                        ifGpu=ifGpu, λ=λ, varargs...)
        fcur = perturbedRes.minimum
        # have we stopped progressing during our perturbations?
        if abs(fprev-fcur)/abs(fcur) < eps(typeof(fcur))
            println("there was effectively no change in the objective function, aborting")
            break
        end
        # add condition for if the norm of the function is the source of
        # most of the error, as this is only a fitting concern
        println("----------------------------------------------")
        trueErr = abs(λ*norm(perturbedRes.minimizer) - fcur)
        println("The true error, without regularization: ", trueErr)
        println("----------------------------------------------")
        if trueErr < 1e-1
            println("most of the error is from the norm, breaking")
            break
        end
        fprev = fcur

        # increase the time used
        totalUsed += perturbedRes.time_run

        # record the error
        errThisRun = [x.value for x in perturbedRes.trace]
        append!(errGrouped,errThisRun)
        
        #  have we had tooMany missesInARow?, then decrease the distance
        #  we're stepping away 
        if tooMany >= missesInARow
            rate *= rateFraction
            tooMany=0
        end
        
        # are we perturbing by too small of a step?
        if rate < 1e-4
            println("had to decrease the rate too many times")
            break
        end
        # are we over time?
        if totalUsed > timeAl
            println("Out of time")
            break
        end
        # are we sufficiently converged?
        if perturbedRes.minimum < 1e-7
            break
        end
        if 1.01*oldRes.minimum < perturbedRes.minimum
            println("the new one is more than 10% worse, starting again from the old one")
            perturbedRes = oldRes
        end
    end
    return perturbedRes, λ, errGrouped, totalUsed
end

function perturb(x, obj, stepSize = .01f0, K=100, noiseType=:pink,tooMany=0)
    tmp = genNoise(size(x), K, noiseType) # get your favorite kind of noise
    tmp = stepSize*norm(x) * tmp ./ 
        reshape([norm(w) for w in eachslice(tmp, dims=4)], (1,1,1,K)) # normalize
    # it to match stepSize times the norm of x
    perturbed = x .+ tmp
    errs = [obj(w) for w in eachslice(perturbed, dims=4)]
    μ = argmin(errs)            # find the best
    if errs[μ] > obj(x)
        println("couldn't find a better location. Adjust stepSize or number of samples K")
        tooMany+=1
    else
        tooMany = max(0,tooMany-1)
    end
    return perturbed[:,:,:,μ], errs[μ], tooMany
end

function genNoise(sz, K, noiseType)
    if noiseType==:white
        tmp = randn(Float32, sz..., K)
        return tmp/norm(tmp)
    elseif noiseType==:pink
        N = sz[1]
        pink = 1 ./ Float32.(1 .+ (0:N>>1)) # note: not 1:N>>1+1
        phase = exp.(Float32(2π)*im) .*rand(Float32, N>>1+1, sz[2:end]..., K)
        res = irfft(pink .* phase, N, 1)
        return res ./ norm(res)
    end
end

"""
given a list of optim solutions, choose the one with the largest entry at
location p
"""
function chooseLargest(p,res, st)
    ii = argmax([st(x.minimizer)[p][1] for x in res])
    return res[ii]
end

function continueTrain(N, pathOfDescent, err; opt=Momentum, ifGpu=identity, 
                       obj = obj, keepWithin = -1,stochastic=false) 
    prevLen = size(pathOfDescent,4)
    currentGuess = pathOfDescent[:,:,:,end] |> ifGpu;
    pathOfDescent = cat(pathOfDescent,
                        zeros(Float64, size(pathOfDescent,1),1,size(pathOfDescent,3),N), 
                        dims=4) |> ifGpu;
    err = cat(err, zeros(Float64, N), dims=1)
    return justTrain(N, pathOfDescent, err, prevLen, currentGuess, opt=opt, ifGpu=ifGpu, obj=obj, keepWithin=keepWithin,stochastic=stochastic)
end



function fitUsingOptim(N, ongoing, p, target, st; ifGpu=identity, 
                       obj = nothing, λ=1e-10, timeAl=NaN, tryCatch=true,
                       allowIncrease = false, lineSearch=HagerZhang(),
                       algo=BFGS, kwargs...)
    obj = makeObjFun(target, p, st, true, λ)
    function obj∇!(F, ∇, x)
        y, back = pullback(obj, x)
        if ∇ != nothing
            ∇[:] = back(y)[1]
        end
        if F != nothing
            return y
        end
    end
    if tryCatch
        try
            res = defaultOptim(obj∇!, ongoing, N, timeAl, allowIncrease, algo,
                               lineSearch; kwargs...)
        catch e
            if isa(e, AssertionError) &&  (occursin("isfinite", e.msg) || occursin("B > A", e.msg))
                res,lineSearch,λ = catchAndSwitchLineSearch(e, obj∇!, ongoing,
                       N, target, p, st, λ, timeAl, allowIncrease,algo,
                       lineSearch; kwargs... )
            elseif isa(e, ArgumentError)
                # too little weight placed on the normalization
                res,lineSearch,λ = catchAndUpλ(e, ongoing, N, target, p, st, λ,
                       timeAl, allowIncrease,algo, lineSearch; kwargs...)
            else
                rethrow(e)
            end
        end
    else
        res = defaultOptim(obj∇!, ongoing, N, timeAl, allowIncrease, 
                           algo, lineSearch; kwargs...)
    end
    return res, λ 
end
function defaultOptim(obj∇!,ongoing,N,timeAl, allowIncrease=false, algo=BFGS,
                      lineSearch=HagerZhang(); kwargs...)
    optimize(Optim.only_fg!(obj∇!), ongoing,
             algo(;linesearch=lineSearch, kwargs...),
             Optim.Options(store_trace=true, show_trace=true, iterations=N,
                           show_every=20, time_limit=timeAl, g_tol = 1e-8,
                           f_tol = 1e-4,
                           allow_f_increases=allowIncrease))
end

function catchAndSwitchLineSearch(e, obj∇!, ongoing, N, target, p, st, λ, timeAl, allowIncrease,algo, lineSearch; kwargs...)
    if typeof(lineSearch) <: HagerZhang
        lineSearch = BackTracking()
    elseif typeof(lineSearch) <: BackTracking
        lineSearch = MoreThuente()
    else
        println("well, we tried HagerZhang, BackTracking and MoreThente, and they all broke")
        return nothing, nothing, nothing
        #rethrow(e)
    end
    println("the current line search is $(lineSearch)")
    try
        res = defaultOptim(obj∇!, ongoing, N, timeAl, allowIncrease, algo,
                           lineSearch; kwargs...)
        return res, lineSearch, λ
    catch e2
        if isa(e,AssertionError) && (occursin("isfinite", e.msg) || occursin("B > A", e.msg))
            res, lineSearch, λ = catchAndSwitchLineSearch(e2, obj∇!, ongoing, N,
                                                       target, p, st, λ,
                                                       timeAl, allowIncrease, 
                                                       algo, lineSearch;
                                                       kwargs...) 
            return res, lineSearch, λ
        elseif isa(e,ArgumentError) && occursin("Value and slope", e.msg)
            res,λ = catchAndUpλ(e, ongoing, N, target, p, st, λ, timeAl, allowIncrease,algo, lineSearch; kwargs...)
        else
            rethrow(e2)
        end
    end
end

function catchAndUpλ(e, ongoing, N, target, p, st, λ, timeAl, allowIncrease,algo, lineSearch; kwargs...) 
    println("too little weight placed on the normalization, upping to $(λ*1e2)")
    while λ < 1e8
        λ=λ*1e2
        obj = makeObjFun(target, p, st, true,λ)
        function obj∇!(F, ∇, x)
            y, back = pullback(obj, x)
            if ∇ != nothing
                ∇[:] = back(y)[1]
            end
            if F != nothing
                return y
            end
        end
        try
            res = defaultOptim(obj∇!,ongoing,N,timeAl, allowIncrease, algo,
                               lineSearch; kwargs...)
            return res, lineSearch, λ
        catch e2
            if isa(e2, AssertionError)
                res, lineSearch, λ = catchAndSwitchLineSearch(e2, obj∇!,
                                                              ongoing, N,
                                                              target, p, st, λ,
                                                              timeAl,
                                                              allowIncrease,
                                                              algo,
                                                              lineSearch; 
                                                              kwargs...)
                return res, lineSearch, λ
            elseif isa(e2, ArgumentError)
                if λ < 1e3
                    println("still broken, upping to $(λ*1e2)")
                else
                    println("well, it seems even $λ isn't large enough.")
                    return nothing, nothing, nothing
                end
            else
                rethrow(e2)
            end
        end
    end
end
