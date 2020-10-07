# the fully specified version
# TODO: use a layered transform parameter to determine if we're returning a st or thinst instead
@doc """
    layers = stParallel(m, Xlength; CWTType=morl, subsampling = 2, 
                    outputSize=[1,1], varargs...)
 """
function stParallel(m, Xlength; CWTType=morl, subsampling = 2, 
                    outputSize=[1,1], varargs...)
    CWTType = makeTuple(m+1, CWTType)
    subsampling = makeTuple(m+1, subsampling)
    pairedArgs = processArgs(m+1, varargs)
    shears = map(x->wavelet(x[1]; x[2]...), zip(CWTType,pairedArgs))
    # @info "Treating as a 1D Signal. Vector Lengths: $Xlength nScales:" *
    #         "$nScales subsampling: $subsampling"
    stParallel{typeof(shears[1]), 1, m, typeof(subsampling), typeof(outputSize)}((Xlength,), shears, subsampling, outputSize)
end
