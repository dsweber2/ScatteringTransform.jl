round(Int,4.51)
log2(10)




Pkg.checkout("Wavelets","")
immutable ContinuousFourierTransform end
const Fourier = ContinuousFourierTransform()

abstract ContinuousWaveletClass <: WaveletClass

for (TYPE, CLASSNAME, NAMEBASE, MOMENTS, SUPERCLASS) in (
        (:Morlet,   "Morlet",   "morl", 0, :ContinuousWaveletClass),
        (:Paul,     "Paul",     "paul", 0, :ContinuousWaveletClass), # moments?
        (:DOG,      "DOG",      "dog",  0, :ContinuousWaveletClass), # moments?
        )
    @eval begin
        immutable $TYPE <: $SUPERCLASS end
        class(::$TYPE) = string($CLASSNAME)::ASCIIString
        name(::$TYPE) = string($NAMEBASE)::ASCIIString
        vanishingmoments(::$TYPE) = $MOMENTS
    end
    CONSTNAME = symbol(NAMEBASE)
    @eval begin
        const $CONSTNAME = $TYPE()                  # type shortcut
    end
end

immutable CFW <: ContinuousWavelet
    sparam # TODO transform definition
    fourierfactor
    coi
    daughterfunc
    name    ::ASCIIString
    CFW(sparam, fourierfactor, coi, daughter_func, name) = new(sparam, fourierfactor, coi, daughter_func, name)
end

function CFW{WC<:WT.ContinuousWaveletClass}(w::WC)
    name = WT.name(w)
    tdef = get(CONT_DEFS, name, nothing)
    tdef == nothing && error("transform definition not found")
    return CFW(tdef..., name)
end

# call example: wavelet(WT.morl) or wavelet(WT.morl, WT.Fourier)
function wavelet(c::WT.ContinuousWaveletClass)
    return wavelet(c, WT.Fourier)
end
function wavelet(c::WT.ContinuousWaveletClass, t::WT.ContinuousFourierTransform)
    return CFW(c)
end

const CONT_DEFS = @compat Dict{ASCIIString,NTuple{4}}(
"morl" => (sparam,
    FourierFactor,
    COI,
    DaughterFunc) # TODO define functions and constants
,
...
)
