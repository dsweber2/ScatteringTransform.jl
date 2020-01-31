if isdefined(Base, :LIBEXECDIR)
  const exe7z = joinpath(Sys.BINDIR, Base.LIBEXECDIR, "7z.exe")
else
  const exe7z = joinpath(Sys.BINDIR, "7z.exe")
end
using FFTW
println("num threads set to 1")
FFTW.set_num_threads(1)
