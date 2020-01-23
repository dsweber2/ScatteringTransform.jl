if isdefined(Base, :LIBEXECDIR)
  const exe7z = joinpath(Sys.BINDIR, Base.LIBEXECDIR, "7z.exe")
else
  const exe7z = joinpath(Sys.BINDIR, "7z.exe")
end
