using Arrows
import Arrows: psl, supervised, traceprop!, UnknownArrow, simpletracewalk, trace_values, supervisedloss, AbValues
import Arrows: in_trace_values, out_trace_values

function test_psl(arr::Arrow, batch_size=64)
  orig = deepcopy(arr)
  invarr = aprx_invert(arr)
  pslarr = psl(invarr)
  superarr = supervised(orig, pslarr)
  suploss = supervisedloss(superarr)
  insizes = Dict(sprt => AbValues(:size => Size([batch_size, 10, 10])) for sprt in ▹(suploss))
  abtvals = traceprop!(suploss, insizes)
  nnettarr = first(filter(tarr -> deref(tarr) isa UnknownArrow, simpletracewalk(x->x, suploss)))
  @show insizes = [abtvals[tval][:size]  for tval in in_trace_values(nnettarr)]
  @show outsizes = [abtvals[tval][:size]  for tval in out_trace_values(nnettarr)]
  insizes, outsizes
end

# function test_mlp_run()
