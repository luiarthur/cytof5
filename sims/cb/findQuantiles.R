b1 = c(-48.83529072724049, -41.216229461896795, -8.40342517660578 )
b2 = c(-48.127018779059114, -36.976153355118164, -6.867335260498027 )
b3 = c(-41.1431768602219, -36.11463436333189, -7.6163134079426 )
get_q = function(b) {
  pmiss = function(y, b) 1 / (1 + exp(-sum(y^(0:2) * b)))
  y = seq(-5, 0, len=1E4)
  p = sapply(y, function(yi) pmiss(yi, b))
  y_outer = sort(y[head(order(abs(p - .05)), 2)])
  y_center = y[which.min(abs(p - .8))]
  y_quantiles = c(y_outer[1], y_center, y_outer[2])
  y_quantiles
}

print(get_q(b1))
print(get_q(b2))
print(get_q(b3))


