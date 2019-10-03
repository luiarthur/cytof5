import BSON
import JSON
import Random

Random.seed!(0)

struct Bob
  x::String
  y::Vector{Float64}
end

bob = Bob("Arthur", randn(3))

save_path = "results/julia_bob.bson"
BSON.bson(save_path, bob=bob)

bob_in = BSON.load(save_path)

# TODO: Try JSON?
# jbob = JSON.json(bob)
# jbob
