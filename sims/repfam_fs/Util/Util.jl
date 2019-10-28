module Util

s3sync(from, to) = run(`aws s3 sync $(from) $(to)`)
s3sync(from, to, tags) = run(`aws s3 sync $(from) $(to) $(tags)`)

end
