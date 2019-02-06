# README

This attempt differs by attempt1 in the following way.

In attempt1, sigma is indexed by `i`.
In attempt3 (this attempt), sigma is indexed by `ell` and `z_jk`.

I find that doing this, sigma is easier to identify.
Need to investigate if there are label switching issues by just
indexing sigma by `i`.

## Update
Realizing that the issue with attempt1 is probably just 
that I need to run it longer, because sigma_i decreases
further.

At this point, it plateaus at 3.5 when it's supposed
to be 1 after 10000 iterations. Could really be label switching.
