# Questions

- [X] `+=` vs `x = x + a`?
- [ ] `(yi_imputed, log_qyi)` in `vae.jl`
- [ ] `accumulate` turns `TrackedArray` to `Array` of `TrackedReals`
    - [ ] implement `cumprod`: `accumulate(Base.mul_prod, x; dims)`
    - [ ] implement `cumsum`: `accumulate(Base.add_sum, x; dims)`
- [ ] Try expanding dims of `delta0`, `delta1`, `W`, `eta0`, `eta1`
- [ ] remove asserts
