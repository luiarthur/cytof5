# My Simple Linter
- [ ] Write a simple Julia linter in Python3 or Julia v1.0
    - [ ] catch usage of undefined variables within scope
        - [ ] e.g. catch errors like this:
        ``` julia
        function f(x::Int)::Int
          return x + y # y is not defined!
        end
        ```
    - [ ] catch bad operations on different types
        - [ ] e.g. catch `1 + "xyz"`
    - [ ] check for function argument types
        - e.g.
        ```julia
        f(x::Int) = x + 1

        # This should be caught:
        f("xyz") # incorrect argument!
        ```
    - [ ] check existence of fields in a class object
        - helpful functions: `fieldnames`
        - e.g.
        ```julia
        struct Bob
          x::Int
          y::Float64
        end

        #= This should be caught:
        b = Bob("a", 2.0) # x should be Int!
        =# 

        b = Bob(2, 4.0)

        # This should be caught:
        b.z # There is no field z!
        ```
    - [ ] Look into imported libraries
        - helpful functions: `names`

