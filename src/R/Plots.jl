module Rplots

using RCall
const RSOURCE_DIR = "$(pwd())/src/R"
const RPLOT_SOURCE = "$(RSOURCE_DIR)/plots.R"
rplotlib() = R"source($RPLOT_SOURCE, chdir=TRUE)"

end # Rplots

#= TEST
using Cytof5
Cytof5.Rplots.RSOURCE_DIR
Cytof5.Rplots.rplotlib()
=#
