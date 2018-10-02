### INSTALL ###
#source("https://bioconductor.org/biocLite.R")
#biocLite("FlowSOM")

library(xtable)
library(FlowSOM)
library(flowCore)
library(rcommon)
library(cytof3)
source('est_Z_from_clusters.R')
set.seed(3)

OUT_SIM  = '../../out/'
OUT_FLOW = OUT_SIM %+% 'FlowSOM/'

cytof3.results = OUT_SIM %+% '/sim_rand_beta_K20_N1000/checkpoint.rda'
load(cytof3.results)

### Indices for each sample ###
idx_upper = cumsum(N)
idx_lower = c(1,idx_upper[-I]+1)
idx = cbind(idx_lower, idx_upper)

### Set column names  ###
for (i in 1:I) colnames(y[[i]]) = paste0('V',1:J)

### Transform to y tilde ###
y_tilde = lapply(y, function(yi) {
  yi[which(is.na(yi))] = -20#-7 #min(yi, na.rm=T)
  yi
})
#my.image(y_tilde[[1]], col=blueToRed(5), zlim=c(-3,3), xlab='cells', ylab='markers', addL=T)

ff_y = sapply(y_tilde, flowFrame)
Y_tilde = Reduce(rbind, y_tilde)
ff_Y = flowFrame(Y_tilde)

### Example ###
# http://bioconductor.org/packages/release/bioc/vignettes/FlowSOM/inst/doc/FlowSOM.pdf

### FlowSOM ###
println("Running FlowSOM...")
tim = system.time(
fSOM <- FlowSOM(ff_Y,
                # Input options:
                colsToUse = 1:J,
                # Metaclustering options:
                #nClus = 20,
                maxMeta=20,
                # Seed for reproducible results:
                seed = 42)
)
print(tim)
#PlotStars(fSOM$FlowSOM, backgroundValues = as.factor(fSOM$metaclustering))
fSOM.clus = fSOM$meta[fSOM$FlowSOM$map$mapping[,1]]


fs.est = as.list(1:I)
fs.clus = as.numeric(fSOM.clus)
### Kmeans ###
#fs.clus = kmeans(Y_tilde, 10)$clus

#for (i in 1:I) {
#  clus = fs.clus[idx[i,1]:idx[i,2]]
#  fs.est[[i]] = est_ZW_from_clusters(y_tilde[[i]], clus)
#  yZ(yi=y[[i]], Zi=fs.est[[i]]$Z*1, Wi=fs.est[[i]]$W,
#     cell_types_i=fs.est[[i]]$clus-1, zlim=c(-3,3), na.color='black', thresh=.9)
#}

mult=1
png(OUT_FLOW %+% 'YZ%03d_FlowSOM_SIM.png', height=500*mult, width=500*mult, type='Xlib')
#source("../../../cytof3/R/yZ.R")
#source("../../../cytof3/R/myimage.R")
for (i in 1:I) {
  clus = fs.clus[idx[i,1]:idx[i,2]]
  print(length(unique(clus))) # Number of clusters learned
  #est = est_ZW_from_clusters(y_tilde[[i]], clus, f=median)
  #yZ(yi=y[[i]], Zi=est$Z*1, Wi=est$W, cell_types_i=est$clus-1,
  #   zlim=c(-3,3), na.color='black', thresh=.9, col=blueToRed(7),
  #   cex.z.b=1.5, cex.z.lab=1.5, cex.z.l=1.5, cex.z.r=1.5,
  #   cex.y.ylab=1.5, cex.y.xaxs=1.4, cex.y.yaxs=1.4, cex.y.leg=1.5, 
  #   prop_lower_panel=0)
  clus = relabel_clusters(clus)
  my.image(y[[i]][order(clus),], col=blueToRed(11), zlim=c(-5,5), addL=TRUE,
           na.color='black', cex.y.leg=2, xlab='cell types',  ylab='cells',
           cex.lab=1.5, cex.axis=1.5, xaxt='n',
           f=function(z) {
             add.cut(clus) 
             axis(1, at=1:J, fg='grey', las=2, cex.axis=1.5)
           })
}
dev.off()


### Principal Components ###
rgba = function(col, a=1) {
  sapply(col, function(co) {
    RGB = col2rgb(co) / 255
    rgb(RGB[1], RGB[2], RGB[3], a)
  })
}

### Compute F Measure to measure how good the clustering is to truth ###
best_idx = sapply(1:I, function(i) estimate_ZWi_index(out,i))
cytof3.clus.ls = lapply(1:I, function(i) out[[best_idx[i]]]$lam[[i]])
fs.clus.ls = lapply(1:I, function(i) fs.clus[idx_lower[i]:idx_upper[i]])
true.clus.ls = dat$lam

println("Computing ARI ...")
ARI = sapply(1:I, function(i) {
  c('fs'=ari(true.clus.ls[[i]], fs.clus.ls[[i]]),
    'cy'=ari(true.clus.ls[[i]], cytof3.clus.ls[[i]]))
})
print(ARI)

ARI.all = round({
  c('FlowSOM'=ari(unlist(true.clus.ls), fs.clus),
    'FAM'=ari(unlist(true.clus.ls), unlist(cytof3.clus.ls)))
}, 4)
print(ARI.all)


#mean(relabel_clusters(cytof3.clus[[1]]) == relabel_clusters(true.clus[[1]]))
#mean(relabel_clusters(fs.clus[[1]]) == relabel_clusters(true.clus[[1]]))

# Plot Clusters in first two Principal Components View ###
ps = prcomp(Y_tilde)$sd
pY = prcomp(Y_tilde)$x[,1:2]
fs.c = relabel_clusters(fs.clus)
cy.c = relabel_clusters(unlist(cytof3.clus.ls))
pdf(OUT_FLOW %+% 'compareClus_FlowSOM_pc2.pdf')
par(mfrow=c(2,2))
plot(pY, main="FlowSOM Clusters", col=rgba(fs.c, .5), pch=16) # appears to be 8 clusters
plot(pY, main="FAM Clusters", col=rgba(cy.c, .5), pch=16) # appears to be 8 clusters
plot(pY, main="Data: First two PC", pch=16, col=rgba('steelblue',.5)) # appears to be 8 clusters
plot(pY, main="Differences", col=ifelse(abs(fs.c-cy.c)==0, 'transparent', 'red'), pch=16)
par(mfrow=c(1,1))
dev.off()


# Timings: cytof3 (39h),  FlowSOM (13s)
compare.results = cbind('ARI'=ARI.all,
                        'Elapsed time (seconds)'=c(tim[3], st[3]))

sink(OUT_FLOW %+% 'timings_FlowSOM_vs_SIM.tex')
xtab = xtable(compare.results, digits=c(0,3,0))
print(xtab, floating=F)
sink()



### Plot Comparison of Clusters ###
cy.c.cs = cumsum(table(cy.c) / sum(table(cy.c)))
fs.c.cs = cumsum(table(fs.c) / sum(table(fs.c)))

### Plot Cumulative Proportion by Clusters ###
pdf(OUT_FLOW %+% 'compareClus_FlowSOM.pdf')
plot(cy.c.cs, type='o', col=rgba('blue', .5), fg='grey', ylim=0:1,
     xlab='cell-types', cex.lab=1.4, cex.axis=1.5,
     ylab='cumulative  proportion', lwd=5)
lines(fs.c.cs, type='o', col=rgba('red', .5), lwd=5)
abline(h=1:10 / 10, v=1:last(out)$prior$K, lty=2, col='grey')
legend('bottomright', col=c('red', 'blue'), legend=c('FlowSOM', 'FAM'), cex=3,
       lwd=5, bty='n')
dev.off()


### UQ on Z / W ###
Z = lapply(out, function(o) o$Z)
my.image(matApply(Z, mean), xlab='cell types', ylab='markers', col=greys(5), addL=TRUE)
my.image(matApply(Z, sd),   xlab='cell types', ylab='markers', col=greys(4), addL=TRUE)

W = lapply(out, function(o) o$W)
cc=5; M = cbind(matrix(rep(1:3,cc), ncol=cc), 4)
layout(M)
my.image(matApply(W, quantile, .975), zlim=c(0,.4), xlab='cell types',
         ylab='markers', col=greys(6), main='97.5% Quantile')
my.image(matApply(W, mean), zlim=c(0,.4), xlab='cell types', ylab='markers',
         col=greys(4), main='mean')
my.image(matApply(W, quantile, .025), zlim=c(0,.4), xlab='cell types',
         ylab='markers', col=greys(6), main='2.5% Quantile')
color.bar(greys(5), zlim=c(0,.4), dig=3)
par(mfrow=c(1,1))


