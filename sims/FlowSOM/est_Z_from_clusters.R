relabel_clusters = function(clusters) {
  clusterLabelsOrdered = clusters
  table_clusters = table(clusters)
  K = length(table_clusters)
  ord = order(table_clusters, decreasing=TRUE)
  ord = as.numeric(names(table_clusters[ord]))
  for (k in 1:K) {
    clusterLabelsOrdered[which(clusters == ord[k])] = k
  }
  clusterLabelsOrdered
}

est_ZW_from_clusters = function(yi, clusters, f=mean) {
  J = NCOL(yi)

  ### Clusters ###
  clusters = relabel_clusters(clusters)
  K = length(unique(clusters))

  ### Cluster Means ###
  mu = matrix(0, J, K)
  for (k in 1:K) {
    yik = yi[clusters == k,]
    if (NCOL(yik) > 1) {
      mu[,k] = apply(yik, 2 ,f)
    } else {
      mu[,k] = yik
    }
  }

  ### Z ###
  Z = mu > 0

  ### W ###
  table_clusters = table(clusters)
  W = table_clusters / sum(table_clusters)

  list(clusters=clusters, Z=Z, W=W, mu=mu)
}
