library(tidyverse) # ggplot2, pipes
library(gridExtra) # marrangeGrob
library(scales)    # squish

N = 10000
J = 32
K = 8

y <- matrix(rnorm(N * J, sd=3), nrow=N, ncol=J)
y <- matrix(ifelse(runif(N*J) > .05, y, NA), nrow=N, ncol=J)

y_tile <- y %>%
  reshape2::melt() %>%
    ggplot(aes(x=Var2, y=Var1, fill=value, color=value), alpha=1) +
    geom_tile() +
    scale_fill_gradient2(na.value="black",  limits=c(-3, 3), oob=squish) +
    scale_color_gradient2(limits=c(-3, 3)) +
    theme_bw()

plot(y_tile)

z <- matrix(rbinom(J*K, size=1, p=0.5), J, K)
z_tile <- z %>%
  reshape2::melt() %>%
    ggplot(aes(x=Var2, y=Var1, fill=value, color=value)) +
    geom_tile() +
    scale_fill_distiller(palette="Greys") +
    scale_color_distiller(palette="Greys") +
    theme_bw() +
    theme(legend.position="none",panel.grid=element_blank())


gridExtra::marrangeGrob(list(z_tile, y_tile),
                        nrow=1, ncol=3, top=NULL,
                        layout_matrix=matrix(c(1, 2, 2), nrow=1), newpage=F)

