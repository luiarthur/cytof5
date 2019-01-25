data = read.csv('data/data.csv')
mod = glm(y ~ x, data=data, family='binomial')
print(summary(mod))
