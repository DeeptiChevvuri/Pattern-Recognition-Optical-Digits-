install.packages("neuralnet")
library(neuralnet)
Var1 <- rpois(100,0.5)
Var2 <- rbinom(100,2,0.6)
Var3 <- rbinom(100,1,0.5)
SUM <- as.integer(abs(Var1+Var2+Var3+(rnorm(100))))
sum.data <- data.frame(Var1+Var2+Var3, SUM)
print(net.sum <- neuralnet( SUM~Var1+Var2+Var3, sum.data, hidden=1, act.fct="tanh"))
Call: neuralnet(formula = SUM ~ Var1 + Var2 + Var3, data = sum.data,     hidden = 1, act.fct = "tanh")

library(RSNNS)
data(iris)
#shuffle the vector
iris <- iris[sample(1:nrow(iris),length(1:nrow(iris))),1:ncol(iris)]
irisValues <- iris[,1:4]
irisTargets <- decodeClassLabels(iris[,5])[,'setosa']
iris <- splitForTrainingAndTest(irisValues, irisTargets, ratio=0.15)
iris <- normTrainingAndTestSet(iris)
model <- mlp(iris$inputsTrain, iris$targetsTrain, size=5, learnFuncParams=c(0.1),
             maxit=50, inputsTest=iris$inputsTest, targetsTest=iris$targetsTest)
predictions <- predict(model,iris$inputsTest)
# Generating prediction intervals
library(nnetpredint)
# S3 Method for rsnns class prediction intervals
xTrain <- iris$inputsTrain
yTrain <- iris$targetsTrain
newData <- iris$inputsTest
yPredInt <- nnetPredInt(model, xTrain, yTrain, newData)
print(yPredInt[1:20,])

nn <- neuralnet(f,data = train_,hidden = c(5,3),linear.output = FALSE)
plot(nn)