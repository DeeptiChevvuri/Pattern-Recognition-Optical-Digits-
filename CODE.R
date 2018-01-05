#loading h2o package
library(h2o)
#Attempts to start and/or connect to and H2O instance.
h2o = h2o.init(ip = "localhost", port = 54321, startH2O = TRUE,min_mem_size = "3g")
trainingFilepath = "http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra"
train = h2o.importFile(path = trainingFilepath) 
#splitting the data 80% as training 20% as validation data set
splits <- h2o.splitFrame(train, 0.8,seed=1234)
#first 80% training set
train  <- h2o.assign(splits[[1]], "train.hex")
# remaining 20% validation set
valid  <- h2o.assign(splits[[2]], "valid.hex")
#setting the last colun i,e, C65 as a categorical coloum (to solve classification problem)
train[,ncol(train)] <- as.factor(train[,ncol(train)]) 
# test file location
testFilepath = "http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes"
testprostate.hex = h2o.importFile(path = testFilepath, destination_frame = "prostatetest.hex") 
#testing frame
testFrame <- as.data.frame(testprostate.hex)
test_h2o <- as.h2o(testFrame)
#setting the last colun i,e, C65 as a categorical coloum (to solve classification problem)
test_h2o[,ncol(test_h2o)] <- as.factor(test_h2o[,ncol(test_h2o)])
#building the deeplearning model with Cross Entrophy Loss function
modeltanhCrossEntro <- h2o.deeplearning(x = 1:64,
                          y = 65,
                          training_frame = train,
                          validation_frame=valid,
                          activation = "Tanh", 
                            loss = "CrossEntropy",
                            hidden = c(300,100,300),
                            epochs = 500,
                            stopping_tolerance=0.01,
                            rate=0.0003,
                            standardize=TRUE,
                            adaptive_rate=FALSE,
                            momentum_start = 0.2, 
                          momentum_ramp =10000,
                          momentum_stable = 0.7
                          )
modeltanhSSE <- h2o.deeplearning(x = 1:64,
                          y = 65,
                          training_frame = train,
                          validation_frame=valid,
                          activation = "Tanh", 
                          loss = "Quadratic",
                          hidden = c(200,100,200),
                          epochs = 500,
                          stopping_tolerance=0.1,
                          rate=0.001,
                          standardize=TRUE,
                          seed=10,
                          adaptive_rate=FALSE,
                          momentum_start = 0.1,
                          momentum_ramp =10000,
                          momentum_stable = 0.7
                          
)
modeltanhCrossEntro
modeltanhSSE
# Confusion Matrix for modeltanhCrossEntro
h2o.confusionMatrix(modeltanhCrossEntro, test_h2o)
# Convergence Speed for modeltanhCrossEntro
modeltanhCrossEntro@model$scoring_history
# Confusion Matrix for modeltanhSSE
h2o.confusionMatrix(modeltanhSSE, test_h2o)
# Convergence Speed for modeltanhSSE
modeltanhSSE@model$scoring_history
#############################################################################################
modeltanh <- h2o.deeplearning(x = 1:64,
                                 y = 65,
                                 training_frame = train,
                                 validation_frame=valid,
                                 activation = "Tanh", 
                                 loss = "CrossEntropy",
                                 hidden = c(200,100,100,200),
                                 epochs = 500, 
                                 stopping_tolerance=0.01,
                                 rate=0.01,
                              standardize=TRUE,
                              seed=1,
                              adaptive_rate=FALSE,
                              momentum_start = 0.1,
                              momentum_ramp =10000,
                              momentum_stable = 0.7
                                 
)
modelRelu <- h2o.deeplearning(x = 1:64,
                           y = 65,
                           training_frame = train,
                           validation_frame=valid,
                           activation = "Rectifier",  
                           loss = "CrossEntropy",
                           hidden = c(200,100,200),
                           epochs = 500,
                           stopping_tolerance=0.01,
                           rate=0.005,
                           standardize=TRUE,
                           adaptive_rate=FALSE,
                           seed=1,
                           momentum_start = 0.1,
                           momentum_ramp =10000,
                           momentum_stable = 0.7
                           
)
modeltanh
modelRelu 
# Confusion Matrix for modeltanh
h2o.confusionMatrix(modeltanh, test_h2o)
# Convergence Speed for modeltanh
modeltanh@model$scoring_history
# Confusion Matrix for modelRelu
h2o.confusionMatrix(modelRelu, test_h2o)
# Convergence Speed for modelRelu
modelRelu@model$scoring_history
