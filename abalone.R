#Predictive Modeling & Machine Learning
# Regression Analysis


data <- read.csv("abalone.csv", header = TRUE)
holdout <- data[(1:floor(nrow(data)*.15)),]
traintest <-data[(floor(nrow(data)*.15):(nrow(data))),]

library(gbm) 
library(randomForest) 

# RANDOM FOREST
maxnumpreds <- 4 
maxnumtrees <- 6 

stats <- data.frame()


samplesize <- nrow(data) 
numfolds <- 10
mse <- as.numeric(numfolds)

quotient <- samplesize %/% numfolds 
remainder <- samplesize %% numfolds 


lstsizes <- rep(quotient,numfolds)
if (remainder > 0) {
  for (i in 1:remainder){
    lstsizes[i] <- lstsizes[i]+1
  }
}

seedval <- 100
for(numpreds in 1 : maxnumpreds){
  for(numtrees in 1 : maxnumtrees) {
    start <- 1
    for(kn in 1:length(lstsizes)){
      end <- start + lstsizes[kn] - 1

      trainsubset <- seq(1:samplesize)[-(start:end)]
      testsubset <- seq(1:samplesize)[start:end]

      set.seed(seedval)
      model.bagged <- randomForest(Rings ~ ., 
                                   data=data, 
                                   subset=trainsubset,
                                   mtry=numpreds, 
                                   ntree=numtrees*50,
                                   importance=TRUE) 
      

      pred.vals.bagged <- predict(model.bagged,
                                  newdata=data[testsubset,])
      testvals <- data$Rings[testsubset]
      mse[kn] <- sqrt(sum(((pred.vals.bagged - testvals)^2)/length(testvals)))
    }
totalmse <- sum(mse)/length(mse)
l <- data.frame("model"="random forest","parameters"=paste("preds",numpreds,"trees",numtrees*50),"mse"=totalmse)
stats <- rbind(stats, l)
  }
  
  print(paste("Processed predictors:", numpreds))
}

# BOOSTED TREES
maxintdepth <- 3 
maxtrees <- 300 

shrinkagevals <- c(0.05, 0.1, 0.2)

for(treeval in seq(50, maxtrees, 50)){
  for(intdepth in seq(1:maxintdepth)){
    for(shrinkval in shrinkagevals){
      
      model <- gbm(Rings ~ .,
                   data = traintest,
                   distribution = "gaussian",
                   n.trees = treeval,
                   interaction.depth = intdepth, 
                   shrinkage = shrinkval,
                   cv.fold=10 
      )
      
      predvals <- predict.gbm(model,newdata = holdout)
      totalmse <- sqrt(sum(((predvals - holdout$Rings)^2)/length(holdout$Rings)))
      l <- data.frame("model"="GBM","parameters"=paste("intdepth",intdepth,"trees",treeval,"shrinkval",shrinkval),"mse"=totalmse)
      stats <- rbind(stats,l)
    }
  }
}

# Linear Regression
lin <- lm(Rings ~ ., data = traintest)
pred <- predict.lm(lin, holdout)
totalmse <- sqrt(sum(((pred - holdout$Rings)^2)/length(holdout$Rings)))
l <- data.frame("model"="linear regression","parameters"= "na","mse"=totalmse)
stats <- rbind(stats,l)
stats <- stats[order(stats$mse),]
stats

pca <- prcomp(data[,c(-1,-9)], scale. = TRUE)
pca <- cbind(as.data.frame(pca$x[,1]), data$Rings)
pca.train <- pca[1:floor(nrow(pca)*0.5),]
pca.test <- pca[floor(nrow(pca)*0.5):nrow(pca),]
colnames(pca.train) <- c("pc1","Rings")
colnames(pca.test) <- c("pc1","Rings")
pca.model <- randomForest(Rings ~ pc1,
                          data=pca.train, 
                          mtry=1, 
                          ntree=150, 
                          importance=TRUE)
pred.pca <- predict(pca.model,pca.test)
totalmse <- sqrt(sum(((pred.pca - pca.test$Rings)^2)/length(pca.test$Rings)))
totalmse
stats[1,]

#Importance stuff
best.rf.model <- randomForest(Rings ~ .,
                       data=data, 
                       mtry=1, 
                       ntree=150, 
                       importance=TRUE)
print(importance(best.rf.model))
best.gbm.model <-gbm(Rings ~ .,
                     data=data,
                     distribution = "gaussian",
                     n.trees = 200,
                     interaction.depth = 3, 
                     shrinkage = 0.2,
                     cv.fold=10)
summary(best.gbm.model)
summary(lin)