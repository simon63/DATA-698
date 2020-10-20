###
title: "DATA-698 Midterm Draft"
author: "Simon U."
date: "10/18/2020"
output: html_document
###


## Libraries
library(caret)
library(ROSE)
library(ggplot2)
library(dplyr)
library(pROC)
library(ROSE)


## Data Preparation
dfpb <-
  read.csv(
    "https://github.com/simon63/DATA-698/blob/master/Midterm%20Draft/part_b_FL.csv?raw=true",
    row.names = 1
  )
glimpse(dfpb)

## Draft Modeling and Performance

# Create class label with P-positive (fraud) and N-negative (non-fraud) factors
dfpb$class = ifelse(dfpb$exclusion == 1, "P", "N")
dfpb$class = as.factor(dfpb$class)
levels(dfpb$class)
contrasts(dfpb$class)
table(dfpb$class)
round(prop.table(table(dfpb$class)), 4)

# Change other categorical columns to factors
dfpb$provider_type = as.factor(dfpb$provider_type)
dfpb$nppes_provider_gender = as.factor(dfpb$nppes_provider_gender)

###################################################################
# Assign weights to model data based on SRVC_YEAR
###################################################################
# Get Geometric Distribution for weight values
dfGeom <- data.frame(matrix(dgeom(c(0:4), 0.8)))
rownames(dfGeom) <- as.character(c(2017:2013))
colnames(dfGeom) <- "prob"
dfGeom

# Generate weights column
dfpb$wts = dfGeom[dfpb$srvc_year, "prob"]
###################################################################

# Perform One-Hot Encoding (using dummy variables for categorical vales)
system.time({dmyVars <- dummyVars(~ provider_type + nppes_provider_gender, dfpb)})

system.time({dfDmy <- data.frame(predict(dmyVars, dfpb))})

glimpse(dfDmy)

# Create data set for use in modeling
dfModel <- cbind(dfpb, dfDmy)
glimpse(dfModel)

# Determine columns not to be used in modeling as predictors
npi_col <- which(colnames(dfModel) == "npi")
nppes_provider_gender_col <- which(colnames(dfModel) == "nppes_provider_gender")
provider_type_col <- which(colnames(dfModel) == "provider_type")
srvc_year_col <- which(colnames(dfModel) == "srvc_year")
exclusion_col <- which(colnames(dfModel) == "exclusion")
class_col <- which(colnames(dfModel) == "class")
wtc_col <- which(colnames(dfModel) == "wts")

# Create training and test split on the data
set.seed(2020)
trainRows <- createDataPartition(dfModel$class, p = 0.8, list=FALSE)

dfTrain <- dfModel[trainRows, ]
dfTest <- dfModel[-trainRows, ]

table(dfTrain$class)
table(dfTest$class)

prop.table(table(dfTrain$class))
prop.table(table(dfTest$class))

system.time({
  ggplot(dfTrain, aes(x = average_submitted_chrg_amt,
                      y = average_medicare_payment_amt)) +
    geom_point(aes(color = class, shape = class)) +
    scale_color_manual(values = c('dodgerblue', 'red'))
})
##############################################################################
# ROS - manage imbalance in data using Random Over Sampling technique #
##############################################################################
n_legit <- table(dfTrain$class)[[1]]
new_frac_legit <- 0.60
new_n_total <- round(n_legit / new_frac_legit)
ros_result <- ovun.sample(formula = class ~.,
                          data = dfTrain[, -c(npi_col,
                                              provider_type_col,
                                              nppes_provider_gender_col,
                                              srvc_year_col,
                                              exclusion_col)],
                          method = "over",
                          N = new_n_total,
                          seed = 2020)
dfTrainData <- ros_result$data
glimpse(dfTrainData)
nrow(dfTrainData)
round(prop.table(table(dfTrainData$class)), 2)
# Make a scatter plot
ggplot(dfTrainData, aes(x = average_submitted_chrg_amt,
                        y = average_medicare_payment_amt)) +
  geom_point(aes(color = class, shape = class)) +
  scale_color_manual(values = c('dodgerblue', 'red'))
###################################################################

###################################################################
# Logistic Regression
###################################################################
dfTrainX <- dfTrainData[, -c(which(colnames(dfTrainData) == "class"),
                             which(colnames(dfTrainData) == "wts"))]

dfTrainY <- dfTrainData$class

glimpse(dfTrainX)

ctrl <- trainControl(method = "repeatedcv",
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE,
                     number = 5,
                     repeats = 2,
                     p = 0.8)

system.time({
  lrModel <- train(x = dfTrainX, 
                   y = dfTrainY,
                   method = "glm",
                   metric = "ROC",
                   trControl = ctrl)
})
lrModel
summary(lrModel)

lrModel$pred <- merge(lrModel$pred, lrModel$bestTune)
lrModel$pred
glimpse(lrModel$pred)
postResample(lrModel$pred$pred, lrModel$pred$obs)

# Get the confusion matrices for the hold-out set
confusionMatrix(lrModel$pred$pred, lrModel$pred$obs, positive = "P")
lrCM <- confusionMatrix(lrModel, norm = "none")
lrCM

# Get the area under the ROC curve for the hold-out set
lrRoc <- roc(response = lrModel$pred$obs,
             predictor = lrModel$pred$P,
             levels = rev(levels(lrModel$pred$obs)))

plot(lrRoc, legacy.axes = TRUE)

###################################################################
# Run the same GLM model with weights
system.time({
  lrModl2 <- train(x = dfTrainX, 
                   y = dfTrainY,
                   method = "glm",
                   metric = "ROC",
                   trControl = ctrl,
                   weights = dfTrainData$wts)
})
lrModl2
summary(lrModl2)

lrModl2$pred <- merge(lrModl2$pred, lrModl2$bestTune)
lrModl2$pred
glimpse(lrModl2$pred)
postResample(lrModl2$pred$pred, lrModl2$pred$obs)

# Get the confusion matrices for the hold-out set
confusionMatrix(lrModl2$pred$pred, lrModl2$pred$obs, positive = "P")
lrCM2 <- confusionMatrix(lrModl2, norm = "none")
lrCM2

# Get the area under the ROC curve for the hold-out set
lrRoc2 <- roc(response = lrModl2$pred$obs,
              predictor = lrModl2$pred$P,
              levels = rev(levels(lrModl2$pred$obs)))

plot(lrRoc2, legacy.axes = TRUE)
###################################################################

auc_folds_reps <- function(mdl, nFolds, nReps) {
  result <- vector(mode = "double", length = nFolds * nReps)
  lvls <- rev(levels(mdl$pred$obs))
  i <- 0
  for (f in seq(1:nFolds)) {
    for (r in seq(1:nReps)) {
      sFold <- paste("Fold", f, sep = "")
      sRep <- paste("Rep", r, sep = "")
      sResample <- paste(sFold, sRep, sep = ".")
      pred <- mdl$pred[mdl$pred$Resample == sResample, ]
      mdlRoc <-
        roc(
          response = pred$obs,
          predictor = pred$P,
          levels = lvls
        )
      mdlAuc <- auc(mdlRoc)
      i <- i + 1
      result[i] = mdlAuc
    }
  }
  return(result)
}

auc1 <- auc_folds_reps(lrModel, 5, 2)
auc2 <- auc_folds_reps(lrModl2, 5, 2)

aucData <- data.frame(rbind(cbind(auc1, rep("LR", length(auc1))),
                 cbind(auc2, rep("LRw", length(auc2)))))
names(aucData) <- c("AUC", "Learner")
aucData$AUC <- as.numeric(aucData$AUC)
aucData$Learner <- as.factor(aucData$Learner)

boxplot(AUC~Learner, data = aucData, outline = FALSE)

# Compute the confidence interval of the AUC
ci.auc(lrRoc)
ci.auc(lrRoc2)

anova.test <- aov(formula = AUC ~ Learner, data = aucData)
summary(anova.test)

################################################################################
### Random Forests
################################################################################

rfctrl <- trainControl(
  method = "repeatedcv",
  summaryFunction = twoClassSummary,
  classProbs = TRUE,
  savePredictions = TRUE,
  number = 2,
  repeats = 1,
  p = 0.8
)

system.time({
  rfFit1 <- train(
    x = dfTrainX,
    y = dfTrainY,
    method = "rf",
    ntree = 100,
    tuneGrid = expand.grid(mtry = floor(sqrt(ncol(
      dfTrainX
    )))),
    importance = TRUE,
    metric = "ROC",
    trControl = rfctrl
  )
})

rfFit1
summary(rfFit1)

rfFit1$pred <- merge(rfFit1$pred, rfFit1$bestTune)
rfFit1$pred
glimpse(rfFit1$pred)

postResample(rfFit1$pred$pred, rfFit1$pred$obs)

# Get the confusion matrices for the hold-out set
confusionMatrix(rfFit1$pred$pred, rfFit1$pred$obs, positive = "P")
rfCM1 <- confusionMatrix(rfFit1, norm = "none")
rfCM1

# Get the area under the ROC curve for the hold-out set
rfRoc1 <- roc(response = rfFit1$pred$obs,
              predictor = rfFit1$pred$P,
              levels = rev(levels(rfFit1$pred$obs)))

plot(rfRoc1, legacy.axes = TRUE)

gc()
################################################################################
# Run the same model with weights
system.time({
  rfFit2 <- train(
    x = dfTrainX,
    y = dfTrainY,
    method = "rf",
    ntree = 100,
    tuneGrid = expand.grid(mtry = floor(sqrt(ncol(
      dfTrainX
    )))),
    importance = TRUE,
    metric = "ROC",
    trControl = rfctrl,
    weights = dfTrainData$wts
  )
})

rfFit2
summary(rfFit2)

rfFit2$pred <- merge(rfFit2$pred, rfFit2$bestTune)
rfFit2$pred
glimpse(rfFit2$pred)

postResample(rfFit2$pred$pred, rfFit2$pred$obs)

# Get the confusion matrices for the hold-out set
confusionMatrix(rfFit2$pred$pred, rfFit2$pred$obs, positive = "P")
rfCM2 <- confusionMatrix(rfFit2, norm = "none")
rfCM2

# Get the area under the ROC curve for the hold-out set
rfRoc2 <- roc(response = rfFit2$pred$obs,
              predictor = rfFit2$pred$P,
              levels = rev(levels(rfFit2$pred$obs)))

plot(rfRoc2, legacy.axes = TRUE)

gc()
################################################################################