# Classic: K-NN Iris Dataset
# (tuned K-NN with k-fold cross-validation using caret package.)
# Last Modified: October 16, 2018

# (http://archive.ics.uci.edu/ml/datasets/Iris for attribute data)
# Attribute Information:
#
# 1. sepal length in cm
# 2. sepal width in cm
# 3. petal length in cm
# 4. petal width in cm
# 5. class:
# -- Iris Setosa
# -- Iris Versicolour
# -- Iris Virginica

# install.packages("caTools")
# install.packages("caret")
# install.packages("e1071")
# install.packages("ggplot2")

library(caTools)
library(caret)
library(e1071)
library(ggplot2)


dataset <- read.csv('iris.csv', header = FALSE)

cor_matrix <- cor(dataset[-5])
round(cor_matrix, 2)

#       V1    V2    V3    V4
# V1  1.00 -0.11  0.87  0.82
# V2 -0.11  1.00 -0.42 -0.36
# V3  0.87 -0.42  1.00  0.96
# V4  0.82 -0.36  0.96  1.00

# There is a negative correlation with feature V2 (sepal width.) It will be removed when training the model.
# Split out the training and test sets.
set.seed(123)
sample <- sample.split(dataset, SplitRatio = 0.8)
train <- subset(dataset, sample == TRUE)
test <- subset(dataset, sample == FALSE)

# set the control parameters and fit the model.
trControl <- trainControl(method = "cv", number = 5)
fit <- train(
  V5 ~ . - V2,
  method = "knn",
  tuneGrid = expand.grid(k = 1:10),
  trControl = trControl,
  metric = "Accuracy",
  data = train
)

fit

# k-Nearest Neighbors
#
# 120 samples
# 4 predictor
# 3 classes: 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'
#
# No pre-processing
# Resampling: Cross-Validated (5 fold)
# Summary of sample sizes: 96, 96, 96, 96, 96
# Resampling results across tuning parameters:
#
#   k   Accuracy   Kappa
# 1  0.9500000  0.9250
# 2  0.9583333  0.9375
# 3  0.9666667  0.9500
# 4  0.9666667  0.9500
# 5  0.9666667  0.9500
# 6  0.9666667  0.9500
# 7  0.9583333  0.9375
# 8  0.9416667  0.9125
# 9  0.9500000  0.9250
# 10  0.9500000  0.9250
#
# Accuracy was used to select the optimal model using the largest value.
# The final value used for the model was k = 6.

# run the test data through the fitted model
y_pred <- predict(fit, newdata = test)
test$pred <- y_pred

# write.csv(test, 'test.csv')

ggplot(test, aes(
  x = V1,
  y = V2,
  shape = V5,
  color = pred
)) +
  geom_point(size = 4)

confusionMatrix(test$V5, test$pred)

# Confusion Matrix and Statistics
#
#                   Reference
# Prediction        Iris-setosa Iris-versicolor Iris-virginica
# Iris-setosa              10               0              0
# Iris-versicolor           0              10              0
# Iris-virginica            0               1              9
#
# Overall Statistics
#
# Accuracy : 0.9667
# 95% CI : (0.8278, 0.9992)
# No Information Rate : 0.3667
# P-Value [Acc > NIR] : 4.476e-12
#
# Kappa : 0.95
# Mcnemar's Test P-Value : NA     
