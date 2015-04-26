
## Introduction



Machine Learning Course Final Project

These are the files produced during a homework assignment of Coursera’s Machine Learning taught by Jeff Leek. 

Created by: Daryl Morris

GitHub repo: https://github.com/demorrisfhcrc/Machine-Learning-Coursera_DEM

##Background Introduction

From the course website, we find the following project synopsis: 
“Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). ”


##Data Sources

The training data for this project are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment. 

##Submission Requirements:

The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases. 

1. Your submission should consist of a link to a Github repo with your R markdown and compiled HTML file describing your analysis. Please constrain the text of the writeup to < 2000 words and the number of figures to be less than 5. It will make it easier for the graders if you submit a repo with a gh-pages branch so the HTML page can be viewed online (and you always want to make it easy on graders :-).
2. You should also apply your machine learning algorithm to the 20 test cases available in the test data above. Please submit your predictions in appropriate format to the programming assignment for automated grading. See the programming assignment for additional details. 



## Loading and preprocessing the data

The following code loads required libraries, downloads, and reads the training and testing sets.


```r
library(ggplot2)
```

```
## Find out what's changed in ggplot2 with
## news(Version == "1.0.0", package = "ggplot2")
```

```r
library(caret)
```

```
## Loading required package: lattice
```

```r
temp <- tempfile()
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",temp,method="curl")
training <- read.csv(temp,na.strings=c("#DIV/0!","NA",""))
unlink(temp)

temp <- tempfile()
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",temp,method="curl")
testing <- read.csv(temp,na.strings=c("#DIV/0!","NA",""))
unlink(temp)
```

The following code investigates missingness.  It was found that most of the variables were missing for more than 19216 of the rows in the training data, whereas 60 variables were complete.  The analysis below restricts models to just those 60 predictors.  The code below also sub-splits the training data into myTraining and myTest sets so that we can estimate the performance of the model.


```r
# only 60 variables have missingness for < 19216 rows, and those 60 have 0 missingness
whichIncomplete = which(apply(training,2,function(x) sum(is.na(x))) > 0)

# now down to 60 predictors.
training = training[,-whichIncomplete]

# set seed and subpartition training data in myTraining and myTesting
set.seed(54321)

inTraining <- createDataPartition(y=training$classe, p=0.6, list=FALSE)
myTraining <- training[inTraining, ]; 
myTesting <- training[-inTraining, ]
```


#Principle Component Analysis
To reduce the number of predictors down, we will pre-process by principle component analysis.  We will reduce the number of predictors to 10 principle components, and all models fit below will only use those 10 predictors.


```r
# pre-processing with PCA
#    for this, we'll remove the factor variables
factorVars = which("factor"==unlist(lapply(names(training),function(x) is(training[[x]])[1])))
addVarsToDrop = which(names(training) %in% c("X","raw_timestamp_part_1","raw_timestamp_part_2","num_window"))
preProc <- preProcess(myTraining[,-c(addVarsToDrop,factorVars)],method="pca",pcaComp=10)
trainPC = predict(preProc,myTraining[,-c(addVarsToDrop,factorVars)])

testPC = predict(preProc,myTesting[,-c(addVarsToDrop,factorVars)])
```


#Model fitting
Various models are fit below.  After each model we create predictions on the myTesting set.  


```r
library(rpart)
# try a decision tree first (with PCA predictors)
modFitDT <- train(myTraining$classe ~ ., method="rpart",data=trainPC)

#print(modFitDT$finalModel)
#fancyRpartPlot(modFitDT$finalModel)

predictionsDT = predict(modFitDT,testPC)

# boosting (trees)
library(gbm)
library(survival)
library(splines)
library(parallel)
library(plyr)
modFitGM <- train(myTraining$classe ~ ., method="gbm",data=trainPC,verbose=FALSE)

#print(modFitGM)

predictionsGBM = predict(modFitGM,testPC)

# linear discriminant analysis
library(MASS)
modFitLDA <- train(myTraining$classe ~ ., method="lda",data=trainPC)

#print(modFitLDA)

predictionsLDA = predict(modFitLDA,testPC)

# naive bayes (had too many warnings for bad fit)
library(klaR)
#modFitNB <- train(myTraining$classe ~ ., method="nb",data=trainPC)

#print(modFitNB)
#predictionsNB = predict(modFitNB,testPC)

save.image("Mach_Learn_Model.RData")
```

## Attempt to make an ensemble model
Since none of the models were performing very well, I decided to try an ensemble method.  The code below does that.


```r
#### Let's ensemble models
##      using naive bayes, LDA, and decision tree

pDT  = predict(modFitDT,trainPC)
pLDA = predict(modFitLDA,trainPC)
pGBM = predict(modFitGBM,trainPC)
```

```
## Error: object 'modFitGBM' not found
```

```r
predDF = data.frame(pDT,pLDA,pGBM) #,classe=myTraining$classe)
```

```
## Error: object 'pGBM' not found
```

```r
combModFit <- train(myTraining$classe ~ ., method="gam",data=predDF)


pDT_test  = predict(modFitDT,testPC)
pLDA_test = predict(modFitLDA,testPC)
pGBM_test = predict(modFitGBM,testPC)
```

```
## Error: object 'modFitGBM' not found
```

```r
predDF_test = data.frame(pDT=pDT_test,pLDA=pLDA_test,pGBM=pGBM_test) #,classe=myTesting$classe)
```

```
## Error: object 'pGBM_test' not found
```

```r
predictionsCombo = predict(combModFit,predDF_test)
```

## Performance on test set for various models
Confusion matrices were calculated below but not included except for the chosen model in the interest of keeping the report short.  Ulimately, the best performing model by accuracy was the GBM ("Gradient Boosting Machine") with an accuracy in my testing set of .73 (0.72, 0.739).  None of the other models had accuracy above 50% and the ensemble method performed awfully.  Hence the GBM method was chosen to run on the final testing set.



```r
#confusionMatrix(predictionsDT, myTesting$classe)
confusionMatrix(predictionsGBM, myTesting$classe)
```

Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1758  144  187   90  112
         B  110 1029   83   66  140
         C  135  146 1022  144  124
         D  182   89   35  939   89
         E   47  110   41   47  977

Overall Statistics
                                       
               Accuracy : 0.73         
                 95% CI : (0.72, 0.739)
    No Information Rate : 0.284        
    P-Value [Acc > NIR] : <2e-16       
                                       
                  Kappa : 0.658        
 Mcnemar's Test P-Value : <2e-16       

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity             0.788    0.678    0.747    0.730    0.678
Specificity             0.905    0.937    0.915    0.940    0.962
Pos Pred Value          0.767    0.721    0.651    0.704    0.800
Neg Pred Value          0.915    0.924    0.945    0.947    0.930
Prevalence              0.284    0.193    0.174    0.164    0.184
Detection Rate          0.224    0.131    0.130    0.120    0.125
Detection Prevalence    0.292    0.182    0.200    0.170    0.156
Balanced Accuracy       0.846    0.807    0.831    0.835    0.820

```r
#confusionMatrix(predictionsLDA, myTesting$classe)
#confusionMatrix(predictionsCombo, myTesting$classe)
```


## Accuracy of the chosen model on the testing set
This section makes predictions on the final testing set.  The course requirements mandated that each prediction be written to a separate file.


```r
testingForMod = testing[,-whichIncomplete]

finalTestPC = predict(preProc,testingForMod[,-c(addVarsToDrop,factorVars)])

pGBM_finalTest = predict(modFitGM,finalTestPC)

#is(pGBM_finalTest)

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
 
pml_write_files(pGBM_finalTest)
```
