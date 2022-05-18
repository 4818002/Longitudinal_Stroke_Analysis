# The following code can be ran to perform analysis on a cleaned HRS dataset. It includes functions like correlation table, LASSO regression, and AUC curve evaluation.

library(car)
library(haven)
library(corrplot)
library(dplyr)
library(base)
library(forestmodel)

# logistic and lasso regression
library(modelr)   # provides easy pipeline modeling functions
library(broom)    # helps to tidy up model outputs
library(caret)    # variable importance & Lasso regression
library(pscl)     # model evaluation (R^2)
library(ROCR)     # model evaluation (ROC)
library(glmnet)   #lasso
library(tidyverse) #data wrangling
library(DMwR2) #knn imputation
library(skimr) #descriptive
library(boot) #calculate bootstrapped AUC 

data <- HRS_cleandata

data$GENDER<-ifelse(data$GENDER=="1",1,0)
data$RACE1<-ifelse(data$RACE=="1",0,1)
data$T0HEART<-ifelse(data$T0HEART=="1",1,0)
data$T1HEART<-ifelse(data$T1HEART=="1",1,0)
data$T0HIBP<-ifelse(data$T0HIBP=="1",1,0)
data$T1HIBP<-ifelse(data$T1HIBP=="1",1,0)
data$T0LUNG<-ifelse(data$T0LUNG=="1",1,0)
data$T1LUNG<-ifelse(data$T1LUNG=="1",1,0)
data$T0PSYCH<-ifelse(data$T0PSYCH=="1",1,0)
data$T1PSYCH<-ifelse(data$T1PSYCH=="1",1,0)
data$T0DIAB<-ifelse(data$T0DIAB=="1",1,0)
data$T1DIAB<-ifelse(data$T1DIAB=="1",1,0)

# correlation
cor.test(data$T0ADLA,data$T1ADLA)
cor.variables<-c("GENDER","T1AGE","RACE1","ETHNICITY","YEARSEDU","T0IMRC","T0DLRC","T0SER7","T0VOCAB","T0MSTOT","T0CESD","T0ADLA","T0MOBILA","T0LGMUSA","T0GROSSA","T0ATOTN","T0BMI","T0HIBP","T0DIAB","T0LUNG","T0CANCR","T0HEART","T0PSYCH","T1IMRC","T1DLRC","T1SER7","T1VOCAB","T1MSTOT","T1CESD","T1ADLA","T1MOBILA","T1LGMUSA","T1GROSSA","T1ATOTN","T1BMI","T1PSYCH","T3_DefAlive") #you can add more
cor.table<-cor(data[cor.variables],use="pairwise.complete.obs") #排除所有缺失值
head(round(cor.table,2)) #to 2 decimal places
corrplot(cor.table, method="circle")

#IV:Depression,Cognition
#DV:ADL
lm1<-lm(T1ADLA~T0ADLA+T0CESD+T0COGTOT+T1CESD+T1COGTOT,data)
summary(lm1)
vif(lm1) #if VIF > 10, then IVs have very strong correlation
lm2<-lm(T2ADLA~T1ADLA+
          T0CESD+T1CESD+T2CESD+
          T0COGTOT+T1COGTOT+T2COGTOT,data)
summary(lm2)
vif(lm2)
lm3<-lm(T3ADLA~T2ADLA+
          T0CESD+T1CESD+T2CESD+T3CESD+
          T0COGTOT+T1COGTOT+T2COGTOT+T3COGTOT,data)
summary(lm3)
vif(lm3)

# check if trajectory memberships are predictive of ADL/alive
hist(data$Trajectory)
# chronic depression versus all other three groups?
# dummy code
data$chronic<-ifelse(data$Trajectory=="1",1,0)
data$resilience<-ifelse(data$Trajectory == "3",1,0)
hist(data$chronic)
hist(data$resilience)

data<-transform(data, SDLRC=1.5*(T3DLRC-T0DLRC)+0.5*(T2DLRC-T1DLRC), 
                 SBMI=1.5*(T3BMI-T0BMI)+0.5*(T2BMI-T1BMI),
                 SMSTOT=1.5*(T3MSTOT-T0MSTOT)+0.5*(T2MSTOT-T1MSTOT),
                 SMOBILA=1.5*(T3MOBILA-T0MOBILA)+0.5*(T2MOBILA-T1MOBILA),
                 SLGMUSA=1.5*(T3LGMUSA-T0LGMUSA)+0.5*(T2LGMUSA-T1LGMUSA),
                 SGROSSA=1.5*(T3GROSSA-T0GROSSA)+0.5*(T2GROSSA-T1GROSSA))

# T3ADLA
lm0<-lm(T3ADLA~resilience,data)
lm1<-lm(T3ADLA~resilience+chronic,data)
lm2<-lm(T1ADLA~T0BMI+T1BMI,data)
lm3<-lm(T0ADLA~T0BMI+T1BMI,data)
summary(lm0)
summary(lm1)
summary(lm3)
anova(lm0,lm1)

# Multiple regression
Mul_reg1 <- glm(T3ADLA ~ chronic+GENDER+T1AGE+RACE1+ETHNICITY+YEARSEDU+T0IMRC+T0DLRC+T0SER7+T0VOCAB
                +T0MSTOT+T0CESD+T0ADLA+T0MOBILA+T0LGMUSA+T0GROSSA+T0ATOTN+T0BMI+T0HIBP+T0DIAB+T0LUNG
                +T0CANCR+T0HEART+T0STROK+T0PSYCH+T1IMRC+T1DLRC+T1SER7+T1VOCAB+T1MSTOT+T1CESD+T1ADLA
                +T1MOBILA+T1LGMUSA+T1GROSSA+T1ATOTN+T1BMI+T0PSYCH, # change1; formula: Y ~ X
                family = "gaussian",      
                data = data)      # the data
summary(Mul_reg1) #这里也想做成表

## simpler descriptives available by doing:
tidy(Mul_reg1)
# But what is the 95% confidence interval of the estimates?
# we can calculate it by doing:
confint(Mul_reg1)
### ODDS RATIOS 
### OR > 1, the larger the IV, the more likely one is alive (1)
### OR < 1, the smaller the IV, the more likely one is dead (0)
exp(                                 # exponentiate estimates
  cbind(                             # Table function
    OR = coef(log_reg1),               # the estimates
    confint(log_reg1)))        # OR 95% CIs

mortality_plot <- 
  # this draws our subjects on the graph. Divided in two lines (0 or 1) as our data is binomial
  ggplot(data, aes(T1COGTOT, T3_DefAlive)) +  geom_point(alpha = .40) +
  #change2; the logistic line drawn on the plot
  geom_smooth(method = "glm", method.args = list(family = "binomial")) +
  # graphical extras (labels)
  ggtitle("Probability of alive versus cognition") +  xlab("Cognition") +  ylab("Probability of Alive")
  #change3

mortality_plot

# predictions!
predict(log_reg1,   # specify the previously estimated regression
        data.frame(chronic=2,GENDER=2,T1AGE=2,RACE1=2,ETHNICITY=2,YEARSEDU=2,T0IMRC=2,T0DLRC=2,T0SER7=2,T0VOCAB=2,T0MSTOT=2,T0CESD=2,T0ADLA=2,T0MOBILA=2,T0LGMUSA=2,T0GROSSA=2,T0ATOTN=2,T0BMI=2,T0HIBP=2,T0DIAB=2,T0LUNG=2,T0CANCR=2,T0HEART=2,T0STROK=2,T0PSYCH=2,T1IMRD=2,T1DLRC=2,T1SER7=2,T1VOCAB=2,T1MSTOT=2,T1CESD=2,T1ADLA=2,T1MOBILA=2,T1LGMUSA=2,T1GROSSA=2,T1ATOTN=2,T1BMI=2,T1PSYCH=2),   #change4; here we specify for women only (at these values)
        type = "response")

# From un-regularized to regularized model: LASSO

# imputation
variable_keep1<-c("GENDER","T1AGE","RACE1","ETHNICITY","YEARSEDU","T0IMRC","T0DLRC","T0SER7","T0VOCAB","T0MSTOT","T0CESD","T0ADLA","T0MOBILA","T0LGMUSA","T0GROSSA","T0ATOTN","T0BMI","T0HIBP","T0DIAB","T0LUNG","T0CANCR","T0HEART","T0PSYCH","T1IMRD","T1DLRC","T1SER7","T1VOCAB","T1MSTOT","T1CESD","T1ADLA","T1MOBILA","T1LGMUSA","T1GROSSA","T1ATOTN","T1BMI","T1PSYCH","T3_DefAlive") #change1
variable_keep2<-c("GENDER","T1AGE","RACE1","ETHNICITY","YEARSEDU","T0IMRC","T0DLRC","T0SER7","T0VOCAB","T0MSTOT","T0CESD","T0ADLA","T0MOBILA","T0LGMUSA","T0GROSSA","T0ATOTN","T0BMI","T0HIBP","T0DIAB","T0LUNG","T0CANCR","T0HEART","T0PSYCH","T3_DefAlive")
variable_keep3<-c("GENDER","T1AGE","RACE1","ETHNICITY","YEARSEDU","T1IMRD","T1DLRC","T1SER7","T1VOCAB","T1MSTOT","T1CESD","T1ADLA","T1MOBILA","T1LGMUSA","T1GROSSA","T1ATOTN","T1BMI","T1PSYCH","T3_DefAlive")

data_keep<-data[variable_keep1]
cleandata<- knnImputation(data_keep,k=10,scale=T,meth="weighAvg",distData=NULL)
#knn: k nearest neighbor

cleandata$T3_DefAlive <- round(cleandata$T3_DefAlive, digits= 0)


table(cleandata$T3_DefAlive)
cleandata$T3_DefAlive <- as.factor(cleandata$T3_DefAlive)
levels(cleandata$T3_DefAlive) <- c("mortality", "alive")
table(cleandata$T3_DefAlive)
skim(cleandata) 
########################FULL MODEL PREDICTING RESILIENCE#############################
#train control setting
up_fitControl1 <- trainControl(method = "repeatedcv", repeats = 3, savePredictions = T, 
                               summaryFunction = twoClassSummary, returnResamp="all", classProbs = T, 
                               sampling = "up") #upsampling improve balance
set.seed(123)
up_lasso1_caret <- train(T3_DefAlive ~. , data=cleandata, 
                         method = "glmnet", trControl = up_fitControl1, 
                         tuneGrid=expand.grid(.alpha=1, .lambda=seq(0.001, 0.1,by = 0.001)), 
                         family="binomial", metric = "ROC", preProcess = c("center", "scale"))

#the optimal lambda that minimizes cross-validation error based on the model performance on ROC metric 
up_lasso1_caret 
up_lasso1_caret$bestTune  #optimal lambda = 0.003

#calculate mean auc and interquartile range
AliveAuc <- up_lasso1_caret$resample
AliveAuc <- AliveAuc[AliveAuc$lambda==0.003,]
mean(AliveAuc$ROC)
quantile(AliveAuc$ROC)

#calculate AUC CI
CI <- function(data, indices){
  dt<-data[indices,]
  
  mean(dt[,3])
}

set.seed(123)
myBootstrap <- boot(AliveAuc, CI, R=5000)
boot.ci(myBootstrap) #BCa CI = .67 to .69

#Examine variable importance 
#using varImp() 
plot(varImp(up_lasso1_caret), top=20)

coef(up_lasso1_caret$finalModel, up_lasso1_caret$bestTune$lambda) #check coef direction

logit<-glm(T3_DefAlive~T1BMI+T1AGE+T0HEART+T0DIAB+T0DLRC+T0LGMUSA+T0BMI+T1MOBILA+T1DLRC
           +T1MSTOT+T1GROSSA+T0PSYCH+GENDER+YEARSEDU+T1ADLA+RACE1+T0LUNG+T0MOBILA+T0GROSSA
           +T0HIBP, data=data, family="binomial")
forest_model(logit)

summary(logit)
exp(logit$coefficients)
#mcfadden's pseudo r-square
library(pscl)
pR2(logit)





