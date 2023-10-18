################################
################################
### DESN Forecasts for EESTN ###
################################
################################

#clear environment and load libraries
rm(list = ls()); gc()
library(tidyverse)
library(Matrix)
library(wql)
library(reshape2)

#parallel libraries
library(foreach)
library(parallel)
library(doParallel)

#load functions
source('deep_functions_with_train.R')
source('functions.R')

#select cores
options(cores = 10)

#load data 
load('L96SimData.RData')


###############################
### DESN as Input for a TNN ###
###############################

#Parameter specification
layers = 3
n.h = c(rep(500,layers))
nu = c(0.4,1.0,1.0)
lambda.r = 1e-1
m = 1
alpha = 0.49
reduced.units = 10

#Fixed parameters
pi.w = rep(0.1, layers)
eta.w = rep(1,layers)
pi.win = rep(0.1, layers)
eta.win = rep(1,layers)
iterations = 100
tau = 20
validLen = 0
trainLen = 980-validLen-tau+1
testLen = 20
locations = dim(rawData)[2]


#Create training and testing sets
sets = cttv(rawData, tau, trainLen, testLen = testLen)
new.train = sets$yTrain
testindex = sets$xTestIndex

#Generating input data
input.dat = gen.input.data(trainLen = trainLen,
                           m = m,
                           tau = tau,
                           yTrain = (new.train),
                           rawData = (rawData),
                           locations = locations,
                           xTestIndex = testindex,
                           testLen = testLen)
y.scale = input.dat$y.scale
y.train = input.dat$in.sample.y
designMatrix = input.dat$designMatrix
designMatrixOutSample = input.dat$designMatrixOutSample
addScaleMat = input.dat$addScaleMat

training_dat = y.scale*y.train + addScaleMat[1,1]

#start = proc.time()
testing = deep.esn.train(y.train = y.train,
                         x.insamp = designMatrix,
                         x.outsamp = designMatrixOutSample,
                         y.test = training_dat,
                         n.h = n.h,
                         nu = nu,
                         pi.w = pi.w, 
                         pi.win = pi.win,
                         eta.w = eta.w,
                         eta.win = eta.win,
                         lambda.r = lambda.r,
                         alpha = alpha,
                         m = m,
                         iter = iterations,
                         future = testLen,
                         layers = layers,
                         reduced.units = reduced.units,
                         startvalues = NULL,
                         activation = 'tanh',
                         distribution = 'Normal',
                         scale.factor = y.scale,
                         scale.matrix = addScaleMat,
                         fork = FALSE)
#proc.time()-start


esn_preds = testing$forecastmean



#preprocess data in format needed for python files
colnames(esn_preds) = 1:locations
tnn_dat = melt(esn_preds)
colnames(tnn_dat) = c('index', 'article', 'views')
tnn_dat$article = as.character(tnn_dat$article)

#generate input 
wanted_lags = c(0)
new_columns = matrix(NaN, nrow = dim(tnn_dat)[1], length(wanted_lags))
index = 1
for(shift in wanted_lags)
{
  new_columns[,index] = data.frame(tnn_dat) %>%
    group_by(article) %>%
    mutate('views_lag_' = lag(views, shift)) %>%
    dplyr::select(article, views_lag_) %>%
    ungroup() %>%
    dplyr::select(views_lag_) %>%
    as.matrix()
  
  index = index + 1
}
colnames(new_columns) = paste0('views_lag_', wanted_lags)
head(new_columns)
new_columns[is.na(new_columns)] = 0
head(new_columns)


#Combine data with lags
esn_for_tnn = cbind(tnn_dat, new_columns)
esn_for_tnn$views = melt(training_dat)[,3]
esn_for_tnn$intercept = rep(1, dim(esn_for_tnn)[1])
head(esn_for_tnn)

write.csv(esn_for_tnn, file = paste0('DESNInputTNN.csv'), row.names = FALSE)







###################################
### DESN Testing Points For TNN ###
###################################


#Parameter specification
layers = 3
n.h = c(rep(500,layers))
nu = c(0.4,1.0,1.0)
lambda.r = 1e-1
m = 1
alpha = 0.49
reduced.units = 10

#Fixed parameters
pi.w = rep(0.1, layers)
eta.w = rep(1,layers)
pi.win = rep(0.1, layers)
eta.win = rep(1,layers)
iterations = 100
tau = 20
validLen = 0
trainLen = 980-tau-validLen+1
testLen = 20
locations = dim(rawData)[2]

#Create training and testing sets
sets = cttv(rawData, tau, trainLen, testLen = testLen)
new.train = sets$yTrain
testindex = sets$xTestIndex

#Generating input data
input.dat = gen.input.data(trainLen = trainLen,
                           m = m,
                           tau = tau,
                           yTrain = (new.train),
                           rawData = (rawData),
                           locations = locations,
                           xTestIndex = testindex,
                           testLen = testLen)
y.scale = input.dat$y.scale
y.train = input.dat$in.sample.y
designMatrix = input.dat$designMatrix
designMatrixOutSample = input.dat$designMatrixOutSample
addScaleMat = input.dat$addScaleMat


#start = proc.time()
testing = deep.esn(y.train = y.train,
                   x.insamp = designMatrix,
                   x.outsamp = designMatrixOutSample,
                   y.test = sets$yTest,
                   n.h = n.h,
                   nu = nu,
                   pi.w = pi.w, 
                   pi.win = pi.win,
                   eta.w = eta.w,
                   eta.win = eta.win,
                   lambda.r = lambda.r,
                   alpha = alpha,
                   m = m,
                   iter = iterations,
                   future = testLen,
                   layers = layers,
                   reduced.units = reduced.units,
                   startvalues = NULL,
                   activation = 'tanh',
                   distribution = 'Normal',
                   scale.factor = y.scale,
                   scale.matrix = addScaleMat,
                   fork = FALSE,
                   parallel = TRUE,
                   logNorm = FALSE)
#proc.time()-start


esn_preds = testing$forecastmean



#preprocess data in format needed for python files
colnames(esn_preds) = 1:locations
tnn_dat = melt(esn_preds)
colnames(tnn_dat) = c('index', 'article', 'views')
tnn_dat$article = as.character(tnn_dat$article)

#generate input for testing
wanted_lags = c(0)
new_columns = matrix(NaN, nrow = dim(tnn_dat)[1], length(wanted_lags))
index = 1
for(shift in wanted_lags)
{
  new_columns[,index] = tnn_dat %>%
    group_by(article) %>%
    mutate('views_lag_' = lag(views, shift)) %>%
    dplyr::select(article, views_lag_) %>%
    ungroup() %>%
    dplyr::select(views_lag_) %>%
    as.matrix()
  
  index = index + 1
}
colnames(new_columns) = paste0('views_lag_', wanted_lags)
head(new_columns)
new_columns[is.na(new_columns)] = 0
head(new_columns)

#Combine data with lags
esn_for_tnn_test = cbind(tnn_dat, new_columns)
esn_for_tnn_test$views = melt(sets$yTest)[,3]
esn_for_tnn_test$intercept = rep(1, dim(esn_for_tnn_test)[1])
head(esn_for_tnn_test)

training_dat = read.csv('DESNInputTNN.csv')
#head(temp_dat)

all_esn_for_tnn = rbind(training_dat, esn_for_tnn_test)
write.csv(all_esn_for_tnn, file = 'DESNInputTNNTest.csv', row.names = FALSE)




