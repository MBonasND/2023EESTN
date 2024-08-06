###########################################
###########################################
### Calibration of Long-Range Forecasts ###
###########################################
###########################################


#clear environment and load libraries
rm(list = ls())
library(tidyverse)
library(Matrix)
library(abind)
library(scam)

#parallel libraries
library(doParallel)
library(parallel)
library(foreach)




################################
### Calibration of Forecasts ###
################################

#load EET forecasts and Data
load('all_EET_AP_Windows.RData')
load('APFour.RData')
rawData = log(t(newpoll) + 0.01)

#specify calibration info
locations = dim(full_EET_forecasts)[1]
n.w = dim(full_EET_forecasts)[3]
tau = dim(full_EET_forecasts)[2]
start.range = 392 - tau
horizon = tau
true.range = (start.range + tau + 1):(start.range + n.w*horizon + tau)


#Get optimal sd for each location
optim.sd.mat = matrix(NaN, nrow = locations, ncol = horizon)
for(l in 1:locations)
{
  location = l
  dat = full_EET_forecasts[location,,]
  true.y = rawData[true.range, location]
  
  #Generate data windows for j-step ahead forecasts
  true.window = list()
  mean.window = list()
  for(i in 1:horizon)
  {
    index = seq(i, (n.w-1)*horizon+i, horizon)
    true.window[[i]] = true.y[index]
    mean.window[[i]] = dat[i, ]
  }
  
  
  ### Optimal SD w/ Monotonic Spline ###
  #Generate optimal SD
  optim.sd = rep(0, horizon)
  for(i in 1:horizon)
  {
    optim.sd[i] = sd(true.window[[i]] - mean.window[[i]])
  }
  
  #Monotonic Spline
  testy = optim.sd
  testx = 1:horizon
  fit = scam(testy~s(testx, k=-1, bs="mpi"), 
             family=gaussian(link="identity"))
  
  #set optimum values from monotonic spline
  optim.sd.mat[l,] = fit$fitted.values
  
  #print progress
  print(l)
}

save(optim.sd.mat, file = 'EET_Optim_SD_AP.RData')