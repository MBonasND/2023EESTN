################################################
################################################
### Stochastic Transformer Network Functions ###
################################################
################################################

#load libraries
library(tidyverse)
library(Matrix)
library(abind)

library(parallel)
library(foreach)
library(doParallel)

set.seed(NULL)


#################################
### Stochastic Transformer NN ###
#################################

deep.stnn = function(y.train,
                     x.insamp,
                     x.outsamp,
                     y.test = NULL,
                     d_ff,
                     d_m,
                     num_heads,
                     pi.wq, 
                     pi.wk,
                     pi.wv,
                     pi.v1, 
                     pi.b1,
                     pi.v2,
                     pi.v3,
                     lambda.r,
                     iter,
                     layers,
                     scale.factor,
                     scale.matrix,
                     fork = FALSE)
{
  
  ###########################
  #Begin Ensemble Iterations
  ###########################
  
  #Specify Parallel clusters
  if(fork)
  {
    cl = parallel::makeForkCluster(getOption('cores')) 
  } else if(!fork)
  {
    cl = parallel::makeCluster(getOption('cores'))
  }
  
  #Activate clusters
  doParallel::registerDoParallel(cl)
  
  #Begin parallel iterations
  ensemb.mat = foreach::foreach(k = 1:iter,
                                .combine = abind,
                                .inorder = FALSE) %dopar%
    {
      set.seed(k)
      
      ##############################################
      ### Generate Multi-Head Attention Matrices ###
      ##############################################
      
      #Split into multiple heads
      d_model = dim(x.insamp)[2]
      
      W_q_head = list()
      W_k_head = list()
      W_v_head = list()
      
      layer_samples_W = list()
      row_dim_W = list()
      d_head = list()
      for(ell in 1:layers)
      {
        if(ell == 1)
        {
          d_head[[ell]] = d_model %/% num_heads
          layer_samples_W[[ell]] = d_head[[ell]] * d_model
          row_dim_W[[ell]] = d_model
        } else {
          d_head[[ell]] = d_ff %/% num_heads
          layer_samples_W[[ell]] = d_head[[ell]] * d_ff
          row_dim_W[[ell]] = d_ff
        }
      }
      
      
      #Different weights for each layer
      for(ell in 1:layers)
      {
        W_q = list()
        W_k = list()
        W_v = list()
        
        for(head in 1:num_heads) #num_heads needs to be a factor of d_model and d_ff
        {
          #Set Sparsity
          gam.wq = purrr::rbernoulli(layer_samples_W[[ell]], p = pi.wq[ell])
          gam.wk = purrr::rbernoulli(layer_samples_W[[ell]], p = pi.wk[ell])
          gam.wv = purrr::rbernoulli(layer_samples_W[[ell]], p = pi.wv[ell])
          
          #Generate W_q
          norm.wq = rnorm(layer_samples_W[[ell]], 0, 1)
          W_q[[head]] = Matrix::Matrix((gam.wq == 1)*norm.wq + (gam.wq == 0)*0,
                                      ncol = d_head[[ell]],
                                      nrow = row_dim_W[[ell]],
                                      sparse = TRUE)
          
          #Generate W_k
          norm.wk = rnorm(layer_samples_W[[ell]], 0, 1)
          W_k[[head]] = Matrix::Matrix((gam.wk == 1)*norm.wk + (gam.wk == 0)*0,
                                      ncol = d_head[[ell]],
                                      nrow = row_dim_W[[ell]],
                                      sparse = TRUE)
          
          #Generate W_v
          norm.wv = rnorm(layer_samples_W[[ell]], 0, 1)
          W_v[[head]] = Matrix::Matrix((gam.wv == 1)*norm.wq + (gam.wv == 0)*0,
                                      ncol = d_head[[ell]],
                                      nrow = row_dim_W[[ell]],
                                      sparse = TRUE)
        }
        W_q_head[[ell]] = W_q
        W_k_head[[ell]] = W_k
        W_v_head[[ell]] = W_v

        
      } #end multi-head weight matrix sampling
      
      
      #############################################
      ### Generate Feed-Forward Weight Matrices ###
      #############################################
      V1 = list()
      b1 = list()
      V2 = list()
      V3 = list()
      
      #Get number of samples for weight matrices
      layer_samples_V1 = list()
      layer_samples_V2_V3 = list()
      layer_samples_b1 = list()
      layer_dim_V1 = list()
      layer_dim_V2_V3 = list()
      for(ell in 1:layers)
      {
        if(ell == 1)
        {
          layer_samples_V1[[ell]] = d_model * d_ff
          layer_samples_V2_V3[[ell]] = d_m * d_ff
          layer_samples_b1[[ell]] = d_ff
          
          layer_dim_V1[[ell]] = d_model
          layer_dim_V2_V3[[ell]] = d_m
        } else {
          layer_samples_V1[[ell]] = d_ff * d_ff
          layer_samples_V2_V3[[ell]] = d_m * d_ff
          layer_samples_b1[[ell]] = d_ff
          
          layer_dim_V1[[ell]] = d_ff
          layer_dim_V2_V3[[ell]] = d_m
        }
      }
      
      for(ell in 1:layers)
      {
        #Set Sparsity
        gam.v1 = purrr::rbernoulli(layer_samples_V1[[ell]], p = pi.v1[ell])
        gam.b1 = purrr::rbernoulli(layer_samples_b1[[ell]], p = pi.b1[ell])
        gam.v2 = purrr::rbernoulli(layer_samples_V2_V3[[ell]], p = pi.v2[ell])
        gam.v3 = purrr::rbernoulli(layer_samples_V2_V3[[ell]], p = pi.v3[ell])
        
        #Generate V1
        norm.v1 = rnorm(layer_samples_V1[[ell]], 0, 1)
        V1[[ell]] = Matrix::Matrix((gam.v1 == 1)*norm.v1 + (gam.v1 == 0)*0,
                                   nrow = layer_dim_V1[[ell]], 
                                   ncol = d_ff,
                                   sparse = TRUE)
        
        #Generate b1
        norm.b1 = rnorm(layer_samples_b1[[ell]], 0, 1)
        b1[[ell]] = Matrix::Matrix((gam.b1 == 1)*norm.b1 + (gam.b1 == 0)*0,
                                   ncol = d_ff,
                                   nrow = 1,
                                   sparse = TRUE)
        
        #Generate V2
        norm.v2 = rnorm(layer_samples_V2_V3[[ell]], 0, 1)
        V2[[ell]] = Matrix::Matrix((gam.v2 == 1)*norm.v2 + (gam.v2 == 0)*0,
                                   nrow = d_ff, 
                                   ncol = layer_dim_V2_V3[[ell]],
                                   sparse = TRUE)
        
        #Generate v3
        norm.v3 = rnorm(layer_samples_V2_V3[[ell]], 0, 1)
        V3[[ell]] = Matrix::Matrix((gam.v3 == 1)*norm.v3 + (gam.v3 == 0)*0,
                                   ncol = d_ff, 
                                   nrow = layer_dim_V2_V3[[ell]],
                                   sparse = TRUE)
        
        
      } #end feed-forward weight matrix sampling
      
      
      ###################################
      ### Loop Through Layers of STNN ###
      ###################################
      
      #Initialize input, output, and ridge items
      input.data = list()
      input.data[[1]] = x.insamp
      output.data = list()
      output.data[[1]] = x.outsamp


      for(ell in 1:layers)
      {

        #Multi-Head Attention Layer
        all_heads <- vector("list", num_heads)
        all_heads_output = vector("list", num_heads)
        for(head in 1:num_heads)
        {
          #Input Data
          Q = input.data[[ell]] %*% W_q_head[[ell]][[head]]
          K = input.data[[ell]] %*% W_k_head[[ell]][[head]]
          V = input.data[[ell]] %*% W_v_head[[ell]][[head]]

          scores = Matrix::tcrossprod(Q, K)/sqrt(d_head[[ell]])
          attn_scores = exp(scores)
          attn_probs = attn_scores / Matrix::rowSums(attn_scores)

          all_heads[[head]] = attn_probs %*% V

          #Output Data
          Q_output = output.data[[ell]] %*% W_q_head[[ell]][[head]]
          K_output = output.data[[ell]] %*% W_k_head[[ell]][[head]]
          V_output = output.data[[ell]] %*% W_v_head[[ell]][[head]]

          scores_output = Matrix::tcrossprod(Q_output, K_output)/sqrt(d_head[[ell]])
          attn_scores_output = exp(scores_output)
          attn_probs_output = attn_scores_output / Matrix::rowSums(attn_scores_output)

          all_heads_output[[head]] = attn_probs_output %*% V_output


        } #end multi-head loop
        multi_head_input = do.call(cbind, all_heads)
        multi_head_output = do.call(cbind, all_heads_output)

        #center and scale multi-head results - Input Data
        attn_results = input.data[[ell]] + multi_head_input
        ar_mean = mean(attn_results)
        ar_sd = sd(attn_results)
        normal_attn_results = (attn_results - ar_mean) / ar_sd

        #center and scale multi-head results - Output Data
        attn_results_output = output.data[[ell]] + multi_head_output
        ar_mean_out = mean(attn_results_output)
        ar_sd_out = sd(attn_results_output)
        normal_attn_results_output = (attn_results_output - ar_mean_out) / ar_sd_out


        #Feed_Forward Layer - Input Data
        u = (normal_attn_results %*% V1[[ell]]) + matrix(b1[[ell]],
                                                         nrow = dim(normal_attn_results)[1],
                                                         ncol = d_ff,
                                                         byrow = TRUE)

        z_prime = pmax(u %*% V2[[ell]], 0) %*% V3[[ell]]
        z_results = u + z_prime
        zr_mean = mean(z_results)
        zr_sd = sd(z_results)
        z_input = (z_results - zr_mean)/zr_sd

        #specify next layer's input data
        input.data[[ell+1]] = z_input



        #Feed_Forward Layer - Output Data
        u_output = (normal_attn_results_output %*% V1[[ell]]) + matrix(b1[[ell]],
                                                                       nrow = dim(normal_attn_results_output)[1],
                                                                       ncol = d_ff,
                                                                       byrow = TRUE)

        z_prime_out = pmax(u_output %*% V2[[ell]], 0) %*% V3[[ell]]

        z_results_out = u_output + z_prime_out
        zr_mean_out = mean(z_results_out)
        zr_sd_out = sd(z_results_out)
        z_output = (z_results_out - zr_mean_out)/zr_sd_out

        #specify next layer's output data
        output.data[[ell+1]] = z_output



      } #end layers loop


      ###################################
      ### Estimate Coefficient Matrix ###
      ###################################

      Ident.Mat = diag(d_ff+1)
      ridgeMat = lambda.r * Ident.Mat
      ones_matrix = matrix(1, nrow = dim(input.data[[layers+1]])[1])
      final.design = t(cbind(ones_matrix, as.matrix(input.data[[layers+1]])))

      #Ridge regression estimates
      Betas = t(y.train) %*% t(final.design) %*% solve(Matrix::tcrossprod(final.design, final.design) + ridgeMat)


      ###########################
      ### Calculate Forecasts ###
      ###########################

      ones_matrix = matrix(1, nrow = dim(output.data[[layers+1]])[1])
      final.design.out = t(cbind(ones_matrix, as.matrix(output.data[[layers+1]])))

      as.matrix(scale.factor * (Betas %*% final.design.out) + scale.matrix)
    } #end parallel loop
  
  
  ########################
  ### Finalize Results ###
  ########################
  
  #Close clusters
  parallel::stopCluster(cl)
  
  #Calulcate Forecast Mean
  locations = dim(y.train)[2]
  future = dim(x.outsamp)[1]
  if(locations > 1 & future == 1)
  {
    forc.mean = apply(ensemb.mat, 1, mean)
  } else if(locations == 1 & future > 1){
    forc.mean = (sapply(1:future, function(x) mean(ensemb.mat[,seq(x, ncol(ensemb.mat), future)])))
  } else if(locations > 1 & future > 1) {
    forc.mean = t(sapply(1:future, function(x) rowMeans(ensemb.mat[,seq(x, ncol(ensemb.mat), future)])))
  } else if(locations == 1 & future == 1) {
    forc.mean = mean(as.numeric(ensemb.mat))
  } else {
    forc.mean = NULL
  }

  #Calculate MSE
  if(!is.null(y.test))
  {
    MSE = mean((y.test-forc.mean)^2)
  } else {
    MSE = NULL
  }


  #Compile output
  stnn.output = list('predictions' = ensemb.mat,
                     'forecastmean' = forc.mean,
                     'MSE' = MSE)

  return(stnn.output)
  
  
  
} #end of function



###########################
### Generate Input Data ###
###########################


gen.input.data = function(trainLen,
                          m,
                          tau,
                          yTrain,
                          rawData,
                          locations,
                          xTestIndex,
                          testLen)
{
  in.sample.len = trainLen - (m * tau)
  
  in.sample.x.raw = array(NA, dim = c(in.sample.len, m+1, locations))
  
  for(i in 1:in.sample.len)
  {
    in.sample.x.raw[i,,] = rawData[seq(i, (m*tau + i), by=tau), ]
  }
  
  #Scale in-sample x and y
  in.sample.y.raw = yTrain[(m*tau + 1):trainLen,]
  y.mean = mean(in.sample.y.raw)
  y.scale = sd(in.sample.y.raw)
  
  in.sample.y = (in.sample.y.raw - y.mean)/y.scale
  
  
  mean.train.x = mean(rawData[1:trainLen,])
  sd.train.x = sd(rawData[1:trainLen,])
  
  
  in.sample.x=(in.sample.x.raw - mean.train.x)/sd.train.x
  
  
  designMatrix = matrix(1,in.sample.len, (m + 1)*locations + 1)
  for(i in 1:in.sample.len){
    designMatrix[i,2:((m + 1)*locations + 1)] = as.vector(in.sample.x[i,,])
  }
  
  
  #Out-Sample
  out.sample.x.raw = array(NA, dim = c(testLen, m + 1, locations))
  for(i in 1:testLen)
  {
    out.sample.x.raw[i,,] = rawData[seq(xTestIndex[i]-(m*tau), xTestIndex[i], by=tau),]
  }
  
  
  #Scale out-sample x and y
  out.sample.x = (out.sample.x.raw - mean.train.x)/sd.train.x
  
  designMatrixOutSample = matrix(1, testLen, (m + 1)*locations + 1)
  for(i in 1:testLen)
  {
    designMatrixOutSample[i,2:((m + 1)*locations + 1)] = as.vector(out.sample.x[i,,])
  }
  
  
  
  #Additive scale matric
  addScaleMat = matrix(y.mean, locations, testLen)
  
  input.data.output = list('y.mean' = y.mean,
                           'y.scale' = y.scale,
                           'in.sample.y' = in.sample.y,
                           'in.sample.x' = in.sample.x,
                           'out.sample.x' = out.sample.x,
                           'in.sample.len' = in.sample.len,
                           'designMatrix' = designMatrix,
                           'designMatrixOutSample' = designMatrixOutSample,
                           'testLen' = testLen,
                           'addScaleMat' = addScaleMat)
  return(input.data.output)
}


#######################################################
### Generate Training, Testing, and Validation Sets ###
#######################################################


cttv = function(rawData, tau, trainLen, testLen, validLen = NULL, valid.flag = FALSE)
{
  #Create training and testing sets
  totlength = trainLen + testLen + tau
  yTrain = rawData[(tau+1):(trainLen+tau),]
  yTest = rawData[(trainLen+tau+1):totlength,]
  xTestIndex = seq((trainLen+1), (totlength-tau), 1)
  
  #Create valid sets
  if(valid.flag)
  {
    xValTestIndex=(trainLen+1-validLen):(trainLen)
    yValTestIndex=(trainLen+tau+1-validLen):(trainLen+tau)
    yValid = rawData[yValTestIndex,]
  } else {
    yValid = NULL
    xValTestIndex = NULL
  }
  
  #Return list
  output = list('yTrain' = yTrain,
                'yTest' = yTest,
                'yValid' = yValid,
                'xTestIndex' = xTestIndex,
                'xValTestIndex' = xValTestIndex)
  return(output)
}



