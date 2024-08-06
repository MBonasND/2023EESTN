############################################
############################################
### PurpleAir Exposure --- Block Kriging ###
############################################
############################################

#clear environment and load librariers
rm(list = ls()); gc()
library(tidyverse)
library(geoR)
library(sf)
library(sp)
library(gstat)


#specify colors for plotting
cols <- c("snow1", "yellow", "orange", "orangered", "red", "red4")


#load data
load('APFour.RData')
rawData = log(t(newpoll) + 0.01)


##############################################
### Calculating Exposure --- Block Kriging ###
##############################################

#load population data and coordinates
load('all_EET_AP_Windows.RData')
load('SanFranPop.RData')
load('AirPollutionLocations.RData')
pollution.locs = t(apply(pollution.locs, 1, rev))



#Creating Grid for San Francisco 
load('SanFranBorder.RData')
gr = pred_grid(sfborders, by = 0.002)
pollut.grid = locations.inside(loc = gr, borders = sfborders)
colnames(pollut.grid) = c('Lon', 'Lat')
full_grid = as.data.frame(pollut.grid)
coordinates(full_grid) = ~ Lon + Lat


#Forecast data using Block kriging
individ_convex = full_EET_forecasts[,,16] #extract the future forecasts
areal_forc = list()
for(t in 1:30)
{
  #preliminaries
  time = t
  stnn.means = exp(individ_convex)[, time]
  forcs = log(stnn.means)
  
  #Preoare data for Block kriging
  full_data = cbind.data.frame(pollution.locs, forcs)
  colnames(full_data) = c('lon', 'lat', 'ap')
  full_data_sf = st_as_sf(full_data, coords = c("lon", "lat"), crs = 4326)  # Assuming original CRS is WGS84
  projected_crs = 32633  # Example projected CRS (UTM Zone 33N)
  full_data_proj = st_transform(full_data_sf, crs = projected_crs)
  sf_pop_proj = st_transform(sf_pop, crs = projected_crs)
  
  # Create a variogram model
  vario.mod = variogram(ap ~ 1, data = as(full_data_proj, "Spatial"))
  vario.fit = fit.variogram(vario.mod, model = vgm(model = 'Mat', range = 5000))  # Adjust range parameter as needed
  
  # Loop through each polygon and perform block kriging
  block_kriging_results <- list()
  for (p in (1:(length(sf_pop_proj$geometry)-1))[-c(61, 82)]) {
    #get block
    block = sf_pop_proj[p, ]
    
    # Perform block kriging
    kriging_result <- krige(ap ~ 1, as(full_data_proj, "Spatial"), newdata = as(block, "Spatial"), model = vario.fit, block = TRUE)
    
    # Store the result
    border = sf_pop$geometry[[p]][[1]][[1]]
    phold = cbind(border, kriging_result$var1.pred)
    colnames(phold) = c('Lon', 'Lat', 'Forecast')
    block_kriging_results[[p]] <-  phold
  }
  block_kriging_results[[82]] = NULL #this district is outside the mainland
  block_kriging_results[[61]] = NULL #his district is outside the mainland
  
  # Combine results into a single data frame
  areal_forc[[t]] = block_kriging_results
  
  print(t)
}




#Convert data into data frames
forc.district = list()
for(t in 1:30)
{
  district.actPM25 = matrix(NaN, ncol = 4)
  for(i in 1:length(areal_forc[[t]]))
  {
    arb = areal_forc[[t]][[i]]
    place = cbind(arb, 'group' = i)
    district.actPM25 = rbind(district.actPM25, place)
  }
  forc.district[[t]] = district.actPM25[-1,]
}
areal.forc = lapply(forc.district, as.data.frame)
# save(areal.forc, file = 'ESN-TNNArealForc_BlockKriging.RData')



#load areal forcs and plot
load('ESN-TNNArealForc_BlockKriging.RData')
ggplot(data = areal.forc[[30]], aes(x = Lon, y = Lat, group = group, fill = exp(Forecast))) +
  geom_polygon() +
  geom_path(color = 'white',
            linewidth = 0.25) +
  labs(x = '', y = '', title = 'Interpolated Forecasts: 2020/05/24 20:00:00', fill = expression(PM[2.5]~('\u03bcg'~m^-3))) +
  scale_fill_gradientn(colors = cols, limit = c(0,10.5)) +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5, size = 20),
        axis.text.x = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks = element_blank(),
        legend.position = 'bottom',
        legend.text = element_text(size = 14),
        legend.title = element_text(size = 18, vjust = 1.25, hjust = 1),
        legend.direction = 'horizontal',
        legend.key.width = unit(1.5,"cm"),
        panel.grid = element_blank(),
        panel.border = element_blank()) +
  scale_y_continuous(expand = c(0,0))



