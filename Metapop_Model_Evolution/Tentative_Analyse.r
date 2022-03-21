########### Analysis of accuracy of metapop algortihm VS direct method algorithm ###########
library(readr)
library(dplyr)

########################################################################################
##################### GET FILES AND DM DATAFRAMES ######################################
########################################################################################

#Get Files
dir <- choose.dir()
dirDMOut <- paste(dir, "DM out reduced", sep = "\\")
dirLeapOut <- paste(dir, "Leap out", sep = "\\")

ls_files_DM <- list.files(dirDMOut)
ls_files_Leap <- list.files(dirLeapOut)

#Prepare pdf report
namereport <- 'rapport_eval2.pdf'
pdf(paste(dir, namereport, sep = '\\'), onefile = T)
#dev.off()

#Read files
cwd <- setwd(dirDMOut)
listfiles <- list.files(cwd)
listfiles
list_tibbles <- lapply(listfiles, read_csv) # Return tibble ? What is that **** ?
list_df <- lapply(list_tibbles, as.data.frame) # Convert to good old dataframe

#Supress 2 first columns (because useless) and round 't' column to 3 digits (to perform comparison of same t in DM and TauLeap)
for (i in 1:length(list_df)){
  
  list_df[[i]]<- subset(list_df[[i]][c(-1,-2)]) # supress two first columns
  list_df[[i]]$t <- round(list_df[[i]]$t, 3) #round t values
  #Avoid duplicates
  doubles <- which(duplicated(list_df[[i]]$t))
  list_df[[i]] <- list_df[[i]][-doubles,]
}
#Get only commons rows between all dataframes (ca commence a devenir un poil fasitidieux mébon)
#Get all t values
t_values <- c(0)
for (i in 1: length(list_df)){
  times <- list_df[[i]]$t
  t_values <- c(t_values, times)
}
t_values <- t_values[-1] # Suppress the value given for initialization
counts <- as.data.frame(table(t_values)) # Count occurences into a table

#Get values that appear nbsim times (common to every df)
nbsim <- length(list_df)
tcommon<- counts$t_values[counts$Freq==nbsim] #Return the value of t that are common to every df

#loop to keep only those t in the whole set of dfs
for (i in 1:length(list_df)){
  indexrows <- which(list_df[[i]]$t %in% tcommon)
  list_df[[i]] <- subset(list_df[[i]][indexrows,])
}

#AND NOW WE CAN WORK, 'c'est pas dommage'




################################################################################
##################### BUILD MEAN SIMULATION FOR DM #############################
################################################################################


MeanSimValue <- function(lsdf, var){ # return mean value for each time of a given variable

  nbsims <- length(lsdf)
  values <- vector('list', nbsims)

  for (i in 1: nbsims){
    idcol <- which(colnames(lsdf[[i]])== var)
    values[[i]] <- lsdf[[i]][,idcol]
  }
  as.data.frame(values)
  meanvalues <- rowSums(as.data.frame(values))/nbsims
  meanvalues
  
} 


BuildMeanSim <- function(lsdf){ # Build the mean dataframe from the whole set of simulations
  namecols<- names(lsdf[[1]])
  MeanSim <- data.frame(matrix(ncol = length(namecols), nrow =length(lsdf[[1]]$t)))
  colnames(MeanSim) <- namecols
  
  for (i in 1:length(namecols)){
    
    meanvar <- MeanSimValue(lsdf,namecols[i])
    MeanSim[,i] <- meanvar
  }
}

#In a function -> Doesnt work, oustide -> perfectly working.... voodoo
namecols<- names(list_df[[1]])
MeanSim <- data.frame(matrix(ncol = length(namecols), nrow =length(list_df[[1]]$t)))
colnames(MeanSim) <- namecols

for (i in 1:length(namecols)){
  
  meanvar <- MeanSimValue(list_df,namecols[i])
  MeanSim[,i] <- meanvar
}

################################################################################
##################### GET FILES AND TAU LEAP DATAFRAMES ########################
################################################################################

#Read files
cwd <- setwd(dirLeapOut)
listfiles <- list.files(cwd)
list_tibbles <- lapply(listfiles, read_csv) # Return tibble ? What is that **** ?
list_df_tau <- lapply(list_tibbles, as.data.frame) # Convert to good old dataframe

#Supress 2 first columns (because useless) and round 't' column to 3 digits (to perform comparison of same t in DM and TauLeap)
for (i in 1:length(list_df_tau)){
  
  list_df_tau[[i]]<- subset(list_df_tau[[i]][c(-1)]) # supress first column
  list_df_tau[[i]]$t <- round(list_df_tau[[i]]$t, 2) #round t values
  #Avoid duplicates
  doubles <- which(duplicated(list_df_tau[[i]]$t))
  list_df_tau[[i]] <- list_df_tau[[i]][-doubles,]
}
#Get only commons rows between all dataframes (ca commence a devenir un poil fasitidieux mébon)
#Get all t values
t_values <- c(0)
for (i in 1: length(list_df_tau)){
  times <- list_df_tau[[i]]$t
  t_values <- c(t_values, times)
}
t_values <- t_values[-1] # Suppress the value given for initialization
counts <- as.data.frame(table(t_values)) # Count occurences into a table
counts
#Get values that appear nbsim times (common to every df)
nbsim <- length(list_df_tau)
tcommon<- counts$t_values[counts$Freq==nbsim] #Return the value of t that are common to every df

#loop to keep only those t in the whole set of dfs
for (i in 1:length(list_df_tau)){
  indexrows <- which(list_df_tau[[i]]$t %in% tcommon)
  list_df_tau[[i]] <- subset(list_df_tau[[i]][indexrows,])
}

################################################################################
##################### BUILD MEAN SIMULATION FOR TAU LEAP #######################
################################################################################
#In a function -> Doesnt work, oustide -> perfectly working.... voodoo
namecols<- names(list_df_tau[[1]])
MeanSimTau <- data.frame(matrix(ncol = length(namecols), nrow =length(list_df_tau[[1]]$t)))
colnames(MeanSimTau) <- namecols

for (i in 1:length(namecols)){
  
  meanvar <- MeanSimValue(list_df_tau,namecols[i])
  MeanSimTau[,i] <- meanvar
}

################################################################################
################## EVALUATION OF TAU ALGORITHM #################################
################################################################################

RMSE <- function(obs, sim){
  sqrt(mean((obs-sim)**2, na.rm=T))
}
rRMSE <- function(obs, sim){
  meanO <- mean(obs)
  100*RMSE(obs, sim)/meanO
}
MAE <- function(obs, sim) {
  mean(abs(obs-sim), na.rm=T)
}
RMSEsystematic <- function(obs, sim){
  nb.obs = length(obs)
  sum_error <- 0
  model <- lm(sim~obs)
  Preg <- fitted.values(model)
  length(obs)
  length(Preg)
  
  for (k in 1:nb.obs){
    error <- (Preg[k]-obs[k])^2
    sum_error <- sum_error+error
  }
  sum_error <- unname(sum_error)
  return(((1/nb.obs)*sum_error)^0.5)
  }
RMSEuncertainity <- function(obs, sim){
  nb.obs = length(obs)
  sum_error <- 0
  model <- lm(sim~obs)
  Preg <- fitted.values(model)
  for (k in 1:length(sim)){
    error <- (sim[k]-Preg[k])^2
    sum_error <- sum_error+error
  }
  sum_error <- unname(sum_error)
  return(((1/nb.obs)*sum_error)^0.5)
}
################################################################################
############## DYNAMICS OF ONE RANDOM SITE #####################################
################################################################################

DM_raw <- list_df[[1]] #Get the first raw Df
Tau_raw <- list_df_tau[[1]] #Idem

#Figure (illustrative)
plot(DM_raw$t, DM_raw$S1,type = 'l', col = 'blue',main = "Dynamique d'un site DM", xlab = 't', ylab = 'Densité')
lines(DM_raw$t, DM_raw$I1, col = 'red')
legend("topright",legend = c("S","I"), col = c("blue", "red"), lty = 1 )

plot(Tau_raw$t, Tau_raw$S2,type = 'l', col = 'blue',main = "Dynamique d'un site Tau", xlab = 't', ylab = 'Densité')
lines(Tau_raw$t, Tau_raw$I2, col = 'red')
legend("topright",legend = c("S","I"), col = c("blue", "red"), lty = 1 )


################################################################################
################## MEAN SITE of MEAN SIMULATION#################################
################################################################################

###### Mean Sim DM
taille_col = (length(MeanSim)-1) # Number of populations columns 
nbsites <- taille_col/2

generate_even_indexes = seq(2, taille_col, 2) # Even indexes are S populations
#generate_even_indexes
generate_odd_indexes = seq(3, taille_col+1, 2) # Odd indexes are I populations
#generate_odd_indexes

Spops <- subset(MeanSim[generate_even_indexes])
Ipops <- subset(MeanSim[generate_odd_indexes])

# We sum by lines to get global densities
Smean <- rowSums(Spops)/nbsites
Imean <- rowSums(Ipops)/nbsites

Mean_Data_DM <- data.frame(t = MeanSim$t, S = Smean, I=Imean)

plot(Mean_Data_DM$t, Mean_Data_DM$S, type = 'l', col='blue', main = "Dynamique du site moyen : DirectMethod", xlab = "t", ylab = "Densité")
lines(Mean_Data_DM$t, Mean_Data_DM$I, col = 'red')
legend("topright",legend = c("S","I"), col = c("blue", "red"), lty = 1 )

###### Mean Sim TAU
taille_col = (length(MeanSimTau)-1) # Number of populations columns 

generate_even_indexes = seq(2, taille_col, 2) # Even indexes are S populations
generate_odd_indexes = seq(3, taille_col+1, 2) # Odd indexes are I populations

Spops <- subset(MeanSimTau[generate_even_indexes])
Ipops <- subset(MeanSimTau[generate_odd_indexes])

# We sum by lines to get global densities
Smean <- rowSums(Spops)/nbsites
Imean <- rowSums(Ipops)/nbsites

Mean_Data_LEAP <- data.frame(t = MeanSimTau$t, S = Smean, I=Imean)

plot(Mean_Data_LEAP$t, Mean_Data_LEAP$S, type = 'l', col='blue', main = "Dynamique du site moyen : TauLEap", xlab = "t", ylab = "Densité")
lines(Mean_Data_LEAP$t, Mean_Data_LEAP$I, col = 'red')
legend("topright",legend = c("S","I"), col = c("blue", "red"), lty = 1 )

#Joint the two dataframes
Mean_jointed_Data <- inner_join(Mean_Data_DM, Mean_Data_LEAP, by = c('t'))

######PLOT MEAN DYNAMICS
par(mfrow=c(1,2))
plot(Mean_Data_DM$t, Mean_Data_DM$S, type = 'l', col='blue', main = "Dynamique du site moyen : DirectMethod", xlab = "t", ylab = "Densité")
lines(Mean_Data_DM$t, Mean_Data_DM$I, col = 'red')
legend("topright",legend = c("S","I"), col = c("blue", "red"), lty = 1 )

plot(Mean_Data_LEAP$t, Mean_Data_LEAP$S, type = 'l', col='blue', main = "Dynamique du site moyen : TauLeap", xlab = "t", ylab = "Densité")
lines(Mean_Data_LEAP$t, Mean_Data_LEAP$I, col = 'red')
legend("topright",legend = c("S","I"), col = c("blue", "red"), lty = 1 )

################################################################################
###################### INDICES AND REGRESSION PLOT #############################
################################################################################

#Indicescomputation
mod_S <- lm(Mean_jointed_Data$S.x ~ Mean_jointed_Data$S.y)
R2 <- summary(mod_S)$r.squared

RMSE_S <- round(RMSE(Mean_jointed_Data$S.x, Mean_jointed_Data$S.y),2)
RMSE_S
pRMSE_S <- rRMSE(Mean_jointed_Data$S.x, Mean_jointed_Data$S.y)
pRMSE_S
MAE_S <- MAE(Mean_jointed_Data$S.x, Mean_jointed_Data$S.y)

RMSES_S <- RMSEsystematic(Mean_jointed_Data$S.x, Mean_jointed_Data$S.y)
RMSES_S
RMSEU_S <-RMSEuncertainity(Mean_jointed_Data$S.x, Mean_jointed_Data$S.y)
RMSEU_S

RMSES_I <- RMSEsystematic(Mean_jointed_Data$I.x, Mean_jointed_Data$I.y)
RMSES_I
RMSEU_I <- RMSEuncertainity(Mean_jointed_Data$I.x, Mean_jointed_Data$I.y)
RMSEU_I
#Figure

plot(Mean_jointed_Data$S.x, Mean_jointed_Data$S.y ,main = "DirectMethod VS TauLEap : S", xlab = "S Direct Method", ylab = "S TauLeap")
abline(lm(Mean_jointed_Data$S.x ~ Mean_jointed_Data$S.y), col = 'blue',)

text(550,300, paste("RMSE =", round(RMSE_S, 2)))
text(550,250, paste("rRMSE =", round(pRMSE_S, 2)))
text(550,200, paste("MAE =", round(MAE_S, 2)))


#Indices for infected
mod_I <- lm(Mean_jointed_Data$I.x ~ Mean_jointed_Data$I.y)
summary(mod_I)$r.squared

RMSE_I <- RMSE(Mean_jointed_Data$I.x, Mean_jointed_Data$I.y)
RMSE_I
pRMSE_I <- rRMSE(Mean_jointed_Data$I.x, Mean_jointed_Data$I.y)
pRMSE_I
MAE_I <- MAE(Mean_jointed_Data$I.x, Mean_jointed_Data$I.y)
MAE_I

#Figure

plot(Mean_jointed_Data$I.x, Mean_jointed_Data$I.y,main = "DirectMethod VS TauLEap : I", xlab = "I Direct Method", ylab = "I TauLeap")
abline(lm(Mean_jointed_Data$I.x ~ Mean_jointed_Data$I.y), col = 'blue')
text(250,125, paste("RMSE =", round(RMSE_I, 2)))
text(250,100, paste("rRMSE =", round(pRMSE_I, 2)))
text(250,75, paste("MAE =", round(MAE_I, 2)))


################################################################################
#################### KEEP ONLY HIGH DENSITY VALUES #############################
################################################################################

MeanHigh_DM <- subset(Mean_Data_DM, Mean_Data_DM$I > 300)
MeanHighLeap <- subset(Mean_Data_LEAP, Mean_Data_LEAP$I >300)
jointedHD <- inner_join(MeanHigh_DM, MeanHighLeap, by = 't')


#Indices S
mod_S <- lm(jointedHD$S.x ~ jointedHD$S.y)
summary(mod_S)$r.squared

RMSE_S <- RMSE(jointedHD$S.x, jointedHD$S.y)
RMSE_S
pRMSE_S <- rRMSE(jointedHD$S.x, jointedHD$S.y)
pRMSE_S
MAE_S <- MAE(jointedHD$S.x, jointedHD$S.y)
MAE_S

#Figure

plot(jointedHD$S.x, jointedHD$S.y,main = "High density only", xlab = "S Direct Method", ylab = "S TauLeap")
abline(lm(jointedHD$S.x ~ jointedHD$S.y), col = 'blue')
text(375,320, paste("RMSE :", round(RMSE_S, 2)))
text(375,310, paste("rRMSE:", round(pRMSE_S, 2)))
text(375,300, paste("MAE :", round(MAE_S, 2)))

#Indices I
mod_I <- lm(jointedHD$I.x ~ jointedHD$I.y)
summary(mod_I)$r.squared

RMSE_I <- RMSE(jointedHD$I.x,jointedHD$I.y)
RMSE_I
pRMSE_I <- rRMSE(jointedHD$I.x,jointedHD$I.y)
pRMSE_I
MAE_I <- MAE(jointedHD$I.x, jointedHD$I.y)
MAE_I

#Figure

plot(jointedHD$I.x,jointedHD$I.y,main = "High density only", xlab = "I Direct Method", ylab = "I TauLeap")
abline(lm(jointedHD$I.x ~ jointedHD$I.y), col = 'blue')
text(380,325, paste("RMSE :", round(RMSE_I, 2)))
text(380,320, paste("rRMSE:", round(pRMSE_I, 2)))
text(380,315, paste("MAE :", round(MAE_I, 2)))

################################################################################
################# PROJECTION IN PHASE SPACE ####################################
################################################################################

#Select only equilibrium values
plot(Mean_Data_DM$S, Mean_Data_DM$I, type = 'l')
plot(Mean_Data_LEAP$S, Mean_Data_LEAP$I, type = 'l')

#Selecting only quasi-equilibrium values
Mean_EQ_DM <- subset(Mean_Data_DM, Mean_Data_DM$t > 8.5)
Mean_EQ_LEAP <- subset(Mean_Data_LEAP, Mean_Data_LEAP$t >8.5)


plot(Mean_EQ_DM$S, Mean_EQ_DM$I, type = 'l')
plot(Mean_EQ_LEAP$S, Mean_EQ_LEAP$I, type = 'l')

#Variance Measures for predictions
var(Mean_EQ_DM$S) # 90.11
Sd_DMS <- sqrt(var(Mean_EQ_DM$S))
var(Mean_EQ_DM$I) #217.72
Sd_DMI <- sqrt(var(Mean_EQ_DM$I))
cov(Mean_EQ_DM$S, Mean_EQ_DM$I)#26.81

var(Mean_EQ_LEAP$S) # 194.43
Sd_LEAPS <- sqrt(var(Mean_EQ_LEAP$S))
var(Mean_EQ_LEAP$I) # 414.08
Sd_LEAPI <- sqrt(var(Mean_EQ_LEAP$I))
cov(Mean_EQ_LEAP$S, Mean_EQ_LEAP$I)#58.8403

RvarS <- Sd_LEAPS/Sd_DMS 
RvarI <- Sd_LEAPI / Sd_DMI
RvarS
RvarI

dev.off()

