#Check for stochastic extinction

#We get the I = 0 for each site when I was = to 1 at the previous time

#Done with DirectMethod algorithm as

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
#namereport <- 'rapport_eval.pdf'
#pdf(paste(dir, namereport, sep = '\\'), onefile = T)
#dev.off()

#Read files
cwd <- setwd(dirDMOut)
listfiles <- list.files(cwd)
listfiles
list_tibbles <- lapply(listfiles, read_csv) # Return tibble ? What is that **** ?
list_df <- lapply(list_tibbles, as.data.frame) # Convert to good old dataframe

#In the search of stochastic extinction
#We get the I = 0 for each site when I was = to 1 at the previous time, for each sim

#Supress 2 first columns (because useless) and round 't' column to 3 digits (to perform comparison of same t in DM and TauLeap)
for (i in 1:length(list_df)){
  
  list_df[[i]]<- subset(list_df[[i]][c(-1,-2)]) # supress two first columns
  list_df[[i]]$t <- round(list_df[[i]]$t, 3) #round t values

}

Ipopsdf <- list()
for (i in 1:length(list_df)){
  dataframe <- as.data.frame(list_df[[i]])
  taille_col = (length(dataframe)-1) # Number of populations columns 
  #Get only infected values
  generate_odd_indexes = seq(3, taille_col+1, 2) # Odd indexes are I populations
  #generate_odd_indexes
  Ipopsdf[[i]] <- as.data.frame(subset(dataframe[generate_odd_indexes]))

  
}

#Chek values
sum_extinct_total <- c(0)
for (j in length(Ipopsdf)){
  sum_extinct_persim <- c(0)
  for(i in 1: length(Ipopsdf[[j]])){
  
    zero_rows_index <- which(Ipopsdf[[j]][,i]== 0)
    length(Ipopsdf[[j]][,i])
    zero_rows_index
    Indexes_toCheck <- zero_rows_index-1
  
    Stochastic_extinction_index <- which(Ipopsdf[[j]][Indexes_toCheck,i] ==1)
    Stochastic_extinctions <- Indexes_toCheck[Stochastic_extinction_index]
    nb_exctintion <- length(Stochastic_extinctions)
    sum_extinct_persim <- sum_extinct_persim + nb_exctintion
  }
  sum_extinct_total <- sum_extinct_total+sum_extinct_persim
} 
Mean_number_extinct <- mean(sum_extinct_total)
#Moyenne de 15 extinctions stochastiques par metapopulation
#Mais on finit avec un équilibre endémique ou tous les sites sont infectés avec succès
