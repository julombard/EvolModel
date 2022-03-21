# Analysing Python model outputs
library(readr)
Metapop_Outputs <- read_csv("Metapop_Outputs.csv")

################################ BASIC CHECKS #########################################

#Dropping the first column because it's useless (number of iterations, not model time)
Data <- subset(Metapop_Outputs[2:length(Metapop_Outputs)])

#Check if local dynamics seems correct
plot(Data$t, Data$S0, type = 'l', col ='blue')
lines(Data$t, Data$I0, col = 'red')

#Check if migration can be seen locally
plot(Data$t, Data$S7, type = 'l', col ='blue')
lines(Data$t, Data$I7, col = 'red')

################################ PROPENSITIES CHECK ####################################
Prop_outputs <- read_csv("Propensities_outputs.csv")

plot(Prop_outputs$t, Prop_outputs$`Reproduction S`, type = 'l', col = 1)
lines(Prop_outputs$t, Prop_outputs$`Death S`, col = 2)
lines(Prop_outputs$t, Prop_outputs$`Death I`, col = 3)
lines(Prop_outputs$t, Prop_outputs$`Dispersal I`, col = 5)
lines(Prop_outputs$t, Prop_outputs$`Dispersal S`, col = 6)
lines(Prop_outputs$t, Prop_outputs$Extinction, col = 7)
lines(Prop_outputs$t, Prop_outputs$Infection, col = 8)
lines(Prop_outputs$t, Prop_outputs$Recovery, col = 9)



########################## GLOBAL DENSITIES AND PREVALENCE #############################
taille_col = (length(Data)-1) # Number of populations columns 

generate_even_indexes = seq(2, taille_col, 2) # Even indexes are S populations
generate_odd_indexes = seq(3, taille_col+1, 2) # Odd indexes are I populations

Spops <- subset(Data[generate_even_indexes])
Ipops <- subset(Data[generate_odd_indexes])

# We sum by lines to get global densities
Stot <- rowSums(Spops)
Itot <- rowSums(Ipops)

Global_Data <- data.frame(t = Data$t, S = Stot, I=Itot)

plot(Global_Data$t, Global_Data$S, type = 'l', col='blue')
lines(Global_Data$t, Global_Data$I, col = 'red')
