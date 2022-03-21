library(readr)
dir <- choose.dir()
Data <- read_csv("TimeSeries.csv")
Dynamics <- subset(Data[2:6])


plot(Dynamics$t, Dynamics$S0, type = 'l')
lines(Dynamics$t,Dynamics$I0, col='red')


plot(Dynamics$t,Dynamics$S1, type = 'l')















#####OLD #####
################# Construction de sous tableaux d'intérêt########################

taille_col = (length(Dynamics)-1) # On compte le nombre de sites simulés

generate_even_indexes = seq(2, taille_col, 2) # On récupère tous les indexs pairs qui correspondent aux infectés de chaque pop
generate_odd_indexes = seq(3, taille_col+1, 2) # On récupère tous les indexs imparis qui correspondent aux susceptibles de chaque pop

Tableau_s <- subset(Dynamics[generate_even_indexes]) # Contient l'ensemble des s(t) pour chaque site
Tableau_i <- subset(Dynamics[generate_odd_indexes]) # Contient l'ensemble des i(t) pour chaque site

#Prévalence Totale de l'infection dans la métapop ( densité globale d'individus infectés)
length(Tableau_i)
Prev <- data.frame(Dynamics$X1)
Prev[,"Prévalence moyenne"] <- rowMeans(Tableau_i)
Prev[,'Prévalence totale'] <- rowSums(Tableau_i)

#Figures (non acrobatiques)
plot(Prev$Dynamics.X1, Prev$Prévalence, type = 'l') # Pas foufou
plot(Prev$Dynamics.X1, Prev$`Prévalence totale`, type = 'l') # Pas foufou non plus

########## Dynamique des états des sites ###########################

# Compter les sites {0, DFE, END} = f(t)
#0 si s=0 & I=0
#DFE si S= X, I = 0
#END si  S=x, I=y
#Gros problème si S=0, I=Y mais on espère que ca n'arrivera jamais ô grand jamais

States <- read_csv("Island_outputs_states.csv")
Count_states = data.frame()
for (i in 1:length(States))
{
  Count_states[i,1] = States[i,1]
  Count_states[i,2] = sum(States[i,]=='DFE')
  Count_states[i,3] = sum(States[i,]=='END')
  Count_states[i,4] = sum(States[i,]=='VIDE')
  Count_states[i,5] = sum(States[i,]=='NA')
  }
colnames(Count_states)<- c("t","DFE", "END", "VIDE", "NA") # Renommer les colonnes correctement sinon c'est moche


# Graphe de la dynamique des états des sites
plot(Count_states$t, Count_states$DFE, type = 'l', xlab = 'temps', ylab = 'Densité de sites',xlim = c(0, 100), ylim = c(0, 100))
lines(Count_states$END)
lines(Count_states$VIDE)
