import numpy as np
#Model parameters
beta = 0.005  # Infectious contact rate
r = 1.5  # Per capita growth rate
k = 1000  # Carrying capacity
d = 0.05  # Dispersal propensity
gamma = 1.5  # Parasite Clearance
alpha = 0.10  # Parasite Virulence
rho = 0.1  # Dispersal Cost
epsilon = 0.1  # Extinction rate

class Site(): #Site object containing (non explicit) individuals
    def __init__(self,effectifS,effectifI): #First arg correspond to "IsEvolving"
        self.effectifS = effectifS #S density
        self.effectifI = effectifI #I density
        self.traitvalues = []
        #### DEFINE TRAITS VALUES AS A VECTOR OF LENGTH I (FOR ALPHA EVOLVING) ####

        #self.pos = pos (tuple) : for future improvements, position of the site on the network grid, as matrix coordinates
        #self.neighbor = [] : for future, maybe including neighbors as an attribute (so it's computed only once)

    def Settraitvalues(self): #THIS IS PROBABLY USELESS but let's keep it
        for i in range(self.effectifI):
            self.traitvalues.append(np.random.uniform(0.01,10,1))

class EvolvingTrait():
    def __init__(self, name, IsMutation): #str, bool
        self.Traitname = name
        self.TraitMutation = IsMutation

class Event():
    def __init__(self,name, propensity, Schange, Ichange, order, EvolvingTrait):
        self.name = name # Event name in letter and not in memory address, handful to identify what's happening
        self.S = 0 #Has to take density values to understand the maths
        self.I = 0
        self.formula = propensity # The unique formule (str) given by model construction
        self.propensity = eval(self.formula)#Convert string in maths instruction, very useful to externalise model building
        self.Ichange = eval(Ichange) # State Change due to event, Typically -1, 0, 1 except for extinctions
        self.Schange = eval(Schange)
        self.order = order # Reaction order, not really useful but we never know
        self.EvolvingTrait = EvolvingTrait # Is a 'Evolving trait' Object that has a name, and an enabler parameter for allowing mutation



    #Reminder : Updatepropensity function is called in the 'function' file
    def UpdatePropensity(self, S, I, TraitValues): # Class method to compute propensities without creating new objects
        if self.EvolvingTrait.Traitname in self.formula : # if the event has the name of evolving trait in his propensity formula
            SumtraitValues = sum(TraitValues) # Do some shit to take into account the sum of trait values instead of parameter value
            if self.EvolvingTrait.Traitname == 'alpha':
                self.S = S
                self.I = I
                self.propensity = SumtraitValues
        else : #Otherwise take the fixed value given
            self.S = S
            self.I = I
            self.propensity = eval(self.formula)  # Changes propensity values while keeping formula untouched
        return self.propensity

