#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd                       
import random   
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from copy import deepcopy
from math import exp 


# In[2]:


data = pd.read_excel("Training_Data.xlsx")
data


# In[3]:


data.head()


# # Data Pre-Processing

# In[4]:


#shape of the data
data.shape


# In[5]:


#number of missing values in each variable
data.isnull().sum()


# In[6]:


data['EncounterId'].mode()


# In[7]:


def PreProcessing(data): 
    
    data['EncounterId'] = data['EncounterId'].astype(str)
    data['EncounterId'] = data['EncounterId'].map(lambda x: x.lstrip('PH').rstrip(''))
    data['Race'].fillna('White', inplace = True)
    data['DiabetesMellitus'].fillna('Unknown', inplace = True)
    data['ChronicKidneyDisease'].fillna('Unknown', inplace = True)
    data['Anemia'].fillna('Unknown', inplace = True)
    data['Depression '].fillna('Unknown', inplace = True)
    data['ChronicObstructivePulmonaryDisease'].fillna('Unknown', inplace = True)

PreProcessing(data) 


# In[8]:


data.head()


# # Population

# In[9]:


population = []

def inital_population():
    
    array = data.columns.to_list()
    
    for i in range(0,40):    
        temp = list()
        while(True):
            temp = random.sample(array, 30)
            if "ReadmissionWithin_90Days" in temp:
                if len(np.unique(temp)) == 30:
                    if temp not in population:
                        print(temp)
                        break
                        
            
            population.append(temp)


# In[10]:


inital_population()


# # Train the Model

# In[11]:


data.dtypes


# In[12]:


le = LabelEncoder()


# In[13]:


data['EncounterId'] = le.fit_transform(data['EncounterId'])


# In[14]:


data['DischargeDisposision'] = le.fit_transform(data['DischargeDisposision'])


# In[15]:


data['Gender'] = le.fit_transform(data['Gender'])


# In[16]:


data['Race'] = le.fit_transform(data['Race'])


# In[17]:


data['DiabetesMellitus'] = le.fit_transform(data['DiabetesMellitus'])


# In[18]:


data['ChronicKidneyDisease'] = le.fit_transform(data['ChronicKidneyDisease'])


# In[19]:


data['Anemia'] = le.fit_transform(data['Anemia'])


# In[20]:


data['Depression '] = le.fit_transform(data['Depression '])


# In[21]:


data['ChronicObstructivePulmonaryDisease'] = le.fit_transform(data['ChronicObstructivePulmonaryDisease'])


# In[22]:


data['ReadmissionWithin_90Days'] = le.fit_transform(data['ReadmissionWithin_90Days'])


# # Fitness

# In[23]:


fit = []

def fitness():
    
    for i in range(0,40):
             
        column = population[i]
       
        x = pd.DataFrame(data, columns = column)
        y = data['ReadmissionWithin_90Days']
        x_train,x_test,y_train,y_test=train_test_split(x, y, test_size = 0.2, random_state = 101)

        #training a logistics regression model
        logistic = LogisticRegression(solver='lbfgs', max_iter=100)
        logistic.fit(x_train,y_train)
        predictions = logistic.predict(x_test)
        print("Accuracy = "+ str(accuracy_score(y_test,predictions) * 100))
        fit.append(accuracy_score(y_test,predictions)* 100)
        
        


# In[24]:


fitness()


# In[25]:


fit


# # Selection

# In[26]:


import numpy.random as npr


# In[27]:


def selection(fitness):
    
    max = sum([c for c in fitness])
    selections = [c / max for c in fitness]
    
   
    
    return fitness.index(fitness[npr.choice(len(fitness), p = selections)])


# In[28]:


fit2 = fit.copy()

val1 = selection(fit)
val2 = selection(fit2)
parnets1 = population[val1]
parnets2 = population[val2]
print(val1)
print(parnets1)
print(val2)
print(parnets2)


# # CrossOver

# In[29]:


child_1 = []
child_2 = []

def crossover(parnet1, parnet2):
    
    random = np.random.randint(0, 30)
    
    print("Crossover point:", random)
    global child1
    global child2
    child1 = parnet1[0:random] + parnet2[random:]
    child2 = parnet2[0:random] + parnet1[random:]
    
    child_1.append(child1)
    child_2.append(child2)

    
    return child1,child2
    


# In[30]:


crossover(parnets1, parnets2)


# # Mutation

# In[31]:


childt_1 = []
childt_2 = []

def mutation(child3, child4):
        
        prob = np.random.randint(0,30) 
        threshold = 80
        
        if prob <= threshold:
        
            random_index = np.random.randint(0,30)  

            print("Mutation point:", random_index)
            
#             print(child_1)
#             print(child_2)

            col = data.columns.to_list()
    
            
        
            child3[random_index] = random.sample(col, 1)[0]
            child4[random_index] = random.sample(col, 1)[0]
            childt_1.append(child3)
            childt_2.append(child4)
            
            return child3,child4
            
            

           


# In[32]:


mutation(child1, child2)


# # Accuracy of the Genertic Algorithm

# In[33]:


epochs = 1

while(max(fit) != 72 and epochs!=0):
    
   
        selection(fit)

        crossover(parnets1, parnets2)

        mutation(parnets1, parnets2)



        epochs = epochs - 1
        
   


    
    


# In[34]:


index = fit.index(max(fit))
print(max(fit))
    


# # Columns Selected in GA

# In[35]:


len(population[index])


# # Columns Not Selected In GA

# In[36]:


print("Columns not selected in GA")

selection=population[index]
count=0
for i in data.columns:
    if i not in selection:
        print(i)
        count=count+1
        
print(count)


# In[ ]:




