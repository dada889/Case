import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

################ Load the data ###################


housing = pd.read_csv("housing.data",delim_whitespace=True,header=None)
colnames = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
housing.columns = colnames
housing.dtypes

############## statistics summary

housing_1 = housing.drop('CHAS',1).drop('RAD',1)
housing_1.describe()

housing_2 = housing[['CHAS','RAD']].copy()
housing_2 = housing_2.astype(str)
#housing_2.dtypes
housing_2.describe()



count = housing_2['RAD'].value_counts()
df_count = pd.DataFrame(count)
df_count.columns = ['count']
df_count.index.name = 'RAD'
df_count.transpose()

################## Visualize the data ######################

##### hist of MEDV
fig = plt.figure(figsize=(15,8))
plt.ylabel('frequency', fontsize=15)
plt.xlabel('Median value of owner-occupied homes in $1000\'s',fontsize=15)
plt.title('Histogram of MEDV \n',fontsize=20)
housing.MEDV.hist(bins=200)


##### boxplot for categorical attributes

import matplotlib.gridspec as gridspec
import matplotlib as mpl
import matplotlib.pyplot as plt


## CHAS
CHAS_0 = housing[housing.CHAS==0]
CHAS_1 = housing[housing.CHAS==1]

fig = plt.figure(figsize=(14,6))
gs = gridspec.GridSpec(1, 2)
ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[0,1])
sns.boxplot(housing.MEDV,housing.CHAS,ax=ax1)
#ax1.set_title('')
#ax1.set_xlabel('')
ylim = ax1.get_ylim()
CHAS_0.iloc[:,-1].hist(orientation='horizontal',bins=80,alpha=0.4,ax=ax2)
CHAS_1.iloc[:,-1].hist(orientation='horizontal',bins=80,alpha=0.6,ax=ax2)
ax2.set_ylim((ylim[0], ylim[1]))
#ax2.set_xlabel('')
for tick in ax2.yaxis.get_major_ticks():
    tick.label1On = False
    tick.label2On = False
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0)




## RAD
sns.boxplot(housing.MEDV,housing.RAD)




####### regplot for continuous attributes

df = housing_1.drop("MEDV",1)

fig = plt.figure(figsize=(10,20))
gs = gridspec.GridSpec(11, 2)

for i in range(11):
    ax1 = plt.subplot(gs[i,0])
    ax2 = plt.subplot(gs[i,1])    
#    sns.boxplot(df[df.columns[i]],groupby=df.quality,ax=ax1)
    sns.regplot('MEDV',df.columns[i],housing_1,ax=ax1)
    ax1.set_title('')
    ax1.set_xlabel('')
    
    ylim = ax1.get_ylim()   
    df[df.columns[i]].hist(bins=80,ax=ax2,orientation='horizontal')    
    ax2.set_ylim((ylim[0], ylim[1]))
    
    ax2.set_xlabel('')
    ax2.set_xlim((0,300))
    for tick in ax2.yaxis.get_major_ticks():
        tick.label1On = False
        tick.label2On = True
    if i!=0:
        ax1.set_xticklabels([''])
        ax2.set_xticklabels([''])
    else:
        ax1.set_title('MEDV \n',size=15)
        ax2.set_title('count \n',size=15)
        for tick in ax1.xaxis.get_major_ticks():
            tick.label1On = False
            tick.label2On = True
        for tick in ax2.xaxis.get_major_ticks():
            tick.label1On = False
            tick.label2On = True
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0)




################ Working on dummy varibal ######################
df = housing.copy()
#df.drop('CHAS',1)
df.drop('RAD',1)

#chas = pd.get_dummies(housing['CHAS'],prefix='CHAS_')
##df1 = pd.concat([df,chas],axis=1)
#rad = pd.get_dummies(df.RAD,prefix='RAD_')
#temp = df.CHAS.astype(str)


## convert RAD to dummy variables 
rad = pd.get_dummies(df.RAD,prefix='RAD_')
df1 = pd.concat([df,rad[rad.columns[1:]]],axis=1) 





########### set up the training set and test set #############
df=df1
import random
train_idx = random.sample(df.index,int(0.7*len(df)))
df_train = df.ix[train_idx]
df_test = df[~df.index.isin(train_idx)]

dfx_train = df_train.drop('MEDV',1)
dfy_train = df_train.MEDV

dfx_test = df_test.drop('MEDV',1)
dfy_test = df_test.MEDV


def train_and_evaluate(clf,X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    
    print "Accuracy on training set:"
    print clf.score(X_train, y_train)
    print "Accuracy on test set:"
    print clf.score(X_test, y_test)


from sklearn import linear_model
lm = linear_model.LinearRegression()
train_and_evaluate(lm,dfx_train,dfx_test,dfy_train,dfy_test)


############## fit the model ###############################


from sklearn import linear_model
lm = linear_model.LinearRegression()
lm.fit(dfx_train,dfy_train)

colnames = dfx_train.columns.values
result = pd.DataFrame(lm.coef_).transpose()
result.columns = colnames.tolist()
result['intercept'] = lm.intercept_ 
result = result.transpose()
result.columns = ['coefficient']

result



############ prediction ######################################

housing_test = housing[~df.index.isin(train_idx)]
housing_test_x = housing_test.drop('MEDV',1)
#housing_test_y = dfy_test



housing_test_pred = lm.predict(dfx_test)
pred = housing_test_x.copy()
pred['predicted_MEDV'] = housing_test_pred
pred['actual_MEDV'] = dfy_test
pred.head()







