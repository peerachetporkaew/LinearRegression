import pandas as pd # data processing
import numpy as np # working with arrays
import matplotlib.pyplot as plt # visualization
import seaborn as sb # visualization
from termcolor import colored as cl # text customization

from sklearn.model_selection import train_test_split # data split
from sklearn.linear_model import LinearRegression # OLS algorithm

# evaluation metric
from sklearn.metrics import explained_variance_score as evs 
from sklearn.metrics import r2_score as r2 

sb.set_style('whitegrid') # plot style
plt.rcParams['figure.figsize'] = (20, 10) # plot size

df = pd.read_csv('house.csv')
df.set_index('Id', inplace = True)
x = df.head(5)



#df.dropna(inplace=True)

print(x)

sb.heatmap(df.corr(), annot = True, cmap = 'magma')
plt.savefig('heatmap.png')
plt.show()


y_var = 'SalePrice'    
scatter_df = df.drop(y_var, axis = 1)
i = df.columns

plot1 = sb.scatterplot(i[3], y_var, data = df, color = 'orange', edgecolor = 'b', s = 150)
plt.title('{} / Sale Price'.format(i[3]), fontsize = 16)
plt.xlabel('{}'.format(i[3]), fontsize = 14)
plt.ylabel('Sale Price', fontsize = 14)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.savefig('scatter1.png')
plt.show()

sb.distplot(df['SalePrice'], color = 'r')
plt.title('Sale Price Distribution', fontsize = 16)
plt.xlabel('Sale Price', fontsize = 14)
plt.ylabel('Frequency', fontsize = 14)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)

plt.savefig('distplot.png')
plt.show()


df = df[['LotArea', 'GarageArea','SalePrice']]
df.dropna(inplace=True)

#X_var = df[['LotArea', 'MasVnrArea', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF']].values

X_var = df[['LotArea', 'GarageArea']].values
y_var = df['SalePrice'].values

X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size = 0.2, random_state = 0)

print(cl('X_train samples : ', attrs = ['bold']), X_train[0:5])
print(cl('X_test samples : ', attrs = ['bold']), X_test[0:5])
print(cl('y_train samples : ', attrs = ['bold']), y_train[0:5])
print(cl('y_test samples : ', attrs = ['bold']), y_test[0:5])

ols = LinearRegression()
ols.fit(X_train, y_train)
ols_yhat = ols.predict(X_test)   
print(cl('R-Squared of OLS model is {}'.format(r2(y_test, ols_yhat)), attrs = ['bold']))

ols = LinearRegression()
ols.fit(X_train, y_train)