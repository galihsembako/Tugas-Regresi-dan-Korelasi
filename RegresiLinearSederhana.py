import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
from scipy.stats import pearsonr


df = pd.DataFrame([[27,20],[19,23],[15,18],[26,25],[17,26],[25,24],[21,23],[14,24],[27,20],[26,22],[23,26],[18,28]])
df.columns = ['x', 'y']
X_train = df['x'].values[:,np.newaxis]
Y_train = df['y'].values

model_reg = LinearRegression()
model_reg.fit(X_train,Y_train) #fase training

#regression coefficients
print('Coefficients b = {}'.format(model_reg.coef_))
print('Constant a ={} '.format(model_reg.intercept_))

#model regresi yang didapat
print('Y = ', model_reg.intercept_ ,'+', model_reg.coef_,'X')

#prediksi satu data jika nilai X = 28
print('Y = {}'.format(model_reg.predict([[28]])))

# Apply the pearsonr()
corr, _ = pearsonr(df['x'],df['y'])
print('Pearsons correlation: %.2f' % corr)

# Apply the koef determination
correlation_matrix = np.corrcoef(df['x'],df['y'])
correlation_xy = correlation_matrix[0,1]
koefdet = correlation_xy**2
print("Koefisien Determinasi : {:.0%}" .format(correlation_xy**2))

#prepare plot
pb = model_reg.predict(X_train)
dfc = pd.DataFrame({'x': df['x'],'y':pb})
plt.scatter(df['x'],df['y'])
plt.plot(dfc['x'],dfc['y'],color='red',linewidth=1)
plt.title('Correlation = : %.2f' % corr)
plt.xlabel('Dampak Virus Covid-19')
plt.ylabel('Laju Ekonomi')
plt.show()