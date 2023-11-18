
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf # these draw plots and make confidence bounds automatically

from google.colab import files
uploaded = files.upload()

import io
df = pd.read_csv(io.BytesIO(uploaded['usd-inr.csv']))

df = pd.read_csv('usd-inr.csv', index_col='DATE', parse_dates=True)

fig, ax = plt.subplots(figsize=(10,5)) # Plot_pacf for AR(p) and subplot function is manage the size of the plot
plot_pacf(df['RATE'], ax=ax)     # P=3

fig, ax = plt.subplots(figsize=(15,5)) # Plot_pacf for AR(p) and subplot function is manage the size of the plot
plot_acf(df['RATE'], ax=ax) 
#q=40

df['Diff'] = df['RATE'].diff()

fig, ax = plt.subplots(figsize=(15,5)) # Plot_pacf for AR(p) and subplot function is manage the size of the plot
plot_acf(df['Diff'], ax=ax)

x0 =  np.random.randn(1000)  # Generating IID noics from standard normal

plt.plot(x0)

fig, ax = plt.subplots(figsize=(10,5)) # Plot_pacf for AR(p) and subplot function is manage the size of the plot
plot_pacf(x0, ax=ax)   # this return an acccess ax which then pass into plot function  and we can see most of lagged PACF value very near to zero
 # 1st value is 1 since it meanr autocorrelation of each point wrt itself
 # so our time series is AR(1) process

x1 =[0]
for i in range(1000):
  x=0.5 * x1[-1] +0.1*np.random.randn()
  x1.append(x)
x1=np.array(x1)

x1

plt.plot(x1)

fig, ax = plt.subplots(figsize=(10,5)) # Plot_pacf for AR(p) and subplot function is manage the size of the plot
plot_pacf(x1, ax=ax)  # so our time series is AR(1) process

x2 = [0,0]
for i in range(1000):
  x=0.5 * x2[-1] - 0.3 * x2[-2] + 0.1 *np.random.randn()  #Creating AR(2) processes
  x2.append(x)
x2=np.array(x2)

plt.plot(x2)

fig, ax = plt.subplots(figsize=(10,5)) # Plot_pacf for AR(p) and subplot function is manage the size of the plot
plot_pacf(x2, ax=ax)

x5 = [0,0,0,0,0]
for i in range(1000):
  x=0.5 * x5[-1] - 0.3 * x5[-2] - 0.6 * x5[-5] + 0.1 *np.random.randn()  #Creating AR(2) processes
  x5.append(x)
x5=np.array(x5)

plt.plot(x5)

fig, ax = plt.subplots(figsize=(10,5)) # Plot_pacf for AR(p) and subplot function is manage the size of the plot
plot_pacf(x5, ax=ax)

"""ACF

"""

# ACF
fig, ax = plt.subplots(figsize=(10,5)) 
plot_acf(np.random.randn(1000), ax=ax)

#creting a MA(1)
errors =0.1 * np.random.randn(1000)
mal=[]
for i in range(1000):
  if i >=1:
    x=0.5*errors[i-1]+errors[i]
  else:
    x=errors[i]
  mal.append(x)
mal=np.array(mal)

plt.plot(mal)

fig, ax = plt.subplots(figsize=(10,5)) 
plot_acf(mal, ax=ax)

#creting a MA(2)
errors =0.1 * np.random.randn(1000)
ma2=[]
for i in range(1000):
  x=0.5*errors[i-1] - 0.3 * errors[i-2] + errors[i]
  ma2.append(x)
ma2=np.array(ma2)

plt.plot(ma2)

fig, ax = plt.subplots(figsize=(10,5)) 
plot_acf(ma2, ax=ax)

#creting a MA(3)
errors = 0.1 * np.random.randn(1000)
ma3 = []
for i in range(1000):
  x= 0.5 * errors[i-1] - 0.3 * errors[i-2] + 0.7 * errors[i-3] + errors[i]
  ma3.append(x)
ma3=np.array(ma3)

plt.plot(ma3)
fig, ax = plt.subplots(figsize=(10,5)) 
plot_acf(ma3, ax=ax)

#If look at ACF plot, largest non-zero lag is 3 note that there is one non-zero lag at 25 but as per statistics testing this is supposed to happen about 5% of the time
#so we choose MA(3) instead of MA(5)
#Similarlly, we can generate MA(4),MA(5),MA(6),.......

