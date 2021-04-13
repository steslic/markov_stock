import numpy as np 
import yfinance as yf
import pandas as pd
import talib as ta

import matrix_ops

# Sebastian Teslic 
# NOT COMPLETE
# 04/10/2021
# 
# To do: Volume implementation

# download info from Yahoo Finance
print('Enter dates in this format: YYYY-MM-DD')
stock = yf.download(input('Enter a ticker: '), start = '2018-01-01', end = '2018-12-31') 

# Price changes
data = stock[['Open', 'High', 'Low', 'Adj Close']].copy() 

# at the end of the day
data['% Return'] = data['Adj Close'].pct_change() 


# states
data['State'] = data['% Return'].apply(lambda x: '+' if (x > 0.0015)
                                       else '-' if (x < -0.0015)
                                       else 'Flat')



# previous day price 
data['Yesterday State'] = data['State'].shift(1)


# now, frequency distribution matrix 
# --------------------------------------------------------------------------------
# dropna() drops rows where at least one element is missing
states = data[['Yesterday State', 'State']].dropna()

# size() counts number of elements
states_freqdist = states.groupby(['Yesterday State', 'State']).size().unstack()


# axis = 1: calculate with each column as a bulk
# axis = 0: for rows
transition_matrix = states_freqdist.apply(lambda x: x / float(x.sum()), axis = 1) 
transition_matrix = np.transpose(transition_matrix) # stochastic matrix with column vectors


tm = transition_matrix.copy()
print(tm, '\n')

nparr = np.asarray(tm)
nparr_og = np.copy(nparr)

# get rid of NaN to calculate steady state vector
for rows in range(len(nparr)):
    for cols in range(len(nparr[0])):
        if np.isnan(nparr[rows][cols]):
            nparr[rows][cols] = 0


nparr = nparr - np.identity(3)
zeros = np.zeros((3, 1))
# append zero column for solution
nparr = np.append(nparr, zeros, 1)

echelon = matrix_ops.forwardElimination(nparr)
rref = matrix_ops.backsubstitution(echelon)


x_vector = rref[:, 2]
x_vector = -x_vector
# write in terms of free var
x_vector[2] = 1
x_vector = x_vector / sum(x_vector)

print('Steady state vector:', tm @ x_vector)
