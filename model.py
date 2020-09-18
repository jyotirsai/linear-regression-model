import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import salary data
df = pd.read_csv(r'Salary_Data.csv')

# turn years of experience into X, and salary into Y, find number of training examples m
X = df['YearsExperience'].to_numpy()
Y = df['Salary'].to_numpy()
m = X.size

# guess values of parameters
theta_0 = 0
theta_1 = 10
h = theta_0 + theta_1*X

cost = []
deri = []
deri_J1 = 2
deri_J0 = 2
# gradient descent loop
while abs(deri_J1) > 0.001 and abs(deri_J0) > 0.001:
    h = theta_0 + theta_1*X
    J = np.sum(np.square(h-Y))/(2*m)
    cost.append(J)
    deri_J0 = np.sum(h-Y)/m
    deri_J1 = np.dot(h-Y, X)/m
    deri.append(deri_J1)
    temp0 = theta_0 - 0.01*deri_J0
    temp1 = theta_1 - 0.01*deri_J1
    theta_0 = temp0
    theta_1 = temp1

print(theta_0, theta_1)

df.plot.scatter("YearsExperience", "Salary", c="black")
plt.plot(X, theta_1*X+theta_0)
plt.xlabel("Years of Experiences")
plt.ylabel("Salary")
plt.show()
