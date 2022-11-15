import pandas as pd
import matplotlib.pyplot as plt
import math

df = pd.read_csv("Linear regression\Health.csv")

# selecting the data on which we apply linear regression
df = df[["Low_Confidence_Limit", "Data_Value"]]
df = df.iloc[0:100, :]

# function to return improved m and improved c over the previous one
def gardient_descent(m, c, points, m_rate, c_rate):
    m_loss = 0
    c_loss = 0

    n = len(points)

    for i in range(n):
        x = float(points.iloc[i].Data_Value)
        y = float(points.iloc[i].Low_Confidence_Limit)
        if (not (math.isnan(x) or math.isnan(y))):

            # adding the derivates of the root mean squares to the loss function
            m_loss += -1 * (2 / n) * x * (y - (m * x + c))
            c_loss += -1 * (2 / n) * (y - (m * x + c))

   # improving the m and c based on the limit
    m = (m - m_rate * m_loss)
    c = (c - c_rate * c_loss)

    return m, c


# imitial values and rate
m = 1
c = 10
m_rate = 0.0001
c_rate = 0.01
epochs = 3000

# performing the imrovements to m and c over and over again
for i in range(epochs):
    if i % 50 == 0:
        print(i, m, c)
    m, c = gardient_descent(m, c, df, m_rate, c_rate)

# plotting the graph for the regrssion line and scattered points
plt.scatter(df.Data_Value, df.Low_Confidence_Limit)
plt.plot(list(range(0, 60)), [m * x + c for x in range(60)], color="red")
plt.show()
