import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")
df['Date'] = pd.Series(range(1, len(df) + 1))
df = df.dropna()

# Find distribution of Calories
print("Distribution of Calories:")
print(df['Calories'].describe())

# Plot combined line graph for Calories and Maxpulse
plt.figure()
plt.plot(df['Calories'].values, label='Calories', marker='o')
plt.plot(df['Maxpulse'].values, label='Maxpulse', marker='s')
plt.title('Calories and Maxpulse over Date')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.savefig('combined_line.png')
print("Combined line graph saved as combined_line.png")

