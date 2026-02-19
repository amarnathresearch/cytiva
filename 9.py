import pandas as pd
df = pd.read_csv("insurance.csv")
df_smokers = df[df['smoker'] == 'yes']
df_nonsmokers = df[df['smoker'] == 'no']
print("Smokers Data:")
print(df_smokers)
print("\nNon-Smokers Data:")
print(df_nonsmokers)

import matplotlib.pyplot as plt
# Plot histogram of charges for smokers and non-smokers
plt.figure()
plt.hist(df_smokers['charges'], bins=30, alpha=0.5, label='Smokers', color='red')
plt.hist(df_nonsmokers['charges'], bins=30, alpha=0.5, label='Non-Smokers', color='blue')
plt.title('Distribution of Charges for Smokers and Non-Smokers')
plt.xlabel('Charges')
plt.ylabel('Frequency')
plt.legend()

plt.figure()
# plot smokers and non smokers with insurance claims
plt.scatter(df_smokers['insuranceclaim'], df_smokers['charges'], label='Smokers', color='red', alpha=0.5)
plt.scatter(df_nonsmokers['insuranceclaim'], df_nonsmokers['charges'], label='Non-Smokers', color='blue', alpha=0.5)
plt.title('insurance claim vs Charges for Smokers and Non-Smokers')
plt.xlabel('insurance claim')
plt.ylabel('Charges')
plt.legend()

# plot age vs charges
plt.figure()
plt.scatter(df_smokers['age'], df_smokers['charges'], label='Smokers', color='red', alpha=0.5)
plt.scatter(df_nonsmokers['age'], df_nonsmokers['charges'], label='Non-Smokers', color='blue', alpha=0.5)
plt.title('Age vs Charges for Smokers and Non-Smokers')
plt.xlabel('Age')
plt.ylabel('Charges')
plt.legend()

import seaborn as sns
# Plot correlation heatmap
plt.figure()
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')    

plt.figure()
# Plot boxplot of charges by smoker status
sns.boxplot(x='smoker', y='charges', data=df)
plt.title('Boxplot of Charges by Smoker Status')
plt.show()

plt.figure()
# Plot boxplot of age by smoker status
sns.boxplot(x='smoker', y='age', data=df)
plt.title('Boxplot of Age by Smoker Status')
plt.show()

plt.figure()
# Plot boxplot of bmi by smoker status
sns.boxplot(x='smoker', y='bmi', data=df)
plt.title('Boxplot of BMI by Smoker Status')
plt.show()
plt.figure()

# Plot boxplot of children by smoker status
sns.boxplot(x='smoker', y='children', data=df)
plt.title('Boxplot of Children by Smoker Status')
plt.show()
plt.figure()

# plot 3d scatter plot of age, bmi and charges
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_smokers['age'], df_smokers['bmi'], df_smokers['charges'], label='Smokers', color='red', alpha=0.5)
ax.scatter(df_nonsmokers['age'], df_nonsmokers['bmi'], df_nonsmokers['charges'], label='Non-Smokers', color='blue', alpha=0.5)
ax.set_title('3D Scatter Plot of Age, BMI and Charges')
ax.set_xlabel('Age')
ax.set_ylabel('BMI')
ax.set_zlabel('Charges')
ax.legend()
plt.show()

#heat map for smokers and non smokers separately for charges, age, bmi, children
plt.figure()
sns.heatmap(df_smokers[['charges', 'age', 'bmi', 'children']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap for Smokers')
plt.show()
plt.figure()
sns.heatmap(df_nonsmokers[['charges', 'age', 'bmi', 'children']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap for Non-Smokers')
plt.show()

#sex scatter plot with charges for smokers and non smokers
plt.figure()
sns.scatterplot(x='sex', y='charges', hue='smoker', data=df)
plt.title('Sex vs Charges for Smokers and Non-Smokers')
plt.show()
