import pandas as pd
df = pd.read_csv("titanic.csv")
survived = df[df['Survived'] == 1]
not_survived = df[df['Survived'] == 0]
print("Survived Data:")
print(survived)
print("\nNot Survived Data:")
print(not_survived)
import seaborn as sns
import matplotlib.pyplot as plt
# Plot countplot of survivors by sex
plt.figure()
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title('Survivors by Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['Not Survived', 'Survived'])
plt.show()

# Plot countplot of survivors by passenger class
sns.figure()
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Survivors by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['Not Survived', 'Survived'])
plt.show()

# Plot age distribution of survivors and non-survivors
plt.figure()
sns.histplot(data=df, x='Age', hue='Survived', multiple='stack', bins=30)
plt.title('Age Distribution of Survivors and Non-Survivors')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['Not Survived', 'Survived'])
plt.show()

#price scatter plot of fare vs age and survival
plt.figure()
sns.scatterplot(data=df, x='Age', y='Fare', hue='Survived', style='Survived')
plt.title('Fare vs Age by Survival Status')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.legend(title='Survived', labels=['Not Survived', 'Survived'])
plt.show()

#fare and survival scatter plot with size based on fare
plt.figure()
sns.scatterplot(data=df, x='Pclass', y='Fare', hue='Survived', size='Fare', sizes=(20, 200))
plt.title('Fare vs Passenger Class  by Survival Status')
plt.xlabel('Passenger Class')
plt.ylabel('Fare')
plt.legend(title='Survived', labels=['Not Survived', 'Survived'])
plt.show()