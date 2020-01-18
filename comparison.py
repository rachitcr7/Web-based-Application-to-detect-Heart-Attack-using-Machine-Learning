import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

methods = ["Logistic Regression", "Naive Bayes", "Decision Tree", "Random Forest"]
accuracy = [83.61, 85.25, 77.05, 83.61]
colors = ["yellow", "green", "orange", "red"]

sns.set_style("whitegrid")
plt.figure(figsize=(10,5))
plt.yticks(np.arange(0,100,10))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot(x=methods, y=accuracy, palette=colors)
plt.savefig('heartDiseasecomparisons.png')
plt.show()