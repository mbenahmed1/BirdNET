import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#q_df = pd.read_csv('quality.csv')
m_df = pd.read_csv('meta.csv')

print(m_df.head(5))

ax = sns.barplot(x="Overlap", y="Processing_time", data=m_df, palette="Blues_d")
ax.set_yscale('log')
plt.show()