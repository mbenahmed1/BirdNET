import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 

#sns.set_palette("vlag")
sns.set_style("whitegrid")
map_img = mpimg.imread('s1_spectrogram.png') 
# load data from csv
df = pd.read_csv("./s1_results.csv")

# show first five entries
print(df.head(5))

df = df.loc[df['Confidence'] >= 0.5]
df = df.loc[(df['Overlap'] == 1.0) | (df['Overlap'] == 2.5) | (df['Overlap'] == 2.75)]

# box plot
#bp = sns.catplot(x="distance", y="mbit/s", kind="box", hue="protocol", data=df)
#bp.set(xlabel="Abstand [m]", ylabel="Durchsatz [mbit/s]", title="Durchsatzmessung über 60 Sekunden")

# line plot
ln = sns.relplot(
    data=df,
    x="Begin",
    y="Confidence",
    hue="Name",
    style="Overlap",
    kind="line",
    markers=True   
)
ln.set(xlabel="Time [s]", ylabel="Confidence [%]", title="Classification Confidence (example/Soundscape_1.wav)")
# bar plot
# bar = sns.catplot(x="distance", y="mbit/s", hue="protocol", kind="bar", data=df)
# bar.set(xlabel="Abstand [m]", ylabel="Durchsatz [mbit/s]", title="Durchsatzmessung über 60 Sekunden")


ln.set(xlim=(0, 60), ylim=(0.5, 1.0))
plt.imshow(map_img, extent=[0, 60, 0.5, 1.0], aspect='auto')
plt.show()