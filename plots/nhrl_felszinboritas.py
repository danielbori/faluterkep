import pandas as pd
import seaborn as sns
import numpy as np
import os
import matplotlib.pyplot as plt
import rasterio

def felszin_csoport(df):
    felszin = {}
    felszin['Szántó'] = (df[df['Value'] == 21]['Count'].iloc[0] +
                         df[df['Value'] == 22]['Count'].iloc[0]
                         + df[df['Value'] == 25]['Count'].iloc[0]) / 100
    felszin['Erdő'] = (df[df['Value'] == 31]['Count'].iloc[0] +
                       df[df['Value'] == 33]['Count'].iloc[0]) / 100
    felszin['Vizenyős'] = (df[df['Value'] == 41]['Count'].iloc[0] +
                           df[df['Value'] == 43]['Count'].iloc[0]) / 100
    felszin['Gyep'] = (df[df['Value'] == 35]['Count'].iloc[0] +
                       df[df['Value'] == 36]['Count'].iloc[0]
                       + df[df['Value'] == 26]['Count'].iloc[0]) / 100
    felszin['Bokor'] = df[df['Value'] == 30]['Count'].iloc[0] / 100
    felszin['Gyümölcs'] = df[df['Value'] == 27]['Count'].iloc[0] / 100
    felszin['Szőlő'] = df[df['Value'] == 28]['Count'].iloc[0] / 100
    felszin['Kiskert'] = df[df['Value'] == 29]['Count'].iloc[0] / 100
    felszin['Lucerna'] = df[df['Value'] == 24]['Count'].iloc[0] / 100
    felszin['Épület'] = (df[df['Value'] == 111]['Count'].iloc[0] +
                         df[df['Value'] == 112]['Count'].iloc[0]
                         + df[df['Value'] == 12]['Count'].iloc[0]) / 100
    felszin['Út'] = (df[df['Value'] == 13]['Count'].iloc[0] +
                     df[df['Value'] == 20]['Count'].iloc[0]) / 100


with rasterio.open(r"G:\Geodata\nagyszekely_felszin_2022.tif") as file:
    rst = file.read(1)

nsz_felszin = rst.reshape(-1)

value_counts_series = pd.Series(nsz_felszin).value_counts()
felszin_df = pd.DataFrame({'Value': value_counts_series.index, 'Count': value_counts_series.values})
felszin_df = felszin_df.drop(index=0)

nsz_felsz = {}
nsz_felsz['Szántó'] = (felszin_df[felszin_df['Value']==21]['Count'].iloc[0] + felszin_df[felszin_df['Value']==22]['Count'].iloc[0]
                       + felszin_df[felszin_df['Value']==25]['Count'].iloc[0])/100
nsz_felsz['Erdő'] = (felszin_df[felszin_df['Value']==31]['Count'].iloc[0] + felszin_df[felszin_df['Value']==33]['Count'].iloc[0])/100
nsz_felsz['Vizenyős'] = (felszin_df[felszin_df['Value']==41]['Count'].iloc[0] + felszin_df[felszin_df['Value']==43]['Count'].iloc[0])/100
nsz_felsz['Gyep'] = (felszin_df[felszin_df['Value']==35]['Count'].iloc[0] + felszin_df[felszin_df['Value']==36]['Count'].iloc[0]
                     + felszin_df[felszin_df['Value'] == 26]['Count'].iloc[0])/100
nsz_felsz['Bokor'] = felszin_df[felszin_df['Value']==30]['Count'].iloc[0]/100
nsz_felsz['Gyümölcs'] = felszin_df[felszin_df['Value']==27]['Count'].iloc[0]/100
nsz_felsz['Szőlő'] = felszin_df[felszin_df['Value']==28]['Count'].iloc[0]/100
nsz_felsz['Kiskert'] = felszin_df[felszin_df['Value']==29]['Count'].iloc[0]/100
nsz_felsz['Lucerna'] = felszin_df[felszin_df['Value']==24]['Count'].iloc[0]/100
nsz_felsz['Épület'] = (felszin_df[felszin_df['Value']==111]['Count'].iloc[0] + felszin_df[felszin_df['Value']==112]['Count'].iloc[0]
                       + felszin_df[felszin_df['Value']==12]['Count'].iloc[0])/100
nsz_felsz['Út'] = (felszin_df[felszin_df['Value']==13]['Count'].iloc[0] + felszin_df[felszin_df['Value']==20]['Count'].iloc[0])/100

nsz_felsz= dict(sorted(nsz_felsz.items(), key=lambda x: x[1], reverse=True))
import matplotlib.pyplot as plt

colors = ['forestgreen', 'khaki', 'darkkhaki', 'yellowgreen', 'orange', 'mediumorchid', 'grey', 'cornflowerblue', 'darkred', 'plum', 'indigo']

fig, ax = plt.subplots()
bar_container = ax.bar(nsz_felsz.keys(), nsz_felsz.values(), color=['forestgreen', 'khaki', 'darkkhaki', 'yellowgreen', 'orange', 'mediumorchid', 'grey', 'cornflowerblue', 'darkred', 'plum', 'indigo'])
ax.set(xlabel='Felszínborítás', ylabel='Terület (hektár)')
ax.tick_params(axis='both', which='major', labelsize=14)
ax.bar_label(bar_container, fmt='{:,.0f}')
plt.gca().set_xlabel('Felszínborítás', fontsize=24)
plt.gca().set_ylabel('Terület (hektár)', fontsize=24)
plt.show()

fig, ax = plt.subplots(figsize=(6, 6))

patches, texts, pcts = ax.pie(
    nsz_felsz.values(), labels=nsz_felsz.keys(), autopct='%.1f%%',
    wedgeprops={'linewidth': 3.0, 'edgecolor': 'white'},
    textprops={'size': 'x-large'},
    startangle=90)
# For each wedge, set the corresponding text label color to the wedge's
# face color.
for i, patch in enumerate(patches):
  texts[i].set_color(patch.get_facecolor())
plt.setp(pcts, color='white')
plt.setp(texts, fontweight=600)
ax.set_title('Nagyszékely felszínborítás', fontsize=14)

plt.tight_layout()
