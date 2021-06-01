"""
Created on Sat May 22 22:58:42 2021

@author: srikanthr and neeraj
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import random
from geopy.geocoders import Nominatim
from mpl_toolkits.basemap import Basemap

plt.rcParams.update({'font.size': 12})
fig_save = True
save_path = 'plots/'

# load CSV
all_df = pd.read_csv('../data/metadata.csv')
# print covid status
all_df['covid_status'].unique()

#########################################
# MALE, FEMALE, AGE plot
#########################################
age_labels = ['15-30', '30-45', '45-60', '60-80']
age_cnt_female = all_df[all_df['g']=='female']['a'].values
age_cnt_male = all_df[all_df['g']=='male']['a'].values


age_grouped_male = []
age_grouped_female = []
FS = 14
for i in age_labels:
    age_grouped_male.append(len(age_cnt_male[(age_cnt_male > (int(i.split('-')[0])-1)) & \
                                        (age_cnt_male < int(i.split('-')[1]))]))
    age_grouped_female.append(len(age_cnt_female[(age_cnt_female > (int(i.split('-')[0])-1)) & \
                                        (age_cnt_female < int(i.split('-')[1]))]))
print('% male')
print(len(age_cnt_male)/(len(age_cnt_male)+len(age_cnt_female))*100)
clr_1 = 'gray'
clr_2 = 'gray'
width = .2
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(np.arange(0,len(age_labels)),age_grouped_male, align='center',alpha=0.5,ecolor='black',capsize=5,color=clr_1,width=width,label='MALE')
ax.bar(np.arange(0,len(age_labels))+.3,age_grouped_female, align='center',alpha=1,ecolor='black',hatch='///',capsize=5,color=clr_2,width=width,label='FEMALE')
ax.legend(frameon=False,loc='upper right',fontsize=FS)
plt.ylabel('SIZE', fontsize=FS)
plt.xlabel('AGE GROUP', fontsize=FS)
plt.xticks(np.arange(0,len(age_labels))+.15, age_labels,rotation=0,fontsize=FS)
plt.yticks(fontsize=FS)
ax.grid(color='gray', linestyle='--', linewidth=1,alpha=.3)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)  

if fig_save:
    ax.figure.savefig(os.path.join(save_path,'JBHI_bar_gender.pdf'), bbox_inches='tight')
# plt.show()

print('Total population size:')

#########################################
# COVID, Non-COVID, Recovered Grouping
#########################################
print(len(age_cnt_female)+len(age_cnt_male))
covid_status_values = all_df[(all_df['g']=='female') | (all_df['g']=='male')]['covid_status'].values
covid_count_dict = {}
non_covid_count_dict = {}
recov_covid_count_dict = {}

cat_labels_to_pie_dict = {'positive_mild':'Positive-Mild','positive_asymp':'Positive-Asymptomatic','positive_moderate':'Positive-Moderate','no_resp_illness_exposed':'Exposed','resp_illness_not_identified':'Resp. Ail.','healthy':'Healthy',
                         'recovered_full':'recovered_full'}
for x in covid_status_values:
    if 'positive' in x:
        if cat_labels_to_pie_dict[x] in covid_count_dict:
            covid_count_dict[cat_labels_to_pie_dict[x]]+=1
        else:
            covid_count_dict[cat_labels_to_pie_dict[x]] = 1
    elif 'recovered_full' in x:
        if cat_labels_to_pie_dict[x] in recov_covid_count_dict:
            recov_covid_count_dict[cat_labels_to_pie_dict[x]]+=1
        else:
            recov_covid_count_dict[cat_labels_to_pie_dict[x]] = 1
    else:
        if cat_labels_to_pie_dict[x] in non_covid_count_dict:
            non_covid_count_dict[cat_labels_to_pie_dict[x]]+=1
        else:
            non_covid_count_dict[cat_labels_to_pie_dict[x]] = 1

print('Counts:')
print(covid_count_dict)
vals = []
for x in covid_count_dict.keys():
    vals.append(covid_count_dict[x])
i = 0
print('%')
for x in covid_count_dict.keys():
    print(x)
    print(vals[i]/sum(vals)*100)
    i = i+1
print('All:'+str(sum(vals))+' nos.')

    
print('Counts:')
print(non_covid_count_dict)
vals = []
for x in non_covid_count_dict.keys():
    vals.append(non_covid_count_dict[x])
i = 0
print('%')
for x in non_covid_count_dict.keys():
    print(x)
    print(vals[i]/sum(vals)*100)
    i = i+1
print(recov_covid_count_dict)
print('All:'+str(sum(vals))+' nos.')

#########################################
# COVID pie chart
#########################################

fig = plt.subplots(figsize=[4,4])
ax = plt.subplot(1,1,1)

labels = ['{} ({})'.format(k,v) for k,v in covid_count_dict.items()]
print(labels)
clr = ['darkturquoise','cornflowerblue', 'salmon']
ax.pie(covid_count_dict.values(), shadow=True, startangle=60, radius=10,\
       rotatelabels = True, labeldistance=1.1, textprops={'fontsize': 'xx-large','fontfamily':'monospace'},
      colors=clr)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
if fig_save:
    ax.figure.savefig(os.path.join(save_path,'JBHI_pie_chart_positives.pdf'), bbox_inches='tight')
# plt.show()

#########################################
# Non-COVID pie chart
#########################################
    
fig = plt.subplots(figsize=[4,4])
ax = plt.subplot(1,1,1)

labels = ['{} ({})'.format(k,v) for k,v in non_covid_count_dict.items()]
print(labels)
clr = ['cornflowerblue','darkturquoise','salmon','goldenrod']
ax.pie(non_covid_count_dict.values(), shadow=True, startangle=60, radius=10,colors=clr,\
       rotatelabels = True, labeldistance=1.1, textprops={'fontsize': 'xx-large','fontfamily':'monospace'})
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
if fig_save:
    ax.figure.savefig(os.path.join(save_path,'JBHI_pie_chart_negatives.pdf'), bbox_inches='tight')
# plt.show()

#########################################
# Recovered pie chart
#########################################
fig = plt.subplots(figsize=[4,4])
ax = plt.subplot(1,1,1)

labels = ['{} ({})'.format(k,v) for k,v in recov_covid_count_dict.items()]
print(labels)
clr = ['cornflowerblue','darkturquoise','salmon','goldenrod']
ax.pie(recov_covid_count_dict.values(), shadow=True, startangle=60, radius=10,colors=clr,\
       rotatelabels = True, labeldistance=1.1, textprops={'fontsize': 'xx-large','fontfamily':'monospace'})
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
if fig_save:
    ax.figure.savefig(os.path.join(save_path,'JBHI_pie_chart_recovered.pdf'), bbox_inches='tight')
# plt.show()

#########################################
# Get world map
#########################################

geolocator = Nominatim(user_agent="geoapiExercises")
def geolocate(country):
    try:
        # Geolocate the center of the country
        loc = geolocator.geocode(country)
        # And return latitude and longitude
        return (loc.latitude, loc.longitude)
    except:
        # Return missing value
        return np.nan

country_names = all_df['l_c'].unique()
df = {}
df['country'] = country_names
df['count'] = []
 
for x in country_names:
    temp = all_df[(all_df['g']=='female') | (all_df['g']=='male')]
    df['count'].append(len(temp[temp['l_c']==x]))
##
df = pd.DataFrame.from_dict(df)
df['lat'] = 0
df['long'] = 0
for i in range(len(df)):
    temp = geolocate(df['country'][i])
    df.loc[i,'lat'] = temp[0]
    df.loc[i,'long'] = temp[1]

#########################################
# plot world map distribution
#########################################
fig = plt.subplots(figsize=[9,7])
m = Basemap(llcrnrlon=-180, llcrnrlat=-65, urcrnrlon=180, urcrnrlat=80)
m.drawmapboundary(fill_color='#FFF', linewidth=0)
m.fillcontinents(color='grey', alpha=0.5)
m.drawcoastlines(linewidth=0.1, color="white")

# prepare a color for each point depending on the continent.
df['labels_enc'] = pd.factorize(df['country'])[0]
 
# Add a point per position
m.scatter(
    x=df['long'], 
    y=df['lat'], 
    s=np.log2(df['count'])*100, 
    alpha=0.7, 
    c=df['labels_enc']/100, 
    cmap="Set1"
)
if fig_save:
    plt.savefig(os.path.join(save_path,'JBHI_world_map.pdf'), bbox_inches='tight')
# plt.show()


#########################################
# plot India/Outside pie chart
#########################################
vals = []
vals.append(df[df['country']=='India']['count'][0])
vals.append(sum(df['count'])-vals[0])

print('India %')
print(vals[0]/sum(vals)*100)
print('Outside %')
print(vals[1]/sum(vals)*100)

fig = plt.subplots(figsize=[3,3])
ax = plt.subplot(1,1,1)
clr = ['pink','darkturquoise','cornflowerblue','goldenrod']
clr = ['cornflowerblue','darkturquoise','goldenrod']
ax.pie(vals, shadow=True, startangle=60, radius=10,\
       rotatelabels = True, labeldistance=1.1, textprops={'fontsize': 'xx-large','fontfamily':'monospace'},
      colors=clr)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
fmt = '.pdf'
if fig_save:
    ax.figure.savefig(os.path.join(save_path,'JBHI_pie_chart_country.pdf'), bbox_inches='tight')
# plt.show()

print('Ploting metadata complete!')
