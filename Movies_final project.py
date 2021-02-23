#!/usr/bin/env python
# coding: utf-8

# In[492]:


import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import collections
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import metrics
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score


# In[230]:


tmdb_movies = pd.read_csv(r'C:\Users\preet\Desktop\Movies\tmdb_movies_data.csv')


# ## Replacing 0's in budget with corresponding revenue and vice-versa

# In[231]:


tmdb_movies[(tmdb_movies['budget']==0)].shape[0]


# In[232]:


tmdb_movies[(tmdb_movies['revenue']==0)].shape[0]


# In[233]:


tmdb_movies.budget.replace(0,tmdb_movies.revenue,inplace=True)


# In[234]:


tmdb_test.revenue.replace(0,tmdb_test.budget,inplace=True)


# In[235]:


cast_actress = pd.read_csv(r'C:\Users\preet\Desktop\Movies\The_Best_Actresses_Ever.csv',encoding='latin-1')
cast_actor=pd.read_csv(r'C:\Users\preet\Desktop\Movies\The_Most_Popular_Actors.csv',encoding='latin-1')


# In[236]:


cast=pd.concat([cast_actress, cast_actor], axis=0)


# In[237]:


cast['Description']=cast['Description'].map(lambda x:x.rstrip('points'))


# In[238]:


cast['Description']=cast['Description'].astype(float)


# In[239]:


cast_new=cast[['Name','Description']]


# In[240]:


director=pd.read_csv(r'C:\Users\preet\Desktop\Movies\My Complete Directors List.csv',encoding='latin-1')


# In[241]:


director


# In[242]:


director[director['Description']=='729 üpom']


# In[243]:


director=director.drop(director.index[507])
director['Description']=director['Description'].map(lambda x:x.rstrip('point'))
director['Description']=director['Description'].astype(float)


# In[244]:


director_new=director[['Description','Name']]
director_new=director_new.rename(columns={"Name":"director"})


# In[245]:


production=pd.read_csv(r'C:\Users\preet\Desktop\Movies\production_companies_data.csv')


# In[246]:


production['Total Domestic Box Office']=production['Total Domestic Box Office'].map(lambda x:x.lstrip('$'))
production['Total Domestic Box Office']=production['Total Domestic Box Office'].replace('\,','')
production['Total Domestic Box Office']=production['Total Domestic Box Office'].apply(lambda x: x.replace(',',''))
production['Total Worldwide Box Office']=production['Total Worldwide Box Office'].map(lambda x:x.lstrip('$'))
production['Total Worldwide Box Office']=production['Total Worldwide Box Office'].apply(lambda x: x.replace(',',''))


# In[247]:


from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
scaled_array = min_max_scaler.fit_transform(production[['No. of Movies','Total Domestic Box Office','Total Worldwide Box Office']])
df_normalized = pd.DataFrame(scaled_array)
df_normalized


# In[248]:


production['Total Domestic Box Office']=production['Total Domestic Box Office'].astype('float')
production['Total Worldwide Box Office']=production['Total Worldwide Box Office'].astype('float')


# In[249]:


#production['Score']=production['No. of Movies']*(0.01)+production['Total Domestic Box Office']*(0.005)+production['Total Worldwide Box Office']*(0.008)
production['Score']=production['Total Worldwide Box Office']/(100000000) + production['No. of Movies']


# In[250]:


production


# In[251]:


production=production.rename(columns={"Production Companies":"Name"})


# In[252]:


mydict_prod = dict(zip(production.Name, production.Score))
mydict_cast = dict(zip(cast_new.Name, cast_new.Description))

tmdb_movies['cast'] = tmdb_movies['cast'].astype('str')


# In[253]:


l = []
for index, row in tmdb_movies.iterrows():

  s = row['cast']
  x= s.split('|')
  sum =0
  for i in x:
    if i in mydict_cast:
      sum = sum + mydict_cast.get(i)
    else:
      sum = sum + 0
  l.append(sum)
  
tmdb_movies['cast_points'] = pd.Series(l)


# In[254]:


tmdb_movies['production_points'] = 0

tmdb_movies['production_companies'] = tmdb_movies['production_companies'].astype('str')

l = []
for index, row in tmdb_movies.iterrows():

  s = row['production_companies']
  x=s.split('|')
  sum =0
  for i in x:
    if i in mydict_prod:
      sum = sum + mydict_prod.get(i)
    else:
      sum = sum + 0
  l.append(sum)
  
tmdb_movies['production_points'] = pd.Series(l)


# In[255]:


tmdb_movies


# In[256]:


new_tmdb=pd.merge(tmdb_movies,director_new,on='director',how='left')


# In[257]:


new_tmdb = new_tmdb.rename(columns = {"Description":"Director_points"})


# ## Genres
# 

# In[258]:


def get_decade_by_year(year):
    decade = year - (year%10)
    decade_plus_10 = decade +10
    decade_as_str = str(decade)+'-'+ str(decade_plus_10)
    return decade_as_str


# In[259]:


df_copy = new_tmdb.copy()
year_of_release = []
decade_of_release = []
month_of_release = []
for x in range(len(df_copy)):
    year = int(df_copy.iloc[x]['release_date'][-4:])
    month = int(df_copy.iloc[x]['release_date'].split('/')[0])
    year_of_release.append(year) 
    decade_of_release.append(get_decade_by_year(year))
    month_of_release.append(month)
df_copy['year_of_release'] = year_of_release
df_copy['decade_of_release'] = decade_of_release
df_copy['month_of_release'] = month_of_release


# In[260]:


df_copy_for_genres = df_copy.copy()


# In[261]:


import collections
genre_by_decade = collections.OrderedDict()
for i in range(len(df_copy_for_genres)):
    decade = df_copy_for_genres.iloc[i]['decade_of_release']
    if decade not in genre_by_decade:
        genre_by_decade[decade] = {}
    if type(df_copy_for_genres.iloc[i]['genres'])==str:
        genres_as_list = df_copy_for_genres.iloc[i]['genres'].split('|')
        for each_genre_in_movie in genres_as_list:
            if each_genre_in_movie not in genre_by_decade[decade]:
                genre_by_decade[decade][each_genre_in_movie] =1
            else:
                genre_by_decade[decade][each_genre_in_movie]+=1


# In[488]:


df = pd.DataFrame(genre_by_decade)
cols= sorted(list(df.columns))
df[cols]


# In[499]:


df_with_scores = df.copy()
columns = list(df.columns)
for column in columns:
    df_with_scores[column] = 100*df_with_scores[column] /  df_with_scores[column].sum()
cols= sorted(list(df.columns))
df_with_scores[cols]


# In[264]:


df_with_scores_add_new_column = df_with_scores.copy()


# In[265]:


tmdb_with_genre_score = df_copy.copy()
genre_score = []
current_genre_score = []

for i in range(len(tmdb_with_genre_score)):
    if type(tmdb_with_genre_score.iloc[i]['genres'])==str:
        each_movie_genre_score =0
        each_movie_current_genre_score = 0
        
        decade_of_release = tmdb_with_genre_score.iloc[i]['decade_of_release']
        genres_as_list = df_copy_for_genres.iloc[i]['genres'].split('|')
        for each_genre_in_movie in genres_as_list:
            each_movie_genre_score+=df_with_scores_add_new_column[decade_of_release][each_genre_in_movie]
            each_movie_current_genre_score+=df_with_scores_add_new_column['2010-2020'][each_genre_in_movie]
    else:
        each_movie_genre_score = 0
        each_movie_current_genre_score = 0
        
    genre_score.append(each_movie_genre_score)
    current_genre_score.append(each_movie_current_genre_score)
    
tmdb_with_genre_score['genre_score_atRelease'] = genre_score
tmdb_with_genre_score['genre_score_atPresent'] = current_genre_score


# In[266]:


tmdb_with_genre_score.describe()


# In[267]:


oscars = pd.read_csv(r'C:\Users\preet\Desktop\Movies\oscars.csv')


# In[268]:


oscars


# In[269]:


oscars_new = oscars[['original_title','Winner']]
#director_new=director[['Description','Name']]


# In[270]:


oscars_new.fillna(0)


# In[271]:


oscar_sum = oscars_new.groupby(['original_title']).agg({'Winner': 'sum'})


# In[272]:


oscar_sum


# In[273]:


tmdb_with_oscar=pd.merge(tmdb_with_genre_score,oscar_sum,on='original_title',how='left')


# In[274]:


tmdb_with_oscar


# In[275]:


tmdb_with_oscar=tmdb_with_oscar.rename(columns={"Winner":"oscars_won"})


# In[276]:


tmdb_with_oscar.isna().sum()


# ## Inflation

# In[277]:


inflation = pd.read_csv(r'C:\Users\preet\Desktop\Movies\infla.csv')
inflation['DATE'] = pd.to_datetime(inflation['DATE'])
inflation["year"] = inflation["DATE"].dt.year
inflation["month"] = inflation["DATE"].dt.month


# In[278]:


inflated_budget =[]
inflated_revenue = []
inflated_profit = []

for i in range(len(tmdb_with_oscar)):
    cpi = inflation.loc[inflation.year == tmdb_with_oscar.iloc[i].year_of_release][inflation.month == tmdb_with_oscar.iloc[i].month_of_release]['CPIAUCNS_NBD19130101'].item()
    inflated_budget.append(tmdb_with_oscar.iloc[i].budget/cpi)
    inflated_revenue.append(tmdb_with_oscar.iloc[i].revenue/cpi)

tmdb_with_oscar['inflated_budget'] = inflated_budget
tmdb_with_oscar['inflated_revenue'] = inflated_revenue
tmdb_with_oscar['inflated_profit'] = tmdb_with_oscar['inflated_revenue'] - tmdb_with_oscar['inflated_budget']


# In[280]:


tmdb_with_oscar


# # Duration vs Decade

# In[281]:


def get_category_by_length(duration):
    if duration>=0 and duration <90:
        return "Less (dur)"
    elif duration >=90 and duration<120:
        return "Med (dur)"
    else:
        return "Long (dur)"


# In[282]:


duration_by_decade = collections.OrderedDict()
for i in range(len(tmdb_with_oscar)):
    decade = tmdb_with_oscar.iloc[i]['decade_of_release']
    if decade not in duration_by_decade:
        duration_by_decade[decade] = {}
    movie_duration_category = get_category_by_length(tmdb_with_oscar.iloc[i]['runtime'])
    if movie_duration_category not in duration_by_decade[decade]:
        duration_by_decade[decade][movie_duration_category] =1
    else:
        duration_by_decade[decade][movie_duration_category]+=1


# In[283]:


df = pd.DataFrame(duration_by_decade)
cols= sorted(list(df.columns))
df[cols]


# In[285]:


df_with_dur_scores = df.copy()
columns = list(df.columns)
for column in columns:
    df_with_dur_scores[column] = 100*df_with_dur_scores[column] /  df_with_dur_scores[column].sum()
cols= sorted(list(df.columns))
df_with_dur_scores[cols]


# In[286]:


df_with_dur_scores


# In[287]:


dur_score = []
for i in range(len(tmdb_with_oscar)):
    decade_of_release = tmdb_with_oscar.iloc[i]['decade_of_release']
    d_score = df_with_dur_scores[decade_of_release][get_category_by_length(tmdb_with_oscar.iloc[i]['runtime'])]
    dur_score.append(d_score)
tmdb_with_oscar['dur_score'] = dur_score


# In[288]:


tmdb_with_oscar.dtypes


# In[289]:


tmdb_with_oscar.head(5)


# In[290]:


tmdb_with_oscar.dtypes


# In[291]:


final_tmdb = tmdb_with_oscar.copy()
 


# In[292]:


final_tmdb = final_tmdb.drop(['imdb_id','homepage','tagline','keywords','overview'],axis=1)


# In[293]:


final_tmdb


# In[207]:


final_tmdb['oscars_won'].fillna(-1, inplace=True)


# In[208]:


final_tmdb


# In[209]:


final_tmdb.isna().sum()


# ##  filling nan with -1 in oscars_won
# 

# In[296]:


final_tmdb.oscars_won.fillna(-1, inplace=True)


# In[299]:


final_tmdb.isna().sum()


# ### replacing NAN in director's points

# In[300]:


final_tmdb.describe()


# ### replacing NAN with 25th percentile score 

# In[301]:


final_tmdb.Director_points.fillna(284, inplace=True)


# In[304]:


final_tmdb.isna().sum()


# ## current popularity
# 

# In[454]:


min_max_scaler = preprocessing.MinMaxScaler()
scaled_array = min_max_scaler.fit_transform(final_tmdb[['vote_average','cast_points','production_points','Director_points','inflated_profit','genre_score_atPresent']])
df_normalized_Pc = pd.DataFrame(scaled_array)
df_normalized_Pc


# In[464]:


final_tmdb['Current_popularity'] = (0.5)*df_normalized_Pc[0] + (0.1)*df_normalized_Pc[1] + (0.1)*df_normalized_Pc[2] + (0.1)*df_normalized_Pc[3] +(0.1)*df_normalized_Pc[4] + (0.1)*df_normalized_Pc[5]


# In[465]:


final_tmdb


# ## Release popularity

# In[466]:


min_max_scaler = preprocessing.MinMaxScaler()
scaled_array = min_max_scaler.fit_transform(final_tmdb[['genre_score_atRelease','dur_score','inflated_budget']])
df_normalized_Rp = pd.DataFrame(scaled_array)
df_normalized_Rp


# In[367]:





# In[467]:


final_tmdb['Release popularity'] = (0.4)*df_normalized_Rp[0] + (0.3)*df_normalized_Rp[1] + (0.3)*df_normalized_Rp[2]


# In[468]:


final_tmdb.describe()


# In[469]:


final_tmdb.columns


# In[470]:


final_tmdb['endurance'] = (0.6 * (final_tmdb['Current_popularity'] - final_tmdb['Release popularity']) 
                        + 0.4 * (final_tmdb['Current_popularity'] + final_tmdb['Release popularity']))/ 2


# In[471]:


final_tmdb.nlargest(10,'endurance')


# In[522]:


plt.plot(final_tmdb['original_title'].head(n=10),final_tmdb['Current_popularity'].head(n=10))


# In[472]:


Linear_RegModel = final_tmdb.filter(['popularity', 'cast_points', 'production_points', 'Director_points','year_of_release',
       'genre_score_atRelease', 'genre_score_atPresent', 'oscars_won','inflated_budget', 'inflated_revenue', 'inflated_profit', 'dur_score',
       'Current_popularity', 'Release popularity'])
Linear_RegModel.isna().sum()
Y = final_tmdb.filter(['endurance'])
Y
le = LabelEncoder()
Linear_RegModel = Linear_RegModel.apply(le.fit_transform)
#feature,result = linear_reg.loc[:linear_reg.columns != 'popularity'], linear_reg['popularity']
X_train, X_test, y_train, y_test = train_test_split(Linear_RegModel,Y, test_size=0.2)
lm = LinearRegression().fit(X_train,y_train) 
y_pred = lm.predict(X_test)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:'+str(rmse))


# In[478]:


Ridge_RegModel = final_tmdb.filter(['popularity', 'cast_points', 'production_points', 'Director_points','year_of_release',
       'genre_score_atRelease', 'genre_score_atPresent', 'oscars_won','inflated_budget', 'inflated_revenue', 'inflated_profit', 'dur_score',
       'Current_popularity', 'Release popularity'])
Ridge_RegModel.isna().sum()
Y = final_tmdb.filter(['endurance'])
Y
le = LabelEncoder()
Ridge_RegModel = Ridge_RegModel.apply(le.fit_transform)
#feature,result = linear_reg.loc[:linear_reg.columns != 'popularity'], linear_reg['popularity']
X_train, X_test, y_train, y_test = train_test_split(Ridge_RegModel,Y, test_size=0.2)
lm = Ridge(alpha = 10).fit(X_train,y_train) 
y_pred = lm.predict(X_test)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:'+str(rmse))


# In[481]:


Lasso_RegModel = final_tmdb.filter(['popularity', 'cast_points', 'production_points', 'Director_points','year_of_release',
       'genre_score_atRelease', 'genre_score_atPresent', 'oscars_won','inflated_budget', 'inflated_revenue', 'inflated_profit', 'dur_score',
       'Current_popularity', 'Release popularity'])
Lasso_RegModel.isna().sum()
Y = final_tmdb.filter(['endurance'])
Y
le = LabelEncoder()
Lasso_RegModel = Lasso_RegModel.apply(le.fit_transform)
#feature,result = linear_reg.loc[:linear_reg.columns != 'popularity'], linear_reg['popularity']
X_train, X_test, y_train, y_test = train_test_split(Lasso_RegModel,Y, test_size=0.2)
lm = Lasso(alpha = 0.1).fit(X_train,y_train) 
y_pred = lm.predict(X_test)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:'+str(rmse))


# In[475]:


forest_model = RandomForestRegressor(
    n_estimators=400, max_features=0.4,
    min_samples_leaf=30, n_jobs=-1)
 
X = final_tmdb.filter(['popularity', 'cast_points', 'production_points', 'Director_points','year_of_release',
       'genre_score_atRelease', 'genre_score_atPresent', 'oscars_won','inflated_budget', 'inflated_revenue', 'inflated_profit', 'dur_score',
       'Current_popularity', 'Release popularity', 'endurance'])
X.isna().sum()
Y = final_tmdb.filter(['endurance'])
X = X.apply(le.fit_transform)

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2)
lm = Lasso(alpha = 100).fit(X_train,y_train) 
y_pred = lm.predict(X_test)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:'+str(rmse))


# In[390]:


import seaborn as sns


# In[391]:


final_tmdb.dtypes


# In[392]:


train_1 = final_tmdb.filter(['endurance','genre_score_atPresent','genre_score_atRelease','Current_popularity','Release popularity','vote_average','dur_score','inflated_profit','oscars_won','Director_points','production_points','cast_points']) 


# In[393]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[394]:


plt.figure(figsize=(11,11))
sns.heatmap(train_1.corr(),vmin=-1,vmax=1,cmap='coolwarm',linewidths=0.2,annot=True);


# In[403]:


final_tmdb.dtypes


# ## Oscars effect on endurance

# In[408]:


df_winning = final_tmdb.loc[(final_tmdb['oscars_won'] > 0)]


# In[413]:


df_nominated = final_tmdb.loc[(final_tmdb['oscars_won'] == 0)]


# In[426]:


final_tmdb['endurance'].max()


# In[427]:


final_tmdb['endurance'].mean()


# In[420]:


df_winning['endurance'].mean()


# In[421]:


df_nominated['endurance'].mean()


# ## Genres effect on movie's popularity over time

# In[430]:


western_past = final_tmdb.loc[(final_tmdb['decade_of_release'] == '1960-1970')]


# In[513]:


end =0
count =0
for index, row in western_past.iterrows():
    s = row['genres']
    l = s.split('|')
    #print(l)
    if 'Western' in l or 'War' in l:
        count+=1
        end = end + row['Current_popularity']
print(end)
print(count)


# In[450]:


western_current = final_tmdb.loc[(final_tmdb['decade_of_release'] == '2010-2020')]


# In[453]:


western_current


# In[516]:


end =0
count =0
for index, row in western_current.iterrows():
    s = str(row['genres'])
    l = s.split('|')
    #print(l)
    if 'Western' in l or 'War' in l:
        count+=1
        end = end + row['Current_popularity']
print(end)
print(count)


# In[524]:


final_tmdb.isna().sum()


# In[540]:


tmdb_high_endurance = final_tmdb.loc[(final_tmdb['endurance'] > 0.27)]


# In[542]:



plt.figure(figsize=(14, 6))
tmdb_high_endurance.groupby(['director']).mean()['endurance'].plot(kind="bar", title='director vs endurance',color='orange')


# In[569]:


import math
sorted_final = final_tmdb.copy()
sorted_final = sorted_final.sort_values('Current_popularity',ascending=False)
result = sorted_final.head(math.ceil(0.023*10900))

result


# In[570]:


imdb_top_250 = pd.read_csv(r'C:\Users\preet\Desktop\Movies\imdb-top-250.csv')


# In[571]:


imdb_top_250=imdb_top_250.dropna()
imdb_top_250


# In[572]:


imdb_top_250=pd.merge(imdb_top_250,result,on='original_title')


# In[573]:


imdb_top_250


# In[580]:


plt.figure(figsize=(14, 6))
sns.lineplot(result['Current_popularity'],result['endurance']).set_title('current popularity vs endurance')


# In[582]:


plt.figure(figsize=(14, 6))
plt.style.use('ggplot')
sns.lineplot(result['Release popularity'],result['endurance']).set_title('Release popularity vs endurance')

