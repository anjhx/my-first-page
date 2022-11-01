import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pylab import *
import plotly.express as px
import plotly.graph_objects as go

#from pyecharts.charts import Pie
#import streamlit_echarts as ste

plt.style.use('seaborn')
df = pd.read_csv('AppleStore.csv',index_col=0)


st.header('Static analysis of moblie app store by Xinyu Hu and Linjie Zhou')


# A VIEW OF raw data
st.subheader('Exploratory Data Analysis')





# add a slider
price_filter = st.sidebar.slider('Mobile App Price', 0 , 5, 2)
df = df[df.price <= price_filter]

st.write('Before we start analysis, lets have a quick look at the data filtered by price :blush:')
st.write(df)

df.isnull().sum()
df.groupby('prime_genre').sum()
genres = df['prime_genre'].unique()



# show the pie chart


st.write('\n')
st.write('\n')
st.write('\n')
st.subheader('First let us show the distrubution of free and paid apps.')



fig, ax = plt.subplots()
freeapps = df[df.price == 0.0]
paidapps = df[df.price != 0.0]
data = np.array([56.4,43.6])
pie_labels = np.array(['Free','Paid'])
# 绘制饼图
plt.pie(data,radius=0.5,labels=pie_labels,autopct='%3.1f%%')

st.pyplot(fig)


# Return the numbers of free app in each genres
def genreFree(gen):
    return len(df[(df['price'] == 0.0) & (df['prime_genre']== gen)])
# Return the numbers of paid app in each genres
def genrePaid(gen):
    return len(df[(df['price'] != 0.0) & (df['prime_genre']== gen)])
def genreFreeRating(gen):
    a = df[(df['price'] == 0.0) & (df['prime_genre']== gen)]
    b = a.user_rating.mean()
    return b
def genrePaidRating(gen):
    a = df[(df['price'] != 0.0) & (df['prime_genre']== gen)]
    b = a.user_rating.mean()
    return b

def genreRating(gen):
    a = df[df.prime_genre==gen]
    b = a.user_rating.mean()
    return b
def pricestotal (gen):
    dff = df[df.prime_genre==gen]
   
    a = dff.price.sum()
    return a
def pricesmean (gen):
    dff = df[df.prime_genre==gen]

    a = dff.price.mean()
    return a


# Make list of each genre , its free app, paid app and total app . then merge it into one dataframe
genre_list = []
genreFree_list = []
genrePaid_list = []
genreTotal_list = []
genreFree_rating_list = []
genrePaid_rating_list =[]
genre_rating_list = []

pricestotal_list = []
pricesmean_list = []

# append all details in respective list
for gen in genres:  
    free_gen = genreFree(gen)
    paid_gen = genrePaid(gen)
    totalapp_gen = free_gen + paid_gen

    genreFree_rating = genreFreeRating(gen)
    genrePaid_rating = genrePaidRating(gen)
    genre_rating = genreRating(gen)

    prices_total = pricestotal(gen)
    prices_mean = pricesmean(gen)

    genre_list.append(gen)
    genreFree_list.append(free_gen)
    genrePaid_list.append(paid_gen)
    genreTotal_list.append(totalapp_gen)
    genreFree_rating_list.append(genreFree_rating)
    genrePaid_rating_list.append(genrePaid_rating)
    genre_rating_list.append(genre_rating)

    pricestotal_list.append(prices_total)
    pricesmean_list.append(prices_mean)

# Let's make a dataframe of it
from codecs import ignore_errors


genre_df = pd.DataFrame({
    "genre_name" : genre_list,
    "genre_freeApp" : genreFree_list,
    "genre_paidApp" : genrePaid_list,
    "genre_totalApp" : genreTotal_list,
    "genre_free_App_rating": genreFree_rating_list,
    "genre_paid_App_rating": genrePaid_rating_list,
    "genre_rating":genre_rating_list,
    "pricestotal":pricestotal_list,
    "pricesmean" :pricesmean_list
},columns=['genre_name','genre_freeApp','genre_paidApp','genre_totalApp','genre_free_App_rating','genre_paid_App_rating','pricestotal','pricesmean','genre_rating'])

#sorting into descending order
app_amounts = genre_df.sort_values('genre_totalApp', ascending=False)
appfree_ratings = genre_df.sort_values('genre_free_App_rating',ascending = False, ignore_index=True)
apppaid_ratings = genre_df.sort_values('genre_paid_App_rating',ascending = False, ignore_index= True)
apppaid_ratings = genre_df.sort_values('genre_paid_App_rating',ascending = False, ignore_index= True)



# remove duplicate genre 
app_amounts.drop_duplicates('genre_name',inplace=True)
appfree_ratings.drop_duplicates('genre_name',inplace=True)
apppaid_ratings.drop_duplicates('genre_name',inplace=True)
appfree_ratings = appfree_ratings[['genre_name','genre_free_App_rating']]





x = appfree_ratings['genre_name']
y1 = appfree_ratings['genre_free_App_rating']
y2 = apppaid_ratings['genre_paid_App_rating']






#plt.bar(range(23), y, color = 'thistle'),折线图1
fig, ax = plt.subplots()
plt.plot(range(23), y1, label= 'Users rating of free apps',marker = '*', color = 'coral') #coral
plt.plot(range(23), y2, label= 'Users rating of paid apps',marker = '.', color = 'violet')
plt.xticks(range(23), x,rotation=80,color = 'darkblue')
plt.xlabel('Genre',color = 'darkblue')
plt.ylabel("Users rating",color = 'darkblue')
plt.legend()
st.pyplot(fig)




# bubble 折线图2
def Bubble():
    fig,ax = subplots()
    a = genre_df.sort_values(by = 'genre_rating', ascending = True)
    fig = px.scatter(
    a,
    x="genre_name",
    y="genre_rating",
    hover_data=["genre_name","pricesmean"],   # 列表形式
    color="pricesmean",
    size="genre_totalApp",
    size_max=60,

    )

    fig.show()

    st.plotly_chart(fig)


# 折线图2
fig,ax = plt.subplots()
a = genre_df.sort_values(by = 'pricesmean', ascending = True)
x = a['genre_name']
y = a['pricesmean']
#plt.bar(range(23), y, color = 'thistle')
plt.plot(range(23), y, label= 'Average prices of different genres',marker = '*', color = 'purple') #coral

plt.xticks(range(23), x,rotation=80,color = 'darkblue')
plt.xlabel('Genre',color = 'darkblue')
plt.ylabel("Average prices",color = 'green')
plt.legend()
st.pyplot(fig)





popular_apps = df.sort_values(['user_rating','rating_count_tot'], ascending=False)
popular_apps.head() 
## top 前20的popular app:() rating

fig = plt.figure(figsize = (20, 8))                               
plt.bar(popular_apps['track_name'][0:20], (popular_apps['rating_count_tot']/popular_apps['user_rating'])[0:20]) 
plt.xticks(rotation=45,ha='right')  
st.pyplot(fig)



# All higher rating applications 
ratingapp = popular_apps[(popular_apps['user_rating'] == 4.0) | (popular_apps['user_rating'] == 5.0) | (popular_apps['user_rating']==4.5)]
ratingapp.head(5)


def dountChart(gen,title):  
    # Create a circle for the center of the plot
    circle=plt.Circle( (0,0), 0.7, color='white')
    
    # just keep on user rating as name not overlapping while pie chart plotting
    plt.pie(ratingapp['user_rating'][ratingapp['prime_genre']==gen][0:10], labels= ratingapp['track_name'][ratingapp['prime_genre']==gen][0:10])
    p=plt.gcf() #gcf = get current figure
    p.gca().add_artist(circle)
    plt.title(title , fontname="arial black")
    gens = ['Games','Shopping','Social Networking','Music','Food & Drink', 'Photo & Video','Sports','Finance']

# 函数
gens = ['Games','Shopping','Social Networking','Music','Food & Drink', 'Photo & Video','Sports','Finance']



st.sidebar.write('Choose')
add_selectbox = st.sidebar.radio(
        "Genres",
        ("Games", "Music", "Shopping","Photo & Video")
    )
if add_selectbox=="Games":
    fig = plt.figure(figsize=(25,30))
    plt.subplot(421)
    dountChart(gens[0],'Top Higher rating '+gens[0]+' apps')  
    st.pyplot(fig)
elif add_selectbox=="Music": 
    fig = plt.figure(figsize=(30,45))
    plt.subplot(421)
    dountChart(gens[3],'Top Higher rating '+gens[3]+' apps')  
    st.pyplot(fig)
elif add_selectbox == "Shopping":
    fig = plt.figure(figsize=(25,30))
    plt.subplot(421)
    dountChart(gens[1],'Top Higher rating '+gens[1]+' apps')  
    st.pyplot(fig)

elif add_selectbox == "Photo & Video":    
    fig = plt.figure(figsize=(25,30))
    plt.subplot(421)
    dountChart(gens[5],'Top Higher rating '+gens[5]+' apps')  
    st.pyplot(fig)


# 补充表单
st.sidebar.button('A bubble chart', on_click=Bubble)









# rating change:
#  lmplot对所选择的数据集做出了一条最佳的拟合直线
fig,ax = plt.subplots(figsize=(10,5))
df['isNotFree'] = df['price'].apply(lambda x: 1 if x > 0 else 0)
sns.lmplot(x='user_rating', y='user_rating_ver',hue='isNotFree', data=df)
st.pyplot(fig)

fig,ax = plt.subplots(figsize=(10,5))
ratingapp['isNotFree'] = ratingapp['price'].apply(lambda x: 1 if x > 0 else 0)
sns.lmplot(x='user_rating', y='user_rating_ver',hue='isNotFree', data=ratingapp)
st.pyplot(fig)


fig,ax = plt.subplots()
plt.figure(figsize=(10,5))
plt.scatter(y=df.prime_genre ,x=df.rating_count_tot,c='DarkBlue')
plt.title('Rating & Category')
plt.xlabel('rating')
plt.ylabel('Category')
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(10,5))
free_apps = df[(df.price==0.00)]
paid_apps  = df[(df.price>0)]
sns.set_style('white')
sns.violinplot(x=paid_apps['user_rating'],color='#79FF79')
plt.xlim(0,5)
plt.xlabel('Rating (0 to 5 stars)')
_ = plt.title('Distribution of Paid Apps Ratings')
st.pyplot(fig)

fig,ax = plt.subplots(figsize=(10,5))
sns.set_style('white')
sns.violinplot(x=free_apps['user_rating'],color='#66B3FF')
plt.xlim(0,5)
plt.xlabel('Rating (0 to 5 stars)')
_ = plt.title('Distribution of free Apps Ratings')
st.pyplot(fig)

fig,ax = plt.subplots(figsize=(10,5))
bins = (0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5)
plt.style.use('seaborn-white')
plt.hist(paid_apps['user_rating'],alpha=.8,bins=bins,color='#79FF79')
plt.xticks((0,1,2,3,4,5))
plt.title('Paid Apps - User Ratings ')
plt.xlabel('Rating')
plt.ylabel('Frequency')
_ = plt.xlim(right=5.5)
st.pyplot(fig)