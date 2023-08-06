#!/usr/bin/env python
# coding: utf-8

# In[153]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import regex as re


# In[154]:


df = pd.read_csv("houseprice.csv")
df = df.drop("society",axis="columns")
nan_values = df[df['size'].isna()]
print(nan_values)
df


# In[155]:


df.isnull().sum()
mv = df["balcony"].median()
print(mv)
df.fillna(df.median(),inplace=True)
print(df.isnull().sum())


# In[156]:


df['size'] = df['size'].astype('str').str.extractall('(\d+)').unstack().fillna('').sum(axis=1).astype(int)
df


# In[157]:


df.total_sqft.unique()


# In[158]:


def isfloat(x):
    try:
        float(x)
    except:
        return False
    return True


    
    


# In[159]:


df[~df.total_sqft.apply(isfloat)]


# In[160]:


def change_total_sqft(x):
    nums = x.split("-")
    res = nums[0]
    try:
        if(len(nums) > 1):
            res = int(nums[0])/2+int(nums[1])/2
    except:
        pass
    res = re.findall('[0-9]+', str(res))[0]
    return res


# In[161]:


df.total_sqft = df.total_sqft.apply(change_total_sqft)
df
df.total_sqft.unique()


# In[162]:


# import matplotlib
# %matplotlib inline
# matplotlib.rcParams["figure.figsize"] = (15,10)
# plt.plot(df.total_sqft,df.price)


# In[163]:


df.location.unique()
len(df.location.unique())


# In[164]:


location_stats = df.groupby("location")["location"].agg("count").sort_values(ascending=False)
print(len(location_stats))
location_stats


# In[165]:


less_than_10_location = location_stats[location_stats <=10]


# In[166]:


df.location = df.location.apply(lambda x:"other" if x in less_than_10_location else x )
df


# In[167]:


#checking for outliers
df2 = df.copy()
df2["total_sqft"] = df2["total_sqft"].astype(float)

df2["price"] = df2["price"].astype(float)


df2["price_per_sqft"] = (df2["price"]*100000)/ df2["total_sqft"]
df2


# In[168]:


df2.shape


# In[169]:


df3 = df2[~(df2["total_sqft"]/df2["size"] < 300)]
df3.shape
df3


# In[170]:


df3.price_per_sqft.describe()


# In[171]:


def remove_price_outliers(df):
    df_out = pd.DataFrame()
    for key,subdf in df.groupby("location"):
        m = np.mean(subdf.price_per_sqft)
        sd = np.std(subdf.price_per_sqft)
        ndf = subdf[(subdf.price_per_sqft > (m-sd)) & (subdf.price_per_sqft < (m+sd))]
        df_out = pd.concat([df_out,ndf],ignore_index=True)
    return df_out

df4 = remove_price_outliers(df3)
df4


# In[ ]:





# In[172]:


def remove_bhk_outliers(df):
    
    exclude_indices = np.array([])
    for location,location_df in  df.groupby("location"):
        bhk_stats = {}
        for bhk,bhk_df in location_df.groupby("size"):
            bhk_stats[bhk] = {
                "mean" :np.mean(bhk_df.price_per_sqft),
                "sd":np.std(bhk_df.price_per_sqft),
                "count":bhk_df.shape[0]
                
            }
        
        for bhk,bhk_df in location_df.groupby("size"):
            stats = bhk_stats.get(bhk-1)
            if stats and stats["count"]>5:
                exclude_indices = np.append(exclude_indices,bhk_df[bhk_df.price_per_sqft < stats["mean"]].index.values)
        return df.drop(exclude_indices,axis="index")
df5 = remove_bhk_outliers(df4)
print(df5.shape)
df5[500:]
            
            
        
    


# In[173]:


plt.hist(df5.price_per_sqft,rwidth=0.8)


# In[174]:


#removing bathroom greater then bedroom+2 kind of unusual
df6 = df5[df5.size+2 > df5.bath]
df7 = df6.drop(["price_per_sqft"],axis=1)
df7


# In[175]:


df7.area_type.unique()

dummies = pd.get_dummies(df7.area_type)
dummies
df8 = pd.concat([df7,dummies],axis="columns")
df8.drop(["area_type","availability"],axis=1)

location_dummies = pd.get_dummies(df8.location)
df9 = pd.concat([df8,location_dummies],axis="columns")
df9.drop(["location"],axis=1)
df10 = df9.drop(["other","area_type","availability","location"],axis=1)

df10


# In[176]:


df11 = df10.dropna()
df11.isna().any()
x = df11["price"]
df11 = df11.drop(["price"],axis=1)
df11


# In[177]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(df11,x,test_size=0.1)


# In[178]:


print(len(x_train),len(y_train))


# In[179]:


from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(x_train,y_train)


# In[180]:


model.score(x_test,y_test)
print(x_test.iloc[1])
model.predict([x_test.iloc[1]])


# In[181]:


import pickle
with open("banglore_predicition_mode.pickle2","wb") as f:
    pickle.dump(model,f)
    


# In[182]:



# for i in df11.columns:
#     print(i)

import json
columns = {
    "data_columns" : [cols.lower() for cols in df11.columns]
}

with open("banglore_prediction_columns","w") as f:
    f.write(json.dumps(columns))


# In[183]:


print(x_test.head())


# In[184]:



def predict(size,total_sqft,bath,balcony,area_type,location):
    arr = np.zeros(len(df11.columns), dtype = float)
    loc = np.where(df11.columns==location)[0][0]
    d={
        "BuiltupArea":4,
        "CarpetArea":5,
        "PlotArea":6,
        "SuperbuiltupArea":7
    }
    print(loc)
    if loc>7:
        arr[loc] = 1
    arr[d[area_type]] = 1
    arr[0] = size
    arr[1] = total_sqft
    arr[2] = bath
    arr[3] = balcony
    
    pv = model.predict([arr])[0]
    return pv

print(predict(2,1200,3,1,"BuiltupArea","1st Block Jayanagar"))
    


# In[185]:


print("...version",pickle.format_version)


# In[ ]:




