import streamlit as st
import pandas as pd 
from matplotlib import pyplot as plt
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np 
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

st.markdown("""
<style>
    #MainMenu, header, footer {visibility: hidden;}
</style>
""",unsafe_allow_html=True)

st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous"><link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">', unsafe_allow_html=True)

st.markdown("""
<nav class="navbar fixed-top" style = "box-shadow: 0 .5rem 1.5rem rgba(0, 0, 0, .1); background-color: white;" >
  <div class="container-fluid" style = "margin-left:110px; margin-top:10px">
  <a href="https://heartify-ui.s3.amazonaws.com/iindex.html" class="logo" style="font-family: 'Poppins', sans-serif; font-size: 162.5%; color: black; text-decoration: none; color: #3c3c3c;"> <i class="fas fa-heartbeat" style = "color: #16a085;"></i> Heartify.</a>
  <a href = "https://heartify-ui.s3.amazonaws.com/index.html" style = "font-size: 1.2rem;color: #777; margin-right: 70px;"> Logout </a>
  </div>
</nav>
""", unsafe_allow_html=True)

st.title("Heartify Food Recommender")
# st.text("Let us help you decide what to eat")
st.image("Foods.png")

disease = st.radio(
     "Which disease do you have ?",
     ('None','Congenital Heart Disease', 'Coronary Artery Disease', 'Dilated Cardiomyopathy'))

st.subheader("Select your desired daily percentage for Macros")
if disease == 'Congenital Heart Disease':
    protein_val = st.slider("Select protein  value",0,20, value = 10)
    carb_val = st.slider("Select carbohydrate value",0,20, value = 5)
    fat_val = st.slider("Select fat value",0,20, value = 2)
elif disease == 'Coronary Artery Disease':
    protein_val = st.slider("Select protein  value",0,20, value = 7)
    carb_val = st.slider("Select carbohydrate value",0,20, value = 8)
    fat_val = st.slider("Select fat value",0,20, value = 4)
elif disease == 'Dilated Cardiomyopathy':
    protein_val = st.slider("Select protein  value",0,20, value = 9)
    carb_val = st.slider("Select carbohydrate value",0,20, value = 6)
    fat_val = st.slider("Select fat value",0,20, value = 3)
elif disease == 'None':
    protein_val = st.slider("Select protein  value",0,20)
    carb_val = st.slider("Select carbohydrate value",0,20)
    fat_val = st.slider("Select fat value",0,20)

food = pd.read_csv("../input/food.csv")
ratings = pd.read_csv("../input/ratings.csv")
combined = pd.merge(ratings, food, on='Food_ID')

def tag(combined):
    if disease == 'Congenital Heart Disease':
        ans = combined.loc[(combined['Congenital Heart Disease'] == 1) & (combined['Carbs DV%'] >= carb_val) & (combined['Proteins DV%'] >= protein_val) & (combined['Fats DV%'] >= fat_val),['Name','Carbs DV%', 'Fats DV%', 'Proteins DV%', 'Congenital Heart Disease', 'Coronary Artery Disease', 'Dilated Cardiomyopathy']]
    elif disease == 'Coronary Artery Disease':
        ans = combined.loc[(combined['Coronary Artery Disease'] == 1) & (combined['Carbs DV%'] >= carb_val) & (combined['Proteins DV%'] >= protein_val) & (combined['Fats DV%'] >= fat_val),['Name','Carbs DV%', 'Fats DV%', 'Proteins DV%', 'Congenital Heart Disease', 'Coronary Artery Disease', 'Dilated Cardiomyopathy']]
    elif disease == 'Dilated Cardiomyopathy':
        ans = combined.loc[(combined['Dilated Cardiomyopathy'] == 1) & (combined['Carbs DV%'] >= carb_val) & (combined['Proteins DV%'] >= protein_val) & (combined['Fats DV%'] >= fat_val),['Name','Carbs DV%', 'Fats DV%', 'Proteins DV%', 'Congenital Heart Disease', 'Coronary Artery Disease', 'Dilated Cardiomyopathy']]
    elif disease == 'None':
        ans = combined.loc[(combined['Carbs DV%'] >= carb_val) & (combined['Proteins DV%'] >= protein_val) & (combined['Fats DV%'] >= fat_val),['Name','Carbs DV%', 'Fats DV%', 'Proteins DV%', 'Congenital Heart Disease', 'Coronary Artery Disease', 'Dilated Cardiomyopathy']]
    return ans

ans = tag(combined)
names = ans['Name'].tolist()
x = np.array(names)
ans1 = np.unique(x)

if len(ans1) > 0:
    finallist = ""
    proceedVal = st.checkbox("Proceed ?")
    if proceedVal == True:
        finallist = st.selectbox("Select a dish you like",ans1)
else:
    st.write("No dishes found, please change the parameters")

##### IMPLEMENTING RECOMMENDER ######
dataset = ratings.pivot_table(index='Food_ID',columns='User_ID',values='Rating')
dataset.fillna(0,inplace=True)
csr_dataset = csr_matrix(dataset.values)
dataset.reset_index(inplace=True)

model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model.fit(csr_dataset)

def food_recommendation(Food_Name):
    n = 10
    FoodList = food[food['Name'].str.contains(Food_Name)]  
    if len(FoodList):        
        Foodi= FoodList.iloc[0]['Food_ID']
        Foodi = dataset[dataset['Food_ID'] == Foodi].index[0]
        distances , indices = model.kneighbors(csr_dataset[Foodi],n_neighbors=n+1)    
        Food_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        Recommendations = []
        for val in Food_indices:
            Foodi = dataset.iloc[val[0]]['Food_ID']
            i = food[food['Food_ID'] == Foodi].index
            Recommendations.append({'Name':food.iloc[i]['Name'].values[0],'Distance':val[1]})
        df = pd.DataFrame(Recommendations,index=range(1,n+1))
        return df['Name']
    else:
        return "No Similar Foods."
try:
    display = np.array(food_recommendation(finallist))
except NameError:
    st.write("")

combined_final = ans[['Name','Fats DV%', 'Carbs DV%','Proteins DV%']]
combined_final = combined_final.drop_duplicates()

try:
    if proceedVal == True:
        proceedVal1 = st.checkbox("View Recommendations ?")
        if proceedVal1 == True:
            st.write(combined_final.loc[combined_final['Name'].isin(display)])
except NameError:
    st.write("")