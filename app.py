
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime
import streamlit as st
import pycountry
from prophet import Prophet
import joblib
st.set_page_config(layout='wide')
df = pd.read_csv("car_prices_cleaned.csv")
df=df[df["country_name"] != "Israel"]
df["saledate"] = pd.to_datetime(df["saledate"], errors="coerce")
# Get counts for makes and countries
df_make=df["make"].value_counts().reset_index()
df_country_name=df["country_name"].value_counts().reset_index()
country_names=[]
# Manual mapping for country codes and states
manual_country_mapping = {
    'Tx': 'United States',
    'Wi': 'United States',
    'Fl': 'United States',
    'Oh': 'United States',
    'Mi': 'United States',
    'Nj': 'United States',
    'Ut': 'United States',
    'Nv': 'United States',
    'Ny': 'United States',
    'Or': 'United States',
    'Wa': 'United States',
    'Hi': 'United States',
    'Ok': 'United States',
    'Nm': 'United States',
    'Al': 'United States',

    'Qc': 'Canada',
    'Ab': 'Canada',
    'On': 'Canada',
    'Ns': 'Canada',

    'Moldova, Republic of': 'Moldova',
    "Lao People's Democratic Republic": 'Laos',
    'Holy See (Vatican City State)': 'Vatican City',
    'Macao': 'Macau',
    'Puerto Rico': 'United States',
    'New Caledonia': 'France',
    'Montserrat': 'United Kingdom'
}
# Map state codes to country names
for code in df["state"]:
    country = pycountry.countries.get(alpha_2=code.upper())
    if country:
        country_names.append(country.name)
    elif code in manual_country_mapping:
        country_names.append(manual_country_mapping[code])
    else:
        country_names.append("unknown")
df["country_name_1"]=country_names
df_con=df[df["condition"].isin(df["condition"].value_counts().index.to_list())]
df_con["condition"]=df_con["condition"].astype("str")
# Prepare monthly sales data
monthly_sales = df.groupby(pd.Grouper(key='saledate', freq='M')).size().reset_index(name='sales_count')

prophet_df = monthly_sales.rename(columns={'saledate': 'ds', 'sales_count': 'y'})

model = Prophet(yearly_seasonality=True)
model.fit(prophet_df)

future = model.make_future_dataframe(periods=6, freq='M')
forecast = model.predict(future)

actual = prophet_df[prophet_df['y'] > 0].copy()
actual['type'] = 'Actual'

pred = forecast[['ds', 'yhat']].copy()
pred.rename(columns={'yhat': 'y'}, inplace=True)
pred['type'] = 'Predicted'
combined = pd.concat([pred, actual], ignore_index=True)


def page1():
    
    tab1,tab2,tab3 = st.tabs(["Sales Overview","Profit Overview","Geographic Analysis"])
    # Sales Overview Tab
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.pie(data_frame=df , names= df["saledate"].dt.year,hole=.6,title="Sales percentages for each year"))
            st.plotly_chart(px.pie(data_frame=df[df["year"] > 2005] ,names="year",title="Car sales by year of manufacture"))
            st.plotly_chart(px.treemap(data_frame=df ,path=["make","year"],title="Number of car sales by companies each year"))
            st.plotly_chart(px.pie(data_frame=df , names="distance_category",title="Percentage of cars sold by distance"))
            st.plotly_chart(px.treemap(data_frame=df[df["model"].isin(df["model"].value_counts().head(20).index.tolist())] , path=["make","model","sale_category"],title=" Car Sales Performance by Company and Model Compared to Expectations"))

        with col2:
            st.plotly_chart(px.pie(data_frame=df , names= "season",hole=.6,title="Sales percentages for each season"))
            st.plotly_chart(px.histogram(data_frame=df_make ,x="make",y="count",text_auto=True,title="Sales of each company"))
            st.plotly_chart(px.treemap(data_frame=df[df["model"].isin(df["model"].value_counts().head(10).index.tolist())], path=["make","model","body"], values="sellingprice", title="Top 10 Car Models by Brand and body"))
            st.plotly_chart(px.histogram(data_frame=df , x = "color",title="Most requested colors"))
    
            fig1 = px.line(combined, x='ds', y='y', color='type', markers=True,
                      title='Actual vs Predicted Monthly Car Sales')
        
            fig1.update_layout(
                xaxis_title='Date',
                yaxis_title='Number of Cars Sold',
                legend_title='Data Type',
                hovermode='x unified'
                    )
            st.plotly_chart(fig1)
    # Profit Overview Tab
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.pie(data_frame=df , names= df["saledate"].dt.year, values="sellingprice",hole=.6, title="Profit percentage for each year"))
            st.plotly_chart(px.pie(data_frame=df , names= "season", values="sellingprice",hole=.6,title="Profit percentages for each season"))
            st.plotly_chart(px.treemap(data_frame=df[df["model"].isin(df["model"].value_counts().head(10).index.tolist())], path=["make","model","year"],values="sellingprice" ,title="Top 10 Car Models by Brand and Year of Manufacture"))
            st.plotly_chart(px.treemap(data_frame=df[df["make"].isin(df["make"].value_counts().head(10).index.tolist())] ,path=["make","distance_category"],values="sellingprice",title="The company's most sold cars by distance"))
            
            
        with col2:
            st.plotly_chart(px.histogram(data_frame=df,x="make" ,y= "sellingprice" ,facet_col="season" ,histfunc="sum",text_auto=True,title="Company Sales Value per Season"))
            st.plotly_chart(px.pie(data_frame=df[df["year"] > 2005], values="sellingprice" ,names="year",title="Car sales value by year of manufacture"))
            st.plotly_chart(px.histogram(data_frame=df , x="distance_category" , y="sellingprice",text_auto=True,title="The effect of distance on the price of the car"))
            st.plotly_chart(px.histogram(data_frame=df , x="sale_category",title="Earnings relative to expectations"))
    # Geographic Analysis Tab        
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.treemap(data_frame=df[df["model"].isin(df["model"].value_counts().head(10).index.tolist())], path=["country_name","make"], values="sellingprice", title="Car Sales by Country and Vehicle Type"))
            
        with col2:
            st.plotly_chart(px.histogram(data_frame=df_country_name ,x="country_name",y="count",text_auto= True,title="Countries that buy the most cars"))
        fig = px.choropleth(df.groupby("country_name_1")["sellingprice"].sum().sort_values().reset_index(),locations="country_name_1",locationmode="country names",color_continuous_scale="Reds",color="sellingprice",title="Profit by country")
        fig.update_layout(
         width=1200,   # العرض
         height=700    # الارتفاع
            )
        st.plotly_chart(fig)   

def page2():
    
    def getInput():
        make = st.selectbox("car name : ",df["make"].unique())
        df_make = df[df["make"] == make]
        model = st.selectbox("car model : ",df_make["model"].unique())
        year = st.slider('Year of manufacture of the car'.title() , min_value=1990 , max_value=2015,value= 2014, step=1)
        odometer = st.slider('The distance traveled by the car'.title() , min_value=1 , max_value=250000, step=1)
        season = st.selectbox('select season'.title() , ["Winter" , "Autumn" , "Summer" ,"Spring" ])
        color  = st.selectbox("select color : ",df["color"].unique())
        condition = st.selectbox("select condition : ",df["condition"].unique())
        country_name = st.selectbox("select country name : ",df["country_name"].unique())
        body =  st.selectbox("select body : ",df["body"].unique())
        interior =  st.selectbox("select interior : ",df["interior"].unique())
        full_model = make + " " + model
        state = df[df["country_name"] == country_name]["state"].unique()[0]
        return pd.DataFrame(data=[[full_model,year,odometer,season,color,condition,state,body,interior]],
                            columns=["full_model","year","odometer","season","color","condition","state","body","interior"])
    st.title("Car Price Prediction")

        
    data = getInput()
    st.dataframe(data)


    model = joblib.load('car_price_RF2_model.pkl')
    predicted_price = model.predict(data)
    st.write(f"expected salary: {np.exp(predicted_price[0]):,.2f}")
    
pages = {
    'analysis' : page1,
    'ML' : page2
}
pg = st.sidebar.radio('Navigate between pages' , pages.keys())
pages[pg]()
