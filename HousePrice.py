#!/usr/bin/env python
# coding: utf-8

# In[51]:


#pip install matplotlib


# In[53]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import matplotlib.pyplot as plt
import folium


# In[54]:


data = pd.read_csv("realestate.csv")


# In[55]:


data


# In[56]:


X = data[['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']]  # Independent variables
y = data['Y']  # Dependent variable


# In[57]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[58]:


# Create the Gradient Boosting Regressor model
gb_regressor = GradientBoostingRegressor(
    n_estimators=100,  # Number of boosting stages to be used
    learning_rate=0.1,  # Step size shrinkage used in each boosting iteration
    max_depth=3,  # Maximum depth of individual trees
    random_state=42  # Seed for reproducibility
)


# In[59]:


gb_regressor.fit(X_train, y_train)


# In[60]:


y_pred = gb_regressor.predict(X_test)


# In[61]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")


# In[62]:


model = gb_regressor  # Load your trained model here


# In[70]:


import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

# Assuming you have already loaded your trained model
model = gb_regressor  # Load your trained model here

# Define the Streamlit app
st.title("House price prediction")

# Input fields for the 7 independent variables
st.sidebar.header("Input Features")

# X1 transaction date input through calendar
X1_date = st.sidebar.date_input("X1 transaction date", min_value=pd.to_datetime("1900-01-01"), max_value=pd.to_datetime("2023-01-01"), value=pd.to_datetime("2022-01-01"))

# Convert the date to an integer (you might want to adjust this conversion based on your use case)
X1 = int(X1_date.strftime("%Y%m%d"))

# Other variables as sliders
X2 = st.sidebar.slider("X2 house age", min_value=0, max_value=100, value=50)
X3 = st.sidebar.slider("X3 distance to the nearest MRT station", min_value=0, max_value=5000, value=50)
X4 = st.sidebar.slider("X4 number of convenience stores", min_value=0, max_value=100, value=50)
X5 = st.sidebar.slider("X5 latitude", min_value=0, max_value=50, value=25)
X6 = st.sidebar.slider("X6 longitude", min_value=0, max_value=500, value=250)
Area = st.sidebar.slider("Area(in metres)", min_value=0, max_value=5000, value=250)

# Make predictions
input_data = pd.DataFrame({
    'X1 transaction date': [X1],
    'X2 house age': [X2],
    'X3 distance to the nearest MRT station': [X3],
    'X4 number of convenience stores': [X4],
    'X5 latitude': [X5],
    'X6 longitude': [X6],
})


prediction = model.predict(input_data)

fig, ax = plt.subplots()
ax.scatter(data['Y'], model.predict(data[['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']]), alpha=0.5)
ax.plot([0, max(data['Y'])], [0, max(data['Y'])], '--r', linewidth=2)
ax.set_xlabel('Actual House Price')
ax.set_ylabel('Predicted House Price')
ax.set_title('Predicted vs Actual House Prices')
st.pyplot(fig)

# Multiply the prediction by the input value
result = prediction[0] * Area

# Display the result
st.subheader("Result")
st.write(f"The result of multiplying the prediction by the input value is: {result:.2f} dollars")




# In[ ]:





# In[ ]:





# In[ ]:




