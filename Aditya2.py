#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


data = pd.read_csv("realestate.csv")


# In[3]:


data


# In[4]:


X = data[['X1 transaction date', 'X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']]  # Independent variables
y = data['Y house price of unit area']  # Dependent variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[5]:


# Create the Gradient Boosting Regressor model
gb_regressor = GradientBoostingRegressor(
    n_estimators=100,  # Number of boosting stages to be used
    learning_rate=0.1,  # Step size shrinkage used in each boosting iteration
    max_depth=3,  # Maximum depth of individual trees
    random_state=42  # Seed for reproducibility
)

# Fit the model to the training data
gb_regressor.fit(X_train, y_train)


# In[6]:


# Make predictions on the test data
y_pred = gb_regressor.predict(X_test)


# In[7]:


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")


# In[8]:


#pip install streamlit


# In[15]:


import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


model = gb_regressor  # Load your trained model here

# Define the Streamlit app
st.title("House price prediction")

# Input fields for the 7 independent variables
st.sidebar.header("Input Features")
X1 = st.sidebar.slider("X1 transaction date", min_value=0.0, max_value=2023.0, value=50.0)
X2 = st.sidebar.slider("X2 house age", min_value=0.0, max_value=100.0, value=50.0)
X3 = st.sidebar.slider("X3 distance to the nearest MRT station", min_value=0.0, max_value=5000.0, value=50.0)
X4 = st.sidebar.slider("X4 number of convenience stores", min_value=0.0, max_value=100.0, value=50.0)
X5 = st.sidebar.slider("X5 latitude", min_value=0.0, max_value=50.0, value=50.0)
X6 = st.sidebar.slider("X6 longitude", min_value=0.0, max_value=500.0, value=50.0)

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

# Display the prediction
st.subheader("Prediction")
st.write(f"The predicted dependent variable value is: {prediction[0]:.2f}")


# In[ ]:




