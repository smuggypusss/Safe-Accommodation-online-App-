import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df = pd.read_csv('C:/Users/Asus/NaiveBayes.csv')
st.title('Crime Analysis')

selected_state = st.selectbox("Select a State:", [''] + list(df['STATE/UT'].unique()))
if selected_state:
    @st.cache_data  # Cache the function to avoid rerunning it unnecessarily
    def safest_district_in_state(state, crime_threshold):
        # Filter data for the given state
        state_data = df[df["STATE/UT"] == state]

        # Define the target variable (safety label)
        state_data["SAFE"] = state_data["TOTAL IPC CRIMES"].apply(lambda x: 1 if x < crime_threshold else 0)

        # Select relevant features (crime statistics)
        features = state_data.drop(columns=["STATE/UT", "DISTRICT", "YEAR", "TOTAL IPC CRIMES", "SAFE"])

        # Standardize the features
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, state_data["SAFE"], test_size=0.2, random_state=42)

        # Train the Gaussian Naïve Bayes model with hyperparameter tuning
        model = GaussianNB()
        param_grid = {'var_smoothing': np.logspace(0, -9, num=100)}
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        # Use the best model to predict safety for each district
        best_model = grid_search.best_estimator_
        state_data["PREDICTED_SAFE"] = best_model.predict(features)

        # Calculate accuracy on the test set
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Find the safest district based on predictions
        safest_district = state_data[state_data["PREDICTED_SAFE"] == 1].iloc[0]

        return safest_district, accuracy

    state = selected_state
    crime_threshold = 5000  # Define your threshold value here
    safest_district_data, accuracy = safest_district_in_state(state, crime_threshold)

    # Display the accuracy
    st.header("Model Accuracy")
    st.write(f"Accuracy on the test set: {accuracy:.2f}")

    # Display the safest district information in a more presentable way
    st.header(f"Safest District in {state}")
    st.markdown(f"Below is the information about the safest district in {state}:")
    st.write("District Name:", safest_district_data["DISTRICT"])
    st.write("Total IPC Crimes:", safest_district_data["TOTAL IPC CRIMES"])
    st.write("Other Crime Statistics:")
    st.table(safest_district_data.drop(columns=["STATE/UT", "DISTRICT", "YEAR", "TOTAL IPC CRIMES", "SAFE", "PREDICTED_SAFE"]))
else:
    st.info("Please select a state!")
