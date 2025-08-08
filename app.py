import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page Config
st.set_page_config(page_title="Titanic EDA Dashboard", layout="wide")
st.title("🚢 Titanic Data Analytics Dashboard")

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_titanic.csv")

df = load_data()

# Show Raw Data
if st.checkbox("Show Raw Data"):
    st.dataframe(df)

# Sidebar Filters
st.sidebar.header("Filter Options")

# Gender filter with 'All' option
gender_options = ["All"] + df["Sex"].unique().tolist()
gender = st.sidebar.selectbox("Select Gender", options=gender_options)

# Pclass filter with 'All' option
pclass_options = ["All"] + sorted(df["Pclass"].unique().tolist())
pclass = st.sidebar.selectbox("Select Passenger Class", options=pclass_options)

# Apply filters
filtered_df = df.copy()
if gender != "All":
    filtered_df = filtered_df[filtered_df["Sex"] == gender]
if pclass != "All":
    filtered_df = filtered_df[filtered_df["Pclass"] == pclass]

# Filtered data preview
st.subheader("Filtered Data Preview")
st.write(filtered_df.head())

# Add Survival Status column (with safe copy)
filtered_df = filtered_df.copy()
filtered_df["Survival Status"] = filtered_df["Survived"].map({0: "Did Not Survive", 1: "Survived"})

# Visualization: Survival Count
st.subheader("🎯 Survival Count by Gender")
fig1, ax1 = plt.subplots()
sns.countplot(data=filtered_df, x="Survival Status", hue="Sex", ax=ax1)
ax1.set_title("Survival Count by Gender")
st.pyplot(fig1)

# Additional Visualization: Age Distribution
st.subheader("📊 Age Distribution")
fig2, ax2 = plt.subplots()
sns.histplot(data=filtered_df, x="Age", bins=30, kde=True, ax=ax2)
ax2.set_title("Age Distribution of Filtered Passengers")
st.pyplot(fig2)

# Summary Statistics
st.subheader("📋 Summary Statistics")
st.write(filtered_df.describe(include="all"))
