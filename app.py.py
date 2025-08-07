import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# --- Page Config ---
st.set_page_config(page_title="Titanic EDA Dashboard", layout="wide")
st.title("🚢 Titanic Data Analytics Dashboard")

# --- Add Background Image ---
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Add the Titanic background image
add_bg_from_local("titanic.jpg")  # Ensure 'titanic.jpg' is in the same directory as app.py

# --- Load Data ---
@st.cache_data
def load_data():
    return pd.read_csv("cleaned_titanic.csv")

df = load_data()

# --- Show Raw Data ---
if st.checkbox("Show Raw Data"):
    st.dataframe(df)

# --- Sidebar Filters ---
st.sidebar.header("Filter Options")

gender_options = ["All"] + df["Sex"].unique().tolist()
pclass_options = ["All"] + sorted(df["Pclass"].unique().tolist())

gender = st.sidebar.selectbox("Select Gender", options=gender_options)
pclass = st.sidebar.selectbox("Select Passenger Class", options=pclass_options)

# Age Slider
age_min = int(df["Age"].min())
age_max = int(df["Age"].max())
age_range = st.sidebar.slider("Select Age Range", min_value=age_min, max_value=age_max, value=(age_min, age_max))

# Fare Slider
fare_min = float(df["Fare"].min())
fare_max = float(df["Fare"].max())
fare_range = st.sidebar.slider("Select Fare Range", min_value=fare_min, max_value=fare_max, value=(fare_min, fare_max))

# --- Apply Filters ---
filtered_df = df.copy()

if gender != "All":
    filtered_df = filtered_df[filtered_df["Sex"] == gender]

if pclass != "All":
    filtered_df = filtered_df[filtered_df["Pclass"] == pclass]

filtered_df = filtered_df[
    (filtered_df["Age"] >= age_range[0]) & (filtered_df["Age"] <= age_range[1]) &
    (filtered_df["Fare"] >= fare_range[0]) & (filtered_df["Fare"] <= fare_range[1])
]

# Add Survival Status Column
filtered_df = filtered_df.copy()
filtered_df["Survival Status"] = filtered_df["Survived"].map({0: "Did Not Survive", 1: "Survived"})

# --- Display Filtered Data ---
st.subheader("📄 Filtered Data Preview")
st.write(filtered_df.head())

# --- 📊 Visualizations in Columns ---

# Row 1: Survival Count | Age Distribution
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🎯 Survival Count by Gender")
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    sns.countplot(data=filtered_df, x="Survival Status", hue="Sex", ax=ax1)
    ax1.set_title("Survival by Gender")
    st.pyplot(fig1)

with col2:
    st.markdown("### 📊 Age Distribution")
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    sns.histplot(data=filtered_df, x="Age", bins=30, kde=True, ax=ax2)
    ax2.set_title("Age Distribution")
    st.pyplot(fig2)

# Row 2: Fare Distribution | Survival by Pclass
col3, col4 = st.columns(2)

with col3:
    st.markdown("### 💰 Fare Distribution")
    fig3, ax3 = plt.subplots(figsize=(5, 4))
    sns.histplot(data=filtered_df, x="Fare", bins=30, kde=True, ax=ax3, color='orange')
    ax3.set_title("Fare Distribution")
    st.pyplot(fig3)

with col4:
    st.markdown("### 🛏️ Survival Rate by Class")
    fig4, ax4 = plt.subplots(figsize=(5, 4))
    sns.barplot(data=filtered_df, x="Pclass", y="Survived", ax=ax4, palette="viridis")
    ax4.set_title("Survival by Passenger Class")
    st.pyplot(fig4)

# Row 3: Embarked | Heatmap
col5, col6 = st.columns(2)

with col5:
    if "Embarked" in filtered_df.columns:
        st.markdown("### 🛳️ Survival by Embarkation Point")
        fig5, ax5 = plt.subplots(figsize=(5, 4))
        sns.barplot(data=filtered_df, x="Embarked", y="Survived", ax=ax5, palette="coolwarm")
        ax5.set_title("Survival by Embarked Port")
        st.pyplot(fig5)

with col6:
    st.markdown("### 🔗 Correlation Heatmap")
    numeric_df = filtered_df.select_dtypes(include=['float64', 'int64'])
    fig6, ax6 = plt.subplots(figsize=(5, 4))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="Blues", fmt=".2f", ax=ax6)
    ax6.set_title("Correlation Between Features")
    st.pyplot(fig6)

# --- Summary Stats ---
st.subheader("📋 Summary Statistics")
st.write(filtered_df.describe(include="all"))

