import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from IPython.display import display
import ipywidgets as widgets
from scipy.stats import skew
st.set_page_config(layout="wide")

# Centered Title
st.markdown("<h1 style='text-align: center;'>Car Price Prediction Showcase 🚗💰</h1>", unsafe_allow_html=True)

st.header("Introduction")
st.write("""
Welcome to my Car Price Prediction Showcase!

The goal of this project was to predict the price of a car based on various features such as engine size, car length, car width, and many more.

While exploring the dataset, I noticed that the target variable — car price — is a continuous numerical value. This made it clear that I needed to approach this as a **regression problem**.

Using technologies like **Python**, **Pandas**, **Scikit-learn**, and **Streamlit**, I built a machine learning model that can accurately predict car prices based on the given features.

Let's dive into the journey and see how the project unfolds! 👇
""")

st.header("Libraries Used 📚")
st.write("""
- **pandas**: For data manipulation and analysis.
- **openpyxl**: For working with Excel files.
- **seaborn**: For beautiful data visualization.
- **scikit-learn**: For building and evaluating machine learning models.
- **category_encoders**: For encoding categorical variables effectively.

*(All libraries were installed using pip commands like `!pip install pandas`.)*
""")


st.header("Dataset Overview 📄")

st.write("""
Here is a glimpse of the dataset used for building the car price prediction model.  
It includes features such as **enginesize**, **carlength**, **carwidth**, and many more.
""")

import pandas as pd

# Load your dataset (replace with your path if needed)
df = pd.read_csv(r"C:\Users\karti\OneDrive\Desktop\Car Price\train (1).csv")  # 👈 replace 'your_dataset.csv' with your actual file name

# Show the first few rows
st.dataframe(df.head())

st.subheader("Feature Selection")
st.write(
    "‘Car ID’ and ‘Car Name’ were removed from the dataset as they did not have a meaningful impact on predicting the car price. "
    "‘Car ID’ was simply a unique identifier without predictive value, and ‘Car Name’ was mostly a branding label that did not significantly affect price predictions. "
    "Removing them helped improve model focus and performance."
)



col1, col2, col3 = st.columns(3)

# First chart in column 1
with col1:
    fig1, ax1 = plt.subplots(figsize=(3,2))  # smaller chart
    df['cylindernumber'].value_counts().plot(kind='bar', ax=ax1)
    ax1.set_title('Cylinders')
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    st.pyplot(fig1)
    st.write("✅ Most cars had four cylinders.")

# Second chart in column 2
with col2:
    fig2, ax2 = plt.subplots(figsize=(3,2))
    df['enginelocation'].value_counts().plot(kind='bar', ax=ax2)
    ax2.set_title('Engine Location')
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    st.pyplot(fig2)
    st.write("✅ Most engines are at the front.")

# Third chart in column 3
with col3:
    fig3, ax3 = plt.subplots(figsize=(3,2))
    df['fueltype'].value_counts().plot(kind='bar', ax=ax3)
    ax3.set_title('Fuel Type')
    ax3.set_xlabel('')
    ax3.set_ylabel('')
    st.pyplot(fig3)
    st.write("✅ Most cars use gas.")

st.subheader("Outlier Detection and Treatment")
st.write(
    "We used boxplots to detect outliers in numerical features. In cases where outliers appeared very rarely (less than 10 instances), "
    "we decided to remove them. This helped ensure the model focused on learning from the main patterns in the data without being skewed by extreme rare values."
)


st.subheader("Feature Engineering and Encoding")
st.write(
    "To prepare the data for modeling, we applied encoding techniques to convert categorical variables into numerical format:\n"
    "- For features like 'fuel type', 'aspiration', 'drive wheel', 'engine location', 'cylinder number', and 'door number', we used **Label Encoding** since they have a simple set of categories.\n"
    "- For more complex features like 'car body', 'engine type', and 'fuel system', we applied **Target Encoding** to capture their relationship with the car price directly.\n"
    "\n"
    "Additionally, we verified that there were **no missing values** in the dataset, ensuring clean and complete data for modeling."
)

clean_df = pd.read_csv(r"C:\Users\karti\OneDrive\Desktop\Car Price Prediction\clean_df.csv")
st.dataframe(clean_df.head())

# Get numerical columns except 'price'
numerical_cols = clean_df.select_dtypes(include='number').columns.tolist()
for col_to_remove in ['price', 'carbody', 'enginetype', 'fuelsystem','fueltype','aspiration','symboling','doornumber','drivewheel','enginelocation','cylindernumber']:
    if col_to_remove in numerical_cols:
        numerical_cols.remove(col_to_remove)

# Dropdown for selecting a feature
selected_feature = st.selectbox("Select a feature to see its relationship with Price:", numerical_cols)
####
####
col1, col2, col3 = st.columns([1,2,1])  # Middle column is wider

# Only plot inside the center column
with col2:
    fig, ax = plt.subplots(figsize=(4, 3))  # Smaller figure
    sns.scatterplot(x=clean_df[selected_feature], y=clean_df['price'], ax=ax)
    ax.set_title(f'{selected_feature} vs Price')
    ax.set_xlabel(selected_feature)
    ax.set_ylabel('Price')
    st.pyplot(fig)

# Streamlit selectbox to select a numerical column
selected_feature = st.selectbox("Select a feature to see its Gaussian Distribution:", numerical_cols)

col1, col2, col3 = st.columns([1,2,1])  # Middle column is wider
# Function to plot Gaussian chart using seaborn
def plot_gaussian(column):
    data = clean_df[column].dropna()
    skew_value = skew(data)
    st.markdown(f"**Skewness for {column}:** `{skew_value:.4f}`")
    fig, ax = plt.subplots(figsize=(8, 4))  # Create figure and axis objects
    sns.kdeplot(data=clean_df, x=column, fill=True, color='skyblue', ax=ax)
    ax.set_title(f'Gaussian Distribution of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Density')
    ax.grid(True)
    st.pyplot(fig)  # Pass the figure explicitly to Streamlit

# Plot the Gaussian chart based on the selected feature
with col2:
    plot_gaussian(selected_feature)
st.write("Compression Ratio is highly skewed towards right whereas car length is left skewed from the chart")

st.write("An important observation was Price of Cars decreases initially inidcating that the cars with higher risks are cheaper but then it increases again indicating that some high end cars are expensive")
col1, col2, col3 = st.columns([1,2,1])  # Middle column is wider

column_name = 'symboling'  # Example
with col2:
    plt.figure(figsize=(8, 6))
    sns.lineplot(x=clean_df[column_name], y=clean_df['price'])
    plt.title(f'{column_name} vs Price')
    plt.xlabel(column_name)
    plt.ylabel('Price')
    st.pyplot(plt)

st.subheader("Key Insights from Feature Analysis")
st.write(
    "- Cars with larger **wheel base** tend to be costlier, although the relation is slightly skewed.\n"
    "- **Car length** and **car width** show strong positive relationships with price.\n"
    "- **Curbweight** has a strong positive correlation with price.\n"
    "- **Engine size** and **boreratio** show moderate positive relationships but are not the only factors affecting price.\n"
    "- **Stroke** and **compression ratio** have very weak or no relationship with price.\n"
    "- **Horsepower** shows a strong positive correlation and is an important predictor of price.\n"
    "- **Peak RPM** has very little relationship with price.\n"
    "- **City MPG** and **highway MPG** show a negative correlation with price (higher MPG, lower price)."
)

excluded_columns = ['carbody', 'enginetype', 'fuelsystem','enginelocation']
df_filtered = clean_df.drop(columns=excluded_columns)

# Select numerical columns
df_numerical = df_filtered.select_dtypes(include=['number'])

@st.cache_data
def compute_correlation_matrix(df):
    return df.corr()

# Compute the correlation matrix and cache it
correlation_matrix = compute_correlation_matrix(df_numerical)

# Center the plot using columns
st.subheader("Feature Correlation Matrix")
fig, ax = plt.subplots(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
ax.set_title("Correlation between Features")
st.pyplot(fig)

st.subheader("Detailed Correlation Analysis")
st.write(
    """
    Analyzing the correlation matrix, we observed the following key insights:

    1. **Symboling** - No significant correlation with price.
    2. **Wheelbase** - Moderate positive correlation with price (0.67).
    3. **Carlength** - Strong positive correlation with price (0.76).
    4. **Carwidth** - Strong positive correlation with price (0.75).
    5. **Carheight** - Weak negative correlation with price (-0.17).
    6. **Curbweight** - Strong positive correlation with price (0.76).
    7. **Enginesize** - Very strong positive correlation with price (0.75).
    8. **Bore** - Moderate correlation with enginesize (0.69).
    9. **Stroke** - Moderate positive with bore but negative with price (-0.56).
    10. **Compression Ratio** - Weakly correlated with price.
    11. **Horsepower** - Strong relationship with enginesize (0.87).
    12. **Peak RPM** - Weakly related to horsepower and enginesize.
    13. **City MPG** - Strong negative correlation with price (-0.71).
    14. **Highway MPG** - Strong negative correlation with price (-0.71).

    **Conclusion:**  
    Price is most strongly influenced by carlength, curbweight, enginesize, and wheelbase. 
    Larger and more powerful cars tend to be more expensive.
    """
)

st.subheader("Feature Selection")
st.write(
    "To identify the most important features for predicting car price, we applied multiple feature selection techniques, including **SelectKBest**, **Recursive Feature Elimination (RFE)**, **Lasso Regression**, and **Random Forest Regressor**.\n"
    "After analyzing feature importance across these methods, we selected the following 10 features for modeling:\n"
    "- Curbweight\n"
    "- Highway MPG\n"
    "- Horsepower\n"
    "- Carlengh\n"
    "- Wheelbase\n"
    "- Enginesize\n"
    "- Carwidth\n"
    "- Fuelsystem\n"
    "- Peak RPM\n"
    "- City MPG\n"
)

# Your final metric data
st.markdown("<h2 style='font-weight: bold;'>Model Evaluation and Tuning</h2>", unsafe_allow_html=True)

linear_regression_metrics = {
    "Method": [
        "Basic Linear Regression",
        "After K-Fold Validation",
        "After Gradient Descent Optimization"
    ],
    "R² Score": [
        "0.7516", 
        "0.7173 (Average)", 
        "0.7699"
    ],
    "MAE": [
        "1239.75 (12.46%)", 
        "1791.15 (12.45%)", 
        "1707.84 (14.56%)"
    ],
    "RMSE": [
        "1823.44 (18.33%)", 
        "-",  # No RMSE calculated for K-Fold here
        "2359.37 (20.11%)"
    ]
}
@st.cache_data
def create_linear_regression_df(metrics):
    return pd.DataFrame(metrics)
# Create DataFrame
linear_regression_df = create_linear_regression_df(linear_regression_metrics)

# Streamlit display

st.subheader("Linear Regression - Model Evaluation at Different Stages")
st.dataframe(linear_regression_df)

sgd_gridsearch_results = {
    "Learning Rate": [0.001, 0.001, 0.01],
    "Max Iterations": [100, 150, 50],
    "Train R²": [0.7702, 0.7702, 0.7668],
    "Test R²": [0.7501, 0.7501, 0.7477],
    "Test MAE": ["1253.69 (12.60%)", "1253.69 (12.60%)", "1242.49 (12.49%)"],
    "Test RMSE": ["1829.06 (18.39%)", "1829.06 (18.39%)", "1837.67 (18.47%)"],
    "Test Loss %": ["24.99%", "24.99%", "25.23%"]
}

# Create DataFrame
@st.cache_data
def gridsearch_df(model_eval):
    return pd.DataFrame(model_eval)
sgd_gridsearch_df = gridsearch_df(sgd_gridsearch_results)

# Show in Streamlit
st.subheader("SGD Regressor - Grid Search CV Results")
st.dataframe(sgd_gridsearch_df)


st.subheader("Training vs Validation Loss - SGDRegressor")
st.image("https://i.ibb.co/RX2xLLQ/mse-losses.png", caption="Training and Validation Loss over Epochs")

model_evaluation = {
    "Model": [
        "Ridge Regression",
        "Lasso Regression",
        "Random Forest (Simple)",
        "Random Forest (K-Fold CV)",
        "XGBoost",
        "SVR"
    ],
    "R² Score": [0.7516, 0.7516, 0.8950, 0.8548, 0.7552, -0.0086],
    "MAE": [
        "1239.88 (12.46%)", 
        "1239.88 (12.46%)", 
        "851.01 (8.56%)", 
        "1004.64 (9.57%)", 
        "1590.62 (14.65%)", 
        "3030.39 (30.46%)"
    ],
    "RMSE": [
        "1823.62 (18.33%)", 
        "1823.65 (18.33%)", 
        "1185.61 (11.92%)", 
        "1596.72 (15.21%)", 
        "2314.40 (21.32%)", 
        "3674.58 (36.94%)"
    ]
}
@st.cache_data
def create_model_eval_df(model_eval):
    return pd.DataFrame(model_eval)
# Create dataframe
model_eval_df = create_model_eval_df(model_evaluation)

# Display in Streamlit
st.subheader("Model Evaluation Results Across Different Models and Techniques")
st.dataframe(model_eval_df)

st.subheader("Training vs Validation MSE - Random Forest Regressor")
st.image("https://i.ibb.co/KjDj84kL/mse-in-rfe.png", caption="Training and Validation Loss over n estimators")

st.write(
    "Random Forest (Simple) achieved the best overall performance in terms of R² and errors, "
    "SVR performed poorly in this case, possibly due to inappropriate hyperparameters or sensitivity to scaling."
)


st.subheader("Feature Importance - Pie Chart")
st.image("https://i.ibb.co/dw14tG5K/feature-importance-graph.png", caption="Feature importance of selected features.")

st.write(
    "Earlier, after handling outliers, a few extreme values remained, mainly in the following features:\n\n"
    "- **Compression Ratio**: 23 outliers\n"
    "- **Stroke**: 16 outliers\n"
    "- **Car Width**: 3 outliers\n"
    "- **Car Length**: 1 outlier\n\n"
    "We do not have **Compression Ratio** in our final model, but **Car Width**, **Car Length**, and **Stroke** are still included."
)

st.subheader("Model Performance After Dropping Outlier-Prone Features")

st.write(
    "After dropping features associated with significant remaining outliers (such as Compression Ratio, Stroke, etc.), "
    "the model was retrained using the selected important features. The performance improved noticeably:\n\n"
    "- **R² Score**: 0.9049\n"
    "- **Mean Absolute Error (MAE)**: 782.96 (7.87%)\n"
    "- **Root Mean Squared Error (RMSE)**: 1128.44 (11.34%)"
)

st.write(
    "After dropping more features which were demonstarting high collinearity (such as Horsepower, Curbweight, etc.), "
    "the model was retrained using the selected important features. The performance improved noticeably:\n\n"
    "- **R² Score**: 0.9062\n"
    "- **Mean Absolute Error (MAE)**: 752.16 (7.56%)\n"
    "- **Root Mean Squared Error (RMSE)**: 1120.35 (11.26%)"
)

st.subheader("Final Observations and Recommendations")

st.write(
    "- **For most normal cars**, our model predicts prices **very accurately**.\n"
    "- **For cars with extreme specifications** (e.g., very large engines or unusually light weights), "
    "the model tends to make **larger prediction errors**.\n"
    "- **Recommendation:** If the client plans to sell vehicles with extreme specs, "
    "we suggest **gathering more targeted data** or **developing a specialized model** for those cases."
)