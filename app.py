import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# Import models
from Model.logistic_regression import train_model as train_lr
from Model.decision_tree import train_model as train_dt
from Model.knn import train_model as train_knn
from Model.naive_bayes import train_model as train_nb
from Model.random_forest import train_model as train_rf
from Model.xgboost_model import train_model as train_xgb

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Clinical Trial ML Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# PROFESSIONAL CUSTOM STYLING
# --------------------------------------------------
st.markdown("""
<style>
    /* Import Professional Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main App Background - Solid Professional Dark */
    .stApp {
        background: #0E1117;
    }
    
    /* Remove default Streamlit header padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Professional Header Container */
    .header-container {
        background: linear-gradient(135deg, #1a1f2e 0%, #2d1b3d 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Header Title */
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .header-subtitle {
        text-align: center;
        color: #a0aec0;
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Section Title */
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #ffffff;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
        display: inline-block;
    }
    
    /* Metric Cards - Professional Design */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(102, 126, 234, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        margin: 0.5rem 0;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    .metric-card h4 {
        color: #94a3b8;
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0 0 0.75rem 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-card h2 {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Info Card for Dataset Overview */
    .info-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d4a6d 100%);
        padding: 1.25rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid rgba(59, 130, 246, 0.3);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    
    .info-card h3 {
        color: #60a5fa;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0 0 0.5rem 0;
        text-transform: uppercase;
    }
    
    .info-card p {
        color: #ffffff;
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #16161a 100%);
        border-right: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    section[data-testid="stSidebar"] > div {
        background: transparent;
    }
    
    /* Sidebar Header */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    section[data-testid="stSidebar"] label {
        color: #e2e8f0 !important;
        font-weight: 500;
    }
    
    /* File Uploader Styling */
    .uploadedFile {
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 8px;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-size: 0.95rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
    }
    
    /* Selectbox Styling */
    .stSelectbox > div > div {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 8px;
        color: #ffffff;
    }
    
    /* Dataframe Styling */
    .dataframe {
        background: #1e293b;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.3), transparent);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Info/Warning/Error Messages */
    .stAlert {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 8px;
    }
    
    /* DataFrame Container */
    [data-testid="stDataFrame"] {
        background: #1e293b;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    /* Metrics Built-in Streamlit */
    [data-testid="stMetricValue"] {
        color: #ffffff;
        font-size: 1.75rem;
    }
    
    [data-testid="stMetricLabel"] {
        color: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER SECTION
# --------------------------------------------------
st.markdown("""
<div class='header-container'>
    <h1 class='header-title'>🧬 Clinical Trial Classification Prediction</h1>
    <p class='header-subtitle'>Advanced Machine Learning Dashboard for Clinical Trial Outcome Analysis</p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR CONFIGURATION
# --------------------------------------------------
st.sidebar.markdown("### ⚙️ Configuration Panel")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader(
    "📁 Upload Dataset",
    type=["csv"],
    help="Upload your clinical trial dataset in CSV format"
)

if uploaded_file is None:
    st.info("👈 **Please upload a dataset from the sidebar to begin analysis**")
    st.stop()

# --------------------------------------------------
# LOAD AND PROCESS DATA
# --------------------------------------------------
df = pd.read_csv(uploaded_file)
df.columns = df.columns.str.strip()

# Dataset Overview Section
st.markdown("<h3 class='section-title'>📊 Dataset Overview</h3>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class='info-card'>
        <h3>Total Rows</h3>
        <p>{df.shape[0]:,}</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class='info-card'>
        <h3>Total Columns</h3>
        <p>{df.shape[1]}</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class='info-card'>
        <h3>Missing Values</h3>
        <p>{df.isna().sum().sum()}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.dataframe(df.head(10), use_container_width=True, height=300)

st.divider()

# --------------------------------------------------
# TARGET COLUMN SELECTION
# --------------------------------------------------
st.sidebar.markdown("---")
target_column = st.sidebar.selectbox(
    "🎯 Select Target Column",
    df.columns,
    help="Choose the column you want to predict"
)

# Data Preprocessing
df = df.dropna(subset=[target_column])

if df[target_column].nunique() < 2:
    st.error("❌ Target column must have at least 2 unique classes.")
    st.stop()

X = df.drop(columns=[target_column])
y = df[target_column]

# Encode categorical features
for col in X.select_dtypes(include="object").columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

y = LabelEncoder().fit_transform(y.astype(str))
y = pd.Series(y)

# Remove rare classes
class_counts = y.value_counts()
valid_classes = class_counts[class_counts >= 2].index
X = X[y.isin(valid_classes)]
y = y[y.isin(valid_classes)]
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# --------------------------------------------------
# TRAIN-TEST SPLIT
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------
st.sidebar.markdown("---")
model_name = st.sidebar.selectbox(
    "🤖 Select ML Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ],
    help="Choose the machine learning algorithm"
)

# Model Training
with st.spinner(f"🚀 Training {model_name} model..."):
    if model_name == "Logistic Regression":
        model = train_lr(X_train, y_train)
    elif model_name == "Decision Tree":
        model = train_dt(X_train, y_train)
    elif model_name == "KNN":
        model = train_knn(X_train, y_train)
    elif model_name == "Naive Bayes":
        model = train_nb(X_train, y_train)
    elif model_name == "Random Forest":
        model = train_rf(X_train, y_train)
    else:
        model = train_xgb(X_train, y_train)

    y_pred = model.predict(X_test)

# Calculate AUC
auc = "N/A"
if hasattr(model, "predict_proba"):
    try:
        y_prob = model.predict_proba(X_test)
        if len(np.unique(y_test)) == 2:
            auc = roc_auc_score(y_test, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
    except:
        auc = "N/A"

# --------------------------------------------------
# MODEL PERFORMANCE METRICS
# --------------------------------------------------
st.markdown("<h3 class='section-title'>📈 Model Performance Metrics</h3>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
mcc = matthews_corrcoef(y_test, y_pred)

# Display metrics in cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Accuracy</h4>
        <h2>{accuracy:.3f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Precision</h4>
        <h2>{precision:.3f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Recall</h4>
        <h2>{recall:.3f}</h2>
    </div>
    """, unsafe_allow_html=True)

col4, col5, col6 = st.columns(3)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <h4>F1 Score</h4>
        <h2>{f1:.3f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="metric-card">
        <h4>MCC</h4>
        <h2>{mcc:.3f}</h2>
    </div>
    """, unsafe_allow_html=True)

with col6:
    auc_display = auc if auc == "N/A" else f"{auc:.3f}"
    st.markdown(f"""
    <div class="metric-card">
        <h4>AUC Score</h4>
        <h2>{auc_display}</h2>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------
st.markdown("<h3 class='section-title'>🎯 Confusion Matrix</h3>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

cm = confusion_matrix(y_test, y_pred)

# Create figure with dark theme
fig, ax = plt.subplots(figsize=(8, 6), facecolor='#1e293b')
ax.set_facecolor('#1e293b')

# Plot heatmap with custom colors
sns.heatmap(
    cm, 
    annot=True, 
    fmt="d", 
    cmap="RdPu",
    ax=ax,
    cbar_kws={'label': 'Count'},
    linewidths=1,
    linecolor='#334155'
)

ax.set_xlabel("Predicted Labels", fontsize=12, color='white', fontweight='bold')
ax.set_ylabel("True Labels", fontsize=12, color='white', fontweight='bold')
ax.set_title(f"Confusion Matrix - {model_name}", fontsize=14, color='white', fontweight='bold', pad=20)

# Style tick labels
ax.tick_params(colors='white', labelsize=10)
plt.setp(ax.get_xticklabels(), rotation=0)
plt.setp(ax.get_yticklabels(), rotation=0)

# Colorbar styling
cbar = ax.collections[0].colorbar
cbar.ax.yaxis.set_tick_params(color='white')
cbar.ax.yaxis.label.set_color('white')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

plt.tight_layout()
st.pyplot(fig)

st.divider()

# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------
st.markdown("<h3 class='section-title'>📋 Detailed Classification Report</h3>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Create styled text for classification report
report = classification_report(y_test, y_pred, zero_division=0)
st.code(report, language="text")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #64748b; padding: 2rem 0; border-top: 1px solid rgba(100, 116, 139, 0.2);'>
    <p style='margin: 0; font-size: 0.875rem;'>Built with ❤️ using Streamlit | Clinical Trial ML Prediction Dashboard v2.0</p>
</div>
""", unsafe_allow_html=True)
