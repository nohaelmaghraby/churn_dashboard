import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import plotly.figure_factory as ff
import pickle


# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="https://img.icons8.com/emoji/48/bar-chart-emoji.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Theme
alt.themes.enable("dark")
import os

# Check if files exist before loading
cleaned_file_path = r"D:\DEPI Graduation Project\New folder\Final_train_df_cleaned.csv"
test_file_path = r"D:\DEPI Graduation Project\project\kaggle_test_prediction_all.csv"
if os.path.exists(cleaned_file_path):
    df_cleaned = pd.read_csv(r"D:\DEPI Graduation Project\New folder\Final_train_df_cleaned.csv")
else:
    st.error(f"Error: File not found at {cleaned_file_path}")

if os.path.exists(test_file_path):
    df_test = pd.read_csv(r"D:\DEPI Graduation Project\project\kaggle_test_prediction_all.csv")
else:
    st.error(f"Error: File not found at {test_file_path}")
@st.cache_data
def load_data():
    df = pd.read_csv("D:\DEPI Graduation Project\Final train_df_cleaned.csv")
    df['song_completion_rate'] = df['num_100'] / (
        df['num_25'] + df['num_50'] + df['num_75'] + df['num_985'] + df['num_100'])
    df['song_completion_rate'] = df['song_completion_rate'].replace([np.inf, -np.inf], 0).fillna(0)
    df['avg_secs_per_song'] = df['total_secs'] / df['num_unq']
    df['avg_secs_per_song'] = df['avg_secs_per_song'].replace([np.inf, -np.inf], 0).fillna(0)
    df['high_engagement'] = (df['total_secs'] > df['total_secs'].median()).astype(int)
    df['membership_days'] = (
        pd.to_datetime(df['membership_expire_date']) - pd.to_datetime(df['transaction_date'])).dt.days.fillna(0)
    df['discount_rate'] = 1 - (df['actual_amount_paid'] / df['plan_list_price'])
    df['discount_rate'] = df['discount_rate'].replace([np.inf, -np.inf], 0).fillna(0)
    df['is_trial'] = (df['actual_amount_paid'] == 0).astype(int)
    df['auto_renew_but_cancel'] = ((df['is_auto_renew'] == 1) & (df['is_cancel'] == 1)).astype(int)
    df['age_group'] = pd.cut(df['bd'], bins=[10, 20, 30, 40, 50, 60, 100], labels=False, right=False)
    df['registration_init_time'] = pd.to_datetime(df['registration_init_time'], format='%Y%m%d', errors='coerce')
    df['account_age_days'] = (
        pd.to_datetime("2017-03-01") - df['registration_init_time']).dt.days.fillna(0)
    df['has_transaction'] = df['payment_method_id'].notna().astype(int)
    df['no_txn_and_churn'] = ((df['has_transaction'] == 0) & (df['is_churn'] == 1)).astype(int)
    df['payment_method_risk'] = df['payment_method_id'].map({
        18: 'low_risk', 11: 'low_risk', 31: 'low_risk',
        3: 'high_risk', 6: 'high_risk', 13: 'high_risk', 22: 'high_risk', -1: 'no_txn'})
    df['payment_method_risk'] = df['payment_method_risk'].fillna('medium_risk')
    df['gender'] = df['gender'].fillna('other')
    df['registered_via'] = df['registered_via'].fillna(-1).astype(int)
    df['payment_method_id'] = df['payment_method_id'].fillna(-1).astype(int)
    df['city'] = df['city'].fillna(-1).astype(int)
    df['age_bucket'] = pd.cut(df['bd'], bins=[13, 18, 25, 35, 45, 55, 65, 75, 85, 95])
    return df


# Main navigation with styled buttons
st.title("📊 Customer Churn Prediction Dashboard")
button_style = """
    <style>
        .stButton>button {
            width: 250px;
            height: 60px;
            background-color: #FF6347;  /* Tomato Red */
            color: white;
            font-size: 20px;
            font-weight: bold;
            border: none;
            border-radius: 30px;  /* Rounded corners */
            box-shadow: 0px 6px 15px rgba(0, 0, 0, 0.15);  /* Shadow */
            transition: all 0.3s ease-in-out;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #FF4500;  /* Orange Red */
            color: black;  /* Change text color to black on hover */
            transform: scale(1.05);  /* Slightly enlarge the button */
            box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.2);  /* Darker shadow on hover */
        }
        .stButton>button:focus {
            outline: none;
        }
    </style>
"""
st.markdown(button_style, unsafe_allow_html=True)

# Define session state if not already defined
if "page" not in st.session_state:
    st.session_state.page = None

# Buttons for navigation
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("Data Exploration"):
        st.session_state.page = "Data Exploration"
with col2:
    if st.button("Feature Engineering"):
        st.session_state.page = "Feature Engineering"
with col3:
    if st.button("Model Comparison"):
        st.session_state.page = "Model Comparison"
with col4:
    if st.button("Predict Churn"):
        st.session_state.page = "Predict Churn"

# Show page based on the button clicked
if st.session_state.page == "Data Exploration":
    st.subheader("🔍 Data Exploration:")
    st.markdown("""Explore the cleaned dataset and gain insights through interactive visualizations.""")
    # Show raw data
    st.subheader("📄 Preview of Cleaned Dataset")
    st.dataframe(df_cleaned.head(20))
    
        # Churn Distribution
    df_cleaned['churn_label'] = df_cleaned['is_churn'].map({0: 'Not Churned', 1: 'Churned'})
    # Key Fields explanation
    st.markdown("""
    - **Key Fields**:
      - `msno`: Unique customer identifier.
      - `is_churn`: Customer churn status — **1** indicates the customer churned, **0** means they renewed.
      - `bd`: Customer age — note that this column contains **outlier values** ranging from **-7000 to 2015**, which should be cleaned.
      - `payment_method_id`: Encoded payment method used by the customer.
      - `num_unq`: Number of **unique songs** played by the customer.
    """)
    # Plot pie chart
    st.subheader("🔁 Churn Distribution")
    churn_counts = df_cleaned['churn_label'].value_counts()
    fig1 = px.pie(names=churn_counts.index, values=churn_counts.values, 
              title='Churn vs Non-Churn Customers', 
              color_discrete_sequence=['#00cc96', '#ff4b4b'])

    fig1.update_traces(textinfo='percent+label', pull=[0.1] * len(churn_counts))
    st.plotly_chart(fig1)

    # Churn Rate by Payment Method
    df_cleaned['payment_method_id'] = df_cleaned['payment_method_id'].fillna(-1).astype(int)
    payment_method_churn = df_cleaned.groupby('payment_method_id')['is_churn'].mean().reset_index()
    payment_method_churn = payment_method_churn.sort_values(by='payment_method_id', ascending=True)

    # Plot churn rate for each payment method
    st.subheader("💳 Churn Rate by Payment Method")
    fig2, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='payment_method_id', y='is_churn', data=payment_method_churn, palette='viridis', ax=ax)
    ax.set_xlabel('Payment Method ID')
    ax.set_ylabel('Churn Rate')
    ax.set_title('Churn Rate by Payment Method')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig2)
    

    
    
    import plotly.express as px

    st.markdown("<h2 style='text-align: center;'>🔍 Churn Behavior Analysis (Interactive)</h2>", unsafe_allow_html=True)

    # ❌ Cancel Behavior and Churn - Interactive
    st.subheader("❌ Cancel Behavior and Churn (Interactive)")
    cancel_counts = df_cleaned.groupby(['is_cancel', 'is_churn']).size().reset_index(name='count')
    cancel_counts['percentage'] = round(100 * cancel_counts['count'] / cancel_counts['count'].sum(), 1)
    cancel_counts['is_cancel'] = cancel_counts['is_cancel'].replace({0: 'Not Cancelled', 1: 'Cancelled'})
    cancel_counts['is_churn'] = cancel_counts['is_churn'].replace({0: 'Stayed', 1: 'Churned'})

    fig3 = px.bar(cancel_counts, x='is_cancel', y='count', color='is_churn', 
              text='percentage', barmode='group',
              labels={'count': 'Number of Users', 'is_cancel': 'Cancel Status', 'is_churn': 'Churn Status'},
              title="Cancel Behavior and Churn")
    fig3.update_traces(texttemplate='%{text}%', textposition='outside')
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # 🔄 Auto Renew Behavior and Churn - Interactive
    st.subheader("🔄 Auto Renew Behavior and Churn (Interactive)")
    renew_counts = df_cleaned.groupby(['is_auto_renew', 'is_churn']).size().reset_index(name='count')
    renew_counts['percentage'] = round(100 * renew_counts['count'] / renew_counts['count'].sum(), 1)
    renew_counts['is_auto_renew'] = renew_counts['is_auto_renew'].replace({0: 'Not Auto Renewed', 1: 'Auto Renewed'})
    renew_counts['is_churn'] = renew_counts['is_churn'].replace({0: 'Stayed', 1: 'Churned'})

    
# ... (نفس الكود اللي قبل كده فوق بدون تغيير)

    # ✅ توزيع العمر ومعدل التسرب حسب الفئات - جنب بعض بنسبة 3:7
    st.markdown("---")
    st.markdown("<h2 style='text-align: center;'>👤 Age Analysis and Churn Rate</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 7])

    with col1:
        st.subheader("👤 Age Distribution")
        fig_age, ax_age = plt.subplots(figsize=(5, 4))  # خليت الارتفاع 4 بدل 3 عشان يكون مناسب زي الشكل الثاني
        sns.histplot(df_cleaned['bd'], bins=40, kde=True, ax=ax_age, color='skyblue')
        ax_age.set_title("Age Distribution", fontsize=14)
        ax_age.set_xlabel("Age", fontsize=12)
        ax_age.set_ylabel("Count", fontsize=12)
        ax_age.tick_params(axis='x', rotation=45)
        st.pyplot(fig_age)

    with col2:
        st.subheader("📊 Churn Rate by Age Bucket")
        # تأكدنا نعمل bucketing هنا قبل الرسم
        df_cleaned['age_bucket'] = pd.cut(df_cleaned['bd'], bins=[13, 18, 25, 35, 45, 55, 65, 75, 85, 95])

        fig_age_bucket, ax_age_bucket = plt.subplots(figsize=(8, 4))  # عرض أكبر 8x4 عشان تناسب النسبة 7
        sns.barplot(x='age_bucket', y='is_churn', data=df_cleaned, ax=ax_age_bucket, palette='viridis')
        ax_age_bucket.set_title("Churn Rate by Age Bucket", fontsize=14)
        ax_age_bucket.set_xlabel("Age Group", fontsize=12)
        ax_age_bucket.set_ylabel("Churn Rate", fontsize=12)
        ax_age_bucket.tick_params(axis='x', rotation=45)
        st.pyplot(fig_age_bucket)

    st.markdown("---")



    



elif st.session_state.page == "Feature Engineering":
    st.title("🔧 Feature Engineering Insights")
    cleaned_with_feature_path = r"D:\DEPI Graduation Project\train_cleaned_with_features.csv" 
   
# Feature Engineering Page
    df_cleaned_featured=pd.read_csv(cleaned_with_feature_path)
    
    st.markdown("### 🧩 Created Features Summary")
    features_info = {
        "avg_secs_per_song": "Average seconds per unique song",
        "high_engagement": "Top 50% users by total listening time",
        "membership_days": "Number of days between transaction and expiration",
        "discount_rate": "Discount percentage received",
        "is_trial": "User paid 0 (trial)",
        "auto_renew_but_cancel": "Auto-renew was ON but user canceled",
        "age_group": "User age bucket (grouped)",
        "account_age_days": "Days since account registration",
        "has_transaction": "Whether user ever made a payment",
        "no_txn_and_churn": "User churned without ever paying",
        "payment_method_risk": "Churn risk based on payment method"
    }

    feature_df = pd.DataFrame(list(features_info.items()), columns=["Feature", "Description"])
    st.dataframe(feature_df)

   

    st.markdown("### 📈 Visualizations")

    # 1. Average seconds per song with sampling
    st.subheader("⏱️ Average Seconds per Song Distribution")

# Sampling جزء بسيط للتسريع
    sampled_data = df_cleaned_featured['avg_secs_per_song'].sample(5000, random_state=42)

    fig1, ax1 = plt.subplots(figsize=(6, 3))
    sns.histplot(sampled_data, bins=50, ax=ax1, color='orange', kde=True)

# تحديد حدود المحور السيني (X-axis) لحد 3000 ثانية
    ax1.set_xlim(0, 3000)

    ax1.set_title("Distribution of Average Seconds per Song", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Average Seconds per Song", fontsize=9)
    ax1.set_ylabel("Count", fontsize=9)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.tick_params(axis='x', labelsize=6)
    ax1.tick_params(axis='y', labelsize=6)

    st.pyplot(fig1)

    col1, col2 = st.columns(2)

    with col1:
      st.subheader("🎧 High Engagement vs. Churn")
      high_engagement_counts = df_cleaned_featured['high_engagement'].value_counts(normalize=True)
      fig2, ax2 = plt.subplots(figsize=(5, 3))
      sns.barplot(x=high_engagement_counts.index, y=high_engagement_counts.values, ax=ax2, palette='Blues')
      ax2.set_xticklabels(['Low', 'High'])
      ax2.set_title("Churn Rate by Engagement", fontsize=14, fontweight='bold')
      ax2.set_xlabel("Engagement Level", fontsize=12)
      ax2.set_ylabel("Proportion", fontsize=12)
      ax2.grid(axis='y', linestyle='--', alpha=0.5)
      st.pyplot(fig2)

    with col2:
      st.subheader("💳 Payment Method Risk vs. Churn")
      fig3, ax3 = plt.subplots(figsize=(6, 3))
      sns.barplot(x='payment_method_risk', y='is_churn', data=df_cleaned_featured, ax=ax3, palette='coolwarm')
      ax3.set_title("Churn Rate by Payment Method Risk", fontsize=14, fontweight='bold')
      ax3.set_xlabel("Payment Risk Level", fontsize=12)
      ax3.set_ylabel("Churn Rate", fontsize=12)
      ax3.grid(axis='y', linestyle='--', alpha=0.5)
      st.pyplot(fig3)


    # 4. Discount Rate Distribution with fewer bins for performance
    col1, col2 = st.columns(2)

    with col1:
     st.subheader("💸 Discount Rate Distribution")
     fig4, ax4 = plt.subplots(figsize=(6, 4))  # خليت الارتفاع 4 عشان الشكل النهائي يبقى مناسب
     sns.histplot(df_cleaned_featured['discount_rate'], bins=30, ax=ax4, color='green', kde=True)
     ax4.set_title("Distribution of Discount Rate", fontsize=14, fontweight='bold')
     ax4.set_xlabel("Discount Rate", fontsize=12)
     ax4.set_ylabel("Count", fontsize=12)
     ax4.grid(axis='y', linestyle='--', alpha=0.5)
     st.pyplot(fig4)

    with col2:
     st.subheader("🆓 Trial User vs. Churn")
     trial_churn_counts = df_cleaned_featured.groupby('is_trial')['is_churn'].mean()
     fig5, ax5 = plt.subplots(figsize=(5, 4))  # برضو ارتفاع 4
     sns.barplot(x=trial_churn_counts.index, y=trial_churn_counts.values, ax=ax5, palette='magma')
     ax5.set_xticklabels(['Paid', 'Trial'])
     ax5.set_title("Churn Rate by Trial Users", fontsize=14, fontweight='bold')
     ax5.set_xlabel("User Type", fontsize=12)
     ax5.set_ylabel("Churn Rate", fontsize=12)
     ax5.grid(axis='y', linestyle='--', alpha=0.5)
     st.pyplot(fig5)

    # 6. Account Age Distribution (Only show a subset of data)
    st.subheader("📅 Account Age Distribution")
    fig6, ax6 = plt.subplots(figsize=(6, 3))
    sns.histplot(df_cleaned_featured['account_age_days'], bins=30, ax=ax6, color='purple', kde=True)
    ax6.set_title("Distribution of Account Age (Days)")
    st.pyplot(fig6)

    # KPIs and Percentages
    st.markdown("### 📊 Percentages")
    churn_rate = df_cleaned_featured['is_churn'].mean()
    trial_users_percentage = (df_cleaned_featured['is_trial'].mean() * 100)
    no_txn_and_churn_percentage = (df_cleaned_featured['no_txn_and_churn'].mean() * 100)
    high_engagement_percentage = (df_cleaned_featured['high_engagement'].mean() * 100)
    discount_users_percentage = (df_cleaned_featured['discount_rate'] > 0).mean() * 100

    kpis = {
        "Churn Rate": f"{churn_rate:.2%}",
        "Trial Users Percentage": f"{trial_users_percentage:.2f}%",
        "No Transaction & Churn Percentage": f"{no_txn_and_churn_percentage:.2f}%",
        "High Engagement Percentage": f"{high_engagement_percentage:.2f}%",
        "Users with Discount Percentage": f"{discount_users_percentage:.2f}%"
    }

    for kpi, value in kpis.items():
        st.markdown(f"**{kpi}:** {value}")

    
   
   
   
   
    # Add code for model comparison here
elif st.session_state.page == "Model Comparison":
    st.title("🤖 Model Comparison")
    st.markdown("Compare the performance of different models.")
   
    model_comparison_results = pd.DataFrame({
    'Model': ['Random Forest', 'Logistic Regression', 'Decision Tree', 'Naive Bayes'],
    'Accuracy': [0.960526, 0.912284, 0.963394, 0.947808],
    'Precision': [0.790479, 0.509234, 0.824695, 0.800118],
    'Recall': [0.763462, 0.681719, 0.753068, 0.559457],
    'F1-Score': [0.776736, 0.582986, 0.787256, 0.658488],
    'AUC': [0.943019, 0.867435, 0.928002, 0.903211]
    })

# دالة لعرض البار بلوت
    def plot_metric(metric, color):
      fig, ax = plt.subplots(figsize=(5.5, 4))
      sns.barplot(x='Model', y=metric, data=model_comparison_results, palette=color, ax=ax)
      ax.set_title(f"{metric} Comparison", fontsize=13)
      ax.set_ylabel(metric)
      ax.set_ylim(0, 1.05)
      ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
      for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., height + 0.015,
                f'{height:.2f}', ha="center", fontsize=9)
      st.pyplot(fig)

   # الصفحة
    st.subheader("📊 Model Comparison (Side-by-Side)")

# صف 1: AUC و F1-Score
    cols1 = st.columns(2)
    with cols1[0]:
     plot_metric('AUC', 'Purples')
    with cols1[1]:
     plot_metric('F1-Score', 'Greens')

# صف 2: Accuracy و Precision
    cols2 = st.columns(2)
    with cols2[0]:
     plot_metric('Accuracy', 'Blues')
    with cols2[1]:
     plot_metric('Precision', 'Oranges')

# صف 3: Recall فقط
    def plot_metric(metric, color, fig_size=(4, 3)):
     fig, ax = plt.subplots(figsize=fig_size)
     sns.barplot(x='Model', y=metric, data=model_comparison_results, palette=color, ax=ax)
     ax.set_title(f"{metric} Comparison", fontsize=7)
     ax.set_ylabel(metric)
     ax.set_ylim(0, 1)
     ax.set_xticklabels(ax.get_xticklabels(), rotation=15, fontsize=5) 
     for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2., height + 0.015,
                f'{height:.2f}', ha="center", fontsize=7)
     st.pyplot(fig)
    cols3 = st.columns(1)
    with cols3[0]:
     plot_metric('Recall', 'BuGn', fig_size=(3, 2))  # حجم أصغر
 


  # Choose the model for detailed evaluation
    st.subheader("🔎 Select a Model to Analyze")
    prediction_roc_path = r"D:\DEPI Graduation Project\project\all_predictions.csv"
    df_pred=pd.read_csv(prediction_roc_path)
    
    # Define model_columns mapping (move this to a global scope for reuse)
    model_columns = {
        "Random Forest": "Predicted_Probability_RF",
        "Logistic Regression": "Predicted_Probability_LR",
        "Decision Tree": "Predicted_Probability_DT",
        "Naive Bayes": "Predicted_Probability_GNB"
    }

# اختيار الموديل
    selected_model = st.selectbox("Select Model", list(model_columns.keys()))
 
# استخراج القيم الحقيقية وقيم الاحتمال
    y_true = df_pred["True_Label"]
# Ensure model_columns is accessible here
    if selected_model in model_columns and model_columns[selected_model] in df_pred.columns:
      y_scores = df_pred[model_columns[selected_model]]
    else:
      st.error(f"خطأ: العمود '{model_columns[selected_model]}' غير موجود في البيانات.")
      st.stop()

# حساب منحنى ROC و AUC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

# رسم ROC باستخدام Plotly
    fig = go.Figure()

# منحنى ROC
    fig.add_trace(go.Scatter(
    x=fpr, y=tpr,
    mode='lines',
    name='ROC Curve',
    line=dict(color='royalblue', width=3)
    ))

# الخط العشوائي
    fig.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    mode='lines',
    name='Random Guess',
    line=dict(color='red', width=2, dash='dash')
    ))

# عرض AUC داخل الرسم
    fig.add_annotation(
    x=0.6,
    y=0.1,
    text=f"AUC = {roc_auc:.2f}",
    showarrow=False,
    font=dict(size=16, color="black"),
    bgcolor="lightyellow",
    bordercolor="black",
    borderwidth=1
)

# تخصيص شكل الرسم
    fig.update_layout(
    title=dict(text=f"<b>ROC Curve - {selected_model}</b>", x=0.5),
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    width=550,
    height=600,
    xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
    yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
    template='plotly_white',
    legend=dict(x=0.7, y=0.3)
    )

# عرض الرسم
    st.plotly_chart(fig, use_container_width=False)






elif st.session_state.page == "Predict Churn":
    st.title("🔮 Predict Churn")
    st.markdown("Use the model to predict customer churn.")
    # Add prediction functionality here

    model_path = r"D:\DEPI Graduation Project\project\random_forest_model.pkl"
    with open(model_path, 'rb') as file:
     rf_model = pickle.load(file)

#   File uploader UI in English
    uploaded_file = st.file_uploader("📁 Upload customer data file (CSV)", type=["csv"])
    if uploaded_file is not None:
     df_test = pd.read_csv(uploaded_file)
    else:
     st.write("Please select a page to explore.")
    # Load the model
    st.title("🎧 Customer Churn Prediction Dashboard")
    st.markdown("Predict churn probability for users in a music streaming service using a trained Random Forest model.")

    # Step 1: Load final submission data
    @st.cache_data
    def load_data():
     return pd.read_csv(r"D:/DEPI Graduation Project/submission_Random_Forest.csv")

    data = load_data()

   # Step 2: Key Metrics
    st.subheader("📌 Key Metrics")
    churned = (data['is_churn'] >= 0.5).sum()
    non_churned = (data['is_churn'] < 0.5).sum()
    total_users = len(data)

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Churners", f"{churned}")
    col2.metric("Non-Churners", f"{non_churned}")
    col3.metric("Churn Rate", f"{(churned / total_users) * 100:.2f}%")

# Step 3: Distribution Plot
    st.subheader("📊 Churn Probability Distribution")
    fig, ax = plt.subplots(figsize=(3,2))  # صغرت الفيجور هنا
    sns.histplot(data['is_churn'], bins=50, kde=True, ax=ax)
    ax.set_title("Distribution of Predicted Churn Probabilities", fontsize=10)
    ax.set_xlabel("Churn Probability", fontsize=8)
    ax.set_ylabel("Number of Users", fontsize=8)
    ax.tick_params(axis='x', labelsize=5)
    ax.tick_params(axis='y', labelsize=5)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    st.pyplot(fig)

   # Step 4: Threshold-based Analysis
    st.subheader("📈 High-Risk Churn Segments")
    thresholds = [0.5, 0.7, 0.9]
    for threshold in thresholds:
     count = (data['is_churn'] >= threshold).sum()
     pct = (count / total_users) * 100
    st.write(f"Users with churn probability ≥ {threshold}: {count} ({pct:.2f}%)")

# Step 5: Show top churners
    st.subheader("🔍 Top 10 Most Likely to Churn")
    st.dataframe(data.sort_values('is_churn', ascending=False).head(10))

# Step 6: Download submission file
    st.download_button(
    label="📥 Download Churn Predictions CSV",
    data=data.to_csv(index=False).encode('utf-8'),
    file_name='churn_predictions.csv',
    mime='text/csv'
     )
