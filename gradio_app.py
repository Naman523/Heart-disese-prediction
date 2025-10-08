# =============================================
# 💗 Heart Disease Prediction Dashboard (Gradio)
# =============================================
import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# -----------------------------------
# 📥 Load Data & Train Model
# -----------------------------------

# Load dataset (make sure the file is in the same folder)
df = pd.read_csv("heart_disease_data.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Build pipeline model
model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])

model.fit(X, y)

# -----------------------------------
# 🩺 Prediction Function
# -----------------------------------

def predict_risk(age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                 exang, oldpeak, slope, ca, thal):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    pred_prob = model.predict_proba(input_data)[0][1]
    pred_class = model.predict(input_data)[0]

    # Assign Risk Level
    if pred_prob < 0.4:
        risk = "🟢 Low Risk"
    elif pred_prob < 0.7:
        risk = "🟡 Moderate Risk"
    else:
        risk = "🔴 High Risk"

    return {
        "Prediction (0=No, 1=Yes)": int(pred_class),
        "Risk Probability": round(pred_prob, 3),
        "Risk Level": risk
    }

# -----------------------------------
# 📊 Feature Importance Chart
# -----------------------------------

def feature_importance_chart():
    try:
        clf = model.named_steps.get("clf", model)
        feature_names = X.columns

        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            importances = abs(clf.coef_[0])
        else:
            raise AttributeError("This model does not provide feature importances or coefficients.")

        feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=True).tail(15)

        plt.figure(figsize=(8,6))
        feat_imp.plot(kind="barh", color="#60a5fa")
        plt.title("Top 15 Important Features", color="white", fontsize=13)
        plt.xlabel("Importance", color="white")
        plt.gca().set_facecolor("#1e293b")
        plt.gcf().patch.set_facecolor("#0f172a")
        plt.tick_params(colors="white")
        plt.tight_layout()
        return plt.gcf()

    except Exception as e:
        plt.figure(figsize=(6,3))
        plt.text(0.5, 0.5, f"⚠️ Error: {str(e)}", ha='center', va='center',
                 color='white', fontsize=12)
        plt.gca().set_facecolor("#1e293b")
        plt.gcf().patch.set_facecolor("#0f172a")
        plt.axis('off')
        return plt.gcf()

# -----------------------------------
# 🎨 Gradio Dashboard UI
# -----------------------------------

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="violet", secondary_hue="pink"),
    css="""
        body {background-color: #0f172a !important;}
        footer {text-align: center; margin-top: 20px; font-size: 14px;}
        .gradio-container {font-family: 'Poppins', sans-serif;}
        h1, h2, h3, h4 {text-align:center;}
    """
) as app:

    gr.Markdown("""
    <div style='text-align: center;'>
        <h1 style='color:#e879f9;'>💗 Heart Disease Prediction Dashboard</h1>
        <p style='color:#cbd5e1;'>
        Predict the likelihood of <b>heart disease</b> using medical parameters and AI models.
        </p>
    </div>
    """)

    with gr.Tabs():

        # --- Predict Risk Tab ---
        with gr.TabItem("💬 Predict Risk"):
            with gr.Row():
                with gr.Column(scale=1):
                    age = gr.Slider(20, 80, 50, label="Age")
                    sex = gr.Radio([0, 1], label="Sex (0=Female, 1=Male)")
                    cp = gr.Radio([0, 1, 2, 3], label="Chest Pain Type (cp)")
                    trestbps = gr.Slider(90, 200, 130, label="Resting Blood Pressure")
                    chol = gr.Slider(100, 600, 250, label="Serum Cholesterol (mg/dl)")
                    fbs = gr.Radio([0, 1], label="Fasting Blood Sugar > 120 mg/dl (1=True)")
                    restecg = gr.Radio([0, 1], label="Resting ECG Results")
                    thalach = gr.Slider(70, 210, 150, label="Maximum Heart Rate Achieved")
                    exang = gr.Radio([0, 1], label="Exercise Induced Angina (1=Yes)")
                    oldpeak = gr.Slider(0.0, 6.0, 1.0, label="ST Depression (oldpeak)")
                    slope = gr.Radio([0, 1, 2], label="Slope of Peak Exercise ST Segment")
                    ca = gr.Slider(0, 4, 0, step=1, label="Number of Major Vessels (ca)")
                    thal = gr.Radio([0, 1, 2, 3], label="Thalassemia (thal)")

                with gr.Column(scale=1):
                    output = gr.JSON(label="📊 Prediction Results")

            predict_btn = gr.Button("🚀 Predict Risk", variant="primary")
            predict_btn.click(
                predict_risk,
                inputs=[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                        exang, oldpeak, slope, ca, thal],
                outputs=output
            )

        # --- Feature Importance Tab ---
        with gr.TabItem("📈 Feature Importance"):
            gr.Markdown("View the most important features influencing predictions.")
            fi_button = gr.Button("🧠 Show Feature Importance", variant="secondary")
            fi_plot = gr.Plot(label="Feature Importance Chart")
            fi_button.click(feature_importance_chart, outputs=fi_plot)

    # --- Footer ---
    gr.Markdown("""
    <div style='text-align:center; color:#a1a1aa; margin-top:25px;'>
        Built with 💜 using <b>Gradio + Scikit-learn</b><br>
        <small>© 2025 HeartCare AI Dashboard</small>
    </div>
    """)

# -----------------------------------
# 🚀 Launch the App
# -----------------------------------
app.launch()
