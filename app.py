import textwrap
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split


st.set_page_config(
    page_title="Breast Tumour Classification",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


INK = "#ECE8F5"
MUTED = "#B3ABC7"
PAPER = "#14101F"
CARD = "#1E1830"
LINE = "#332B4A"
HEM = "#7C58C9"
HEM_DARK = "#5B3BA0"
HEM_LIGHT = "#B79CF0"
SIDEBAR = "#0E0B17"
HERO = "#2A1B47"
MALIGNANT = "#E0679A"
BENIGN = "#3FA6BA"
CLASS_COLORS = {"Malignant": MALIGNANT, "Benign": BENIGN}


def render_html(html):
    st.markdown(textwrap.dedent(html).strip(), unsafe_allow_html=True)


render_html(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&display=swap');

:root {{ color-scheme: dark; }}
html, body, [class*="st-"], .stApp {{
    font-family: 'IBM Plex Sans', system-ui, sans-serif;
}}
.stApp {{ color: {INK}; }}
.stApp, [data-testid="stAppViewContainer"] {{ background: {PAPER}; }}
[data-testid="stHeader"] {{ background: {PAPER}; }}
[data-testid="stHeader"] *, [data-testid="stToolbar"] *, [data-testid="stStatusWidget"] * {{ color: {INK} !important; }}
[data-testid="stDeployButton"], .stDeployButton {{ display: none !important; }}
#MainMenu, footer {{ visibility: hidden; }}

.block-container {{
    max-width: 1120px;
    padding-top: 2.2rem;
    padding-bottom: 4rem;
}}

[data-testid="stMain"] h1, [data-testid="stMain"] h2, [data-testid="stMain"] h3 {{
    font-family: 'Source Serif 4', Georgia, serif;
    color: {INK};
    letter-spacing: -0.01em;
}}
[data-testid="stMain"] h1 {{ font-size: 2.2rem; font-weight: 600; }}
[data-testid="stMain"] h2 {{ font-size: 1.55rem; font-weight: 600; margin-top: 1.6rem; }}
[data-testid="stMain"] h3 {{ font-size: 1.2rem; font-weight: 600; }}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {{
    color: {MUTED};
    line-height: 1.7;
    font-size: 1rem;
    max-width: 75ch;
}}
[data-testid="stMarkdownContainer"] strong {{ color: {INK}; }}
[data-testid="stCaptionContainer"] p {{ color: {MUTED}; font-size: 0.85rem; }}
hr {{ border-color: {LINE}; margin: 1.6rem 0; }}

[data-testid="stSidebar"] {{ background: {SIDEBAR}; border-right: 1px solid {LINE}; }}
[data-testid="stSidebar"] * {{ color: #EFE9F7; }}
[data-testid="stSidebar"] h1 {{
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.35rem;
    color: #FFFFFF;
}}
[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.14); }}
[data-testid="stSidebar"] [data-testid="stRadio"] label {{
    padding: 8px 12px;
    border-radius: 8px;
    width: 100%;
    transition: background 0.15s;
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
    background: rgba(255,255,255,0.08);
}}
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {{
    background: rgba(255,255,255,0.16);
}}
.side-note {{
    font-size: 0.85rem;
    line-height: 1.6;
    color: #CFC3E3 !important;
}}
.side-note b {{ color: #FFFFFF !important; }}

.hero {{
    background: {HERO};
    background-image:
        radial-gradient(circle at 92% 18%, rgba(194,69,122,0.45) 0, rgba(194,69,122,0) 38%),
        radial-gradient(circle at 80% 90%, rgba(43,122,140,0.40) 0, rgba(43,122,140,0) 36%);
    padding: 44px 44px 36px 44px;
    border-radius: 14px;
    margin-bottom: 22px;
}}
.hero h1 {{
    font-family: 'Source Serif 4', Georgia, serif;
    color: #FFFFFF !important;
    font-size: 2.3rem !important;
    line-height: 1.2;
    margin: 0 0 12px 0;
    max-width: 22ch;
}}
.hero p {{
    color: #E3D9F1 !important;
    font-size: 1.05rem;
    line-height: 1.7;
    max-width: 58ch;
    margin: 0 0 26px 0;
}}
.hero-stats {{ display: flex; gap: 40px; flex-wrap: wrap; }}
.hero-stats div {{ display: flex; flex-direction: column; }}
.hero-stats b {{ color: #FFFFFF; font-size: 1.6rem; font-weight: 600; }}
.hero-stats span {{ color: #CFC3E3; font-size: 0.85rem; }}

.notice {{
    background: #2B2212;
    border-left: 4px solid #D69A3B;
    padding: 12px 16px;
    border-radius: 6px;
    margin-bottom: 22px;
    color: #F0D9A8;
    font-size: 0.92rem;
    line-height: 1.6;
}}
.notice b {{ color: #FFE9BD; }}

.card {{
    background: {CARD};
    border: 1px solid {LINE};
    border-radius: 10px;
    padding: 22px 24px;
    height: 100%;
}}
.card h3 {{ margin: 0 0 8px 0 !important; font-size: 1.1rem !important; color: {HEM_LIGHT} !important; }}
.card p {{ margin: 0; color: {MUTED}; line-height: 1.65; font-size: 0.97rem; }}

.result {{
    background: {CARD};
    border: 1px solid {LINE};
    border-radius: 12px;
    padding: 26px 28px;
    margin: 8px 0 16px 0;
}}
.result .tag {{ color: {MUTED}; font-size: 0.85rem; margin-bottom: 4px; }}
.result .verdict {{
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 2rem;
    font-weight: 600;
    margin-bottom: 18px;
}}
.bar {{
    display: flex;
    height: 14px;
    border-radius: 7px;
    overflow: hidden;
    background: {LINE};
}}
.bar .m {{ background: {MALIGNANT}; }}
.bar .b {{ background: {BENIGN}; }}
.bar-legend {{
    display: flex;
    justify-content: space-between;
    margin-top: 10px;
    font-size: 0.92rem;
    color: {INK};
}}
.bar-legend .m b {{ color: {MALIGNANT}; }}
.bar-legend .b b {{ color: {BENIGN}; }}

[data-testid="stMetric"] {{
    background: {CARD};
    border: 1px solid {LINE};
    border-radius: 10px;
    padding: 16px 18px;
}}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-size: 0.88rem; }}
[data-testid="stMetricValue"] {{ color: {INK}; font-family: 'Source Serif 4', Georgia, serif; }}

.stButton button, .stFormSubmitButton button {{
    background: {HEM};
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 500;
}}
.stButton button p, .stFormSubmitButton button p {{ color: #FFFFFF; }}
.stButton button:hover, .stFormSubmitButton button:hover {{ background: {HEM_DARK}; color: #FFFFFF; }}
.stButton button:focus-visible, .stFormSubmitButton button:focus-visible {{
    outline: 3px solid {MALIGNANT};
    outline-offset: 2px;
}}

[data-testid="stForm"] {{
    background: {CARD};
    border: 1px solid {LINE};
    border-radius: 12px;
    padding: 20px 24px;
}}
[data-testid="stWidgetLabel"] p {{ color: {INK}; font-size: 0.9rem; }}

.stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ color: {MUTED}; padding: 8px 14px; }}
.stTabs [aria-selected="true"] {{ color: {HEM_LIGHT}; font-weight: 600; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {HEM_LIGHT}; }}

[data-testid="stDataFrame"] {{
    border: 1px solid {LINE};
    border-radius: 10px;
    overflow: hidden;
}}
[data-testid="stPlotlyChart"] {{
    background: {CARD};
    border: 1px solid {LINE};
    border-radius: 10px;
    padding: 6px;
}}
[data-testid="stExpander"] {{
    background: {CARD};
    border: 1px solid {LINE};
    border-radius: 10px;
}}

[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="base-input"] {{
    background: {PAPER} !important;
    border-color: {LINE} !important;
}}
[data-baseweb="input"] input, [data-baseweb="select"] * {{ color: {INK} !important; }}
[data-testid="stAlert"] p {{ color: {INK}; }}
[data-testid="stExpander"] summary p {{ color: {INK}; }}

[data-testid="stHeaderActionElements"] {{
    display: none !important;
}}

[data-testid="stIconMaterial"],
[data-testid="stExpanderToggleIcon"],
[data-testid="stSidebarCollapseButton"] span,
[data-testid="stSidebarCollapsedControl"] span,
[data-testid="stExpandSidebarButton"] span,
.material-icons, .material-symbols-rounded, .material-symbols-outlined {{
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
    font-feature-settings: 'liga' 1;
    letter-spacing: normal;
}}
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="stExpandSidebarButton"] {{
    color: {HEM_LIGHT} !important;
    background: transparent !important;
    border-radius: 8px;
}}
[data-testid="stSidebarCollapseButton"] button:hover,
[data-testid="stSidebarCollapsedControl"] button:hover,
[data-testid="stExpandSidebarButton"]:hover {{
    background: rgba(183,156,240,0.14) !important;
}}
[data-testid="stSidebarCollapseButton"] span,
[data-testid="stSidebarCollapsedControl"] span,
[data-testid="stExpandSidebarButton"] span {{ color: {HEM_LIGHT} !important; }}

* {{
    scrollbar-width: none;
    -ms-overflow-style: none;
}}
*::-webkit-scrollbar {{
    display: none;
}}

.footer {{
    text-align: center;
    color: {MUTED};
    font-size: 0.85rem;
    padding: 12px 0 0 0;
}}
</style>
""")


def style_chart(fig, height=None):
    fig.update_layout(
        font=dict(family="IBM Plex Sans, Arial", size=13, color=INK),
        title=dict(font=dict(family="Source Serif 4, Georgia", size=18, color=INK), x=0.02),
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        margin=dict(l=40, r=30, t=60, b=40),
        legend=dict(title_text="", orientation="h", y=-0.2),
    )
    fig.update_xaxes(gridcolor=LINE, zeroline=False, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=LINE, zeroline=False, tickfont=dict(color=MUTED))
    if height:
        fig.update_layout(height=height)
    return fig


def show_chart(fig, height=None):
    st.plotly_chart(style_chart(fig, height), use_container_width=True)


def notice(text):
    render_html(f'<div class="notice">{text}</div>')


def result_card(prediction, malignant_pct, benign_pct):
    label = "Malignant" if prediction == 0 else "Benign"
    colour = MALIGNANT if prediction == 0 else BENIGN
    render_html(f"""
    <div class="result">
    <div class="tag">Model classification</div>
    <div class="verdict" style="color:{colour}">{label}</div>
    <div class="bar">
    <div class="m" style="width:{malignant_pct:.2f}%"></div>
    <div class="b" style="width:{benign_pct:.2f}%"></div>
    </div>
    <div class="bar-legend">
    <span class="m">Malignant <b>{malignant_pct:.1f}%</b></span>
    <span class="b">Benign <b>{benign_pct:.1f}%</b></span>
    </div>
    </div>
    """)


def card(title, body):
    return f'<div class="card"><h3>{title}</h3><p>{body}</p></div>'


@st.cache_data
def load_dataset():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y


MODEL_PATH = Path(__file__).parent / "models" / "mediclass_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def train_random_forest(_X_train, _y_train):
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(_X_train, _y_train)
    return rf


X, y = load_dataset()

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found. Expected it at models/mediclass_model.pkl.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(
    y_test, y_pred, target_names=["Malignant", "Benign"], output_dict=True
)
malignant_recall = report["Malignant"]["recall"]
roc_auc = roc_auc_score((y_test == 0).astype(int), model.predict_proba(X_test)[:, 0])


st.sidebar.title("🧬 Tumour Classification")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Predict", "Dataset", "Feature insights", "Model performance"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
<div class="side-note">
<b>Dataset</b><br>
569 records · 30 features<br>
Target: 0 = Malignant, 1 = Benign
<br><br>
<b>Educational use only.</b><br>
Not clinically validated. Do not use for medical decisions.
</div>
""",
    unsafe_allow_html=True,
)


if page == "Overview":
    render_html("""
    <div class="hero">
    <h1>Classifying tumours from diagnostic measurements</h1>
    <p>An interactive look at how machine learning separates malignant from benign
    samples in the Breast Cancer Wisconsin Diagnostic dataset.</p>
    <div class="hero-stats">
    <div><b>569</b><span>records</span></div>
    <div><b>30</b><span>numeric features</span></div>
    <div><b>2</b><span>classes</span></div>
    </div>
    </div>
    """)

    notice(
        "<b>Educational project.</b> The model was trained on 569 records and has not been "
        "clinically validated. Its outputs are not medical advice or risk estimates."
    )

    st.header("What you can do here")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        render_html(card("Predict", "Enter 30 measurements by hand, or pick a held-out test sample and compare the model's answer with the real label."))
    with c2:
        render_html(card("Explore the dataset", "See the class balance, how features correlate, and how each feature differs between malignant and benign samples."))
    st.markdown("")
    c3, c4 = st.columns(2, gap="medium")
    with c3:
        render_html(card("Read the model", "Inspect which features push the Logistic Regression model toward each class."))
    with c4:
        render_html(card("Compare models", "Logistic Regression against Random Forest, on a held-out split and with 5-fold cross-validation."))

    st.header("Models used")
    st.markdown("""
    **Logistic Regression** estimates the probability that a sample belongs to each class
    from standardised features. It is the model behind the Predict page.

    **Random Forest** combines the votes of many decision trees. It is used as a
    comparison on the Model performance page.

    Both are evaluated with a stratified 80/20 train-test split and stratified 5-fold cross-validation.
    """)


elif page == "Predict":
    st.title("Predict")
    st.write("See how the trained Logistic Regression model classifies a set of measurements.")
    notice("Outputs are model-generated probabilities, not clinical risk estimates.")

    input_method = st.radio(
        "Input method",
        ["Manual input", "Test dataset sample"],
        horizontal=True,
    )

    if input_method == "Manual input":
        st.write("Defaults are dataset medians, provided as a starting point.")

        sections = {
            "Mean": [f for f in X.columns if f.startswith("mean ")],
            "Standard error": [f for f in X.columns if f.endswith("error")],
            "Worst": [f for f in X.columns if f.startswith("worst ")],
        }

        user_input = {}
        with st.form("manual_prediction_form"):
            tabs = st.tabs(list(sections.keys()))
            for tab, (section, features) in zip(tabs, sections.items()):
                with tab:
                    cols = st.columns(2)
                    for i, feature in enumerate(features):
                        with cols[i % 2]:
                            user_input[feature] = st.number_input(
                                feature.capitalize(),
                                min_value=0.0,
                                value=float(X[feature].median()),
                                step=0.01,
                                format="%.4f",
                                key=f"manual_{feature}",
                            )
            submitted = st.form_submit_button("Classify", use_container_width=True)

        if submitted:
            input_df = pd.DataFrame([user_input], columns=X.columns)
            prediction = model.predict(input_df)[0]
            probs = model.predict_proba(input_df)[0]
            result_card(prediction, probs[0] * 100, probs[1] * 100)
            st.caption("Model output only. Not a clinical probability or medical risk estimate.")

    else:
        sample_number = st.selectbox(
            "Test sample",
            range(len(X_test)),
            format_func=lambda i: f"Test sample {i + 1}",
        )
        sample = X_test.iloc[[sample_number]]
        actual = y_test.iloc[sample_number]
        prediction = model.predict(sample)[0]
        probs = model.predict_proba(sample)[0]

        result_card(prediction, probs[0] * 100, probs[1] * 100)

        c1, c2 = st.columns(2)
        c1.metric("Model classification", "Malignant" if prediction == 0 else "Benign")
        c2.metric("Actual dataset label", "Malignant" if actual == 0 else "Benign")

        if prediction == actual:
            st.success("The model classified this sample correctly.")
        else:
            st.error("The model misclassified this sample.")

        with st.expander("Show the 30 measurements for this sample"):
            st.dataframe(
                sample.T.rename(columns={sample.index[0]: "Value"}),
                use_container_width=True,
            )


elif page == "Dataset":
    st.title("Dataset")
    st.write("Structure, class balance and relationships in the Wisconsin Diagnostic Breast Cancer dataset.")

    st.header("Class balance")
    counts = y.value_counts().sort_index()
    class_df = pd.DataFrame({
        "Category": ["Malignant", "Benign"],
        "Count": [counts.get(0, 0), counts.get(1, 0)],
    })
    c1, c2 = st.columns([3, 2], gap="medium")
    with c1:
        fig = px.pie(
            class_df, names="Category", values="Count", hole=0.55,
            color="Category", color_discrete_map=CLASS_COLORS,
            title="Target classes",
        )
        fig.update_traces(textinfo="label+percent", textfont_color="#FFFFFF")
        fig.update_layout(showlegend=False)
        show_chart(fig, 360)
    with c2:
        st.dataframe(class_df, use_container_width=True, hide_index=True)
        st.caption("Encoding: 0 = Malignant, 1 = Benign.")

    st.header("Feature correlations")
    st.write(
        "Values near +1 mean two features rise together; values near −1 mean one rises as the other falls. "
        "Many size features are strongly correlated with each other."
    )
    corr = X.corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale=[[0, MALIGNANT], [0.5, CARD], [1, BENIGN]],
        zmin=-1, zmax=1,
        colorbar=dict(title="r"),
    ))
    fig.update_layout(title="Correlation between all 30 features")
    fig.update_xaxes(tickangle=-90, tickfont=dict(size=9))
    fig.update_yaxes(tickfont=dict(size=9), autorange="reversed")
    show_chart(fig, 760)

    st.header("Feature by class")
    selected = st.selectbox("Feature", X.columns)
    plot_df = X.copy()
    plot_df["Diagnosis"] = y.map({0: "Malignant", 1: "Benign"})
    fig = px.box(
        plot_df, x="Diagnosis", y=selected, color="Diagnosis",
        color_discrete_map=CLASS_COLORS, title=selected.capitalize(),
    )
    fig.update_layout(showlegend=False, xaxis_title="")
    show_chart(fig, 420)
    st.caption("Box plots show the median, spread and outliers of one feature for each class.")


elif page == "Feature insights":
    st.title("Feature insights")
    st.write("Logistic Regression coefficients show which standardised features push the model toward each class.")

    classifier = model.named_steps["classifier"]
    coefs = classifier.coef_[0]
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": coefs,
        "Importance": np.abs(coefs),
    }).sort_values("Importance", ascending=False)

    top = importance_df.head(10).sort_values("Coefficient").copy()
    top["Pushes toward"] = np.where(top["Coefficient"] > 0, "Benign", "Malignant")

    fig = px.bar(
        top, x="Coefficient", y="Feature", orientation="h",
        color="Pushes toward", color_discrete_map=CLASS_COLORS,
        title="Ten most influential features",
    )
    fig.update_layout(yaxis_title="", xaxis_title="Coefficient (standardised scale)")
    show_chart(fig, 520)

    st.header("How to read this")
    st.markdown("""
    - **Positive coefficient:** pushes the output toward Benign (label 1).
    - **Negative coefficient:** pushes the output toward Malignant (label 0).
    - **Larger absolute value:** a stronger association on the standardised scale.
    """)
    st.info(
        "Coefficients describe what this model learned. They do not show causation or medical "
        "significance, and correlated features can share or split influence."
    )

    with st.expander("Full coefficient table"):
        st.dataframe(importance_df.reset_index(drop=True), use_container_width=True, hide_index=True)


elif page == "Model performance":
    st.title("Model performance")
    st.write("Logistic Regression and Random Forest on a held-out test set, then with cross-validation.")

    rf_model = train_random_forest(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_report = classification_report(
        y_test, rf_pred, target_names=["Malignant", "Benign"], output_dict=True
    )

    comparison_df = pd.DataFrame({
        "Metric": ["Accuracy", "Malignant recall", "Macro F1"],
        "Logistic Regression": [accuracy, report["Malignant"]["recall"], report["macro avg"]["f1-score"]],
        "Random Forest": [rf_accuracy, rf_report["Malignant"]["recall"], rf_report["macro avg"]["f1-score"]],
    })

    st.header("Model comparison")
    plot = comparison_df.melt(id_vars="Metric", var_name="Model", value_name="Score")
    fig = px.bar(
        plot, x="Metric", y="Score", color="Model", barmode="group",
        text_auto=".3f", title="Logistic Regression vs Random Forest",
        color_discrete_map={"Logistic Regression": HEM_LIGHT, "Random Forest": "#E8A33D"},
    )
    fig.update_layout(yaxis_range=[0, 1.05], xaxis_title="")
    show_chart(fig, 400)
    st.dataframe(comparison_df.set_index("Metric").style.format("{:.4f}"), use_container_width=True)

    st.header("Logistic Regression on the test set")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{accuracy * 100:.2f}%")
    c2.metric("Malignant recall", f"{malignant_recall * 100:.2f}%")
    c3.metric("ROC-AUC", f"{roc_auc:.4f}")
    st.caption("Results from one held-out split. They are not evidence of clinical validity.")

    st.header("Confusion matrix")
    c1, c2 = st.columns([3, 2], gap="medium")
    with c1:
        fig = px.imshow(
            cm, text_auto=True,
            color_continuous_scale=[[0, CARD], [1, HEM_LIGHT]],
            x=["Predicted malignant", "Predicted benign"],
            y=["Actual malignant", "Actual benign"],
            title="Logistic Regression",
        )
        fig.update_coloraxes(showscale=False)
        show_chart(fig, 400)
    with c2:
        st.markdown("""
        - **Top left:** malignant correctly found.
        - **Top right:** malignant missed (false negative).
        - **Bottom left:** benign flagged as malignant (false positive).
        - **Bottom right:** benign correctly found.
        """)

    st.header("Classification report")
    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

    st.header("5-fold cross-validation")
    st.write("Cross-validation repeats training and testing on five different splits, keeping the class ratio in each fold.")

    if st.button("Run cross-validation"):
        with st.spinner("Evaluating both models across 5 folds…"):
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            lr_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
            rf_scores = cross_val_score(
                RandomForestClassifier(n_estimators=300, random_state=42),
                X, y, cv=cv, scoring="accuracy",
            )

        cv_df = pd.DataFrame({
            "Fold": [f"Fold {i + 1}" for i in range(5)],
            "Logistic Regression": lr_scores,
            "Random Forest": rf_scores,
        })

        c1, c2 = st.columns(2)
        c1.metric("Logistic Regression mean accuracy", f"{lr_scores.mean() * 100:.2f}%",
                  help=f"Standard deviation {lr_scores.std():.4f}")
        c2.metric("Random Forest mean accuracy", f"{rf_scores.mean() * 100:.2f}%",
                  help=f"Standard deviation {rf_scores.std():.4f}")

        cv_plot = cv_df.melt(id_vars="Fold", var_name="Model", value_name="Accuracy")
        fig = px.line(
            cv_plot, x="Fold", y="Accuracy", color="Model", markers=True,
            title="Accuracy per fold",
            color_discrete_map={"Logistic Regression": HEM_LIGHT, "Random Forest": "#E8A33D"},
        )
        fig.update_layout(yaxis_range=[0.8, 1.01], xaxis_title="")
        show_chart(fig, 380)
        st.dataframe(cv_df.set_index("Fold").style.format("{:.4f}"), use_container_width=True)

    with st.expander("Model limitations"):
        st.write(
            "The model was trained on 569 records from one public dataset. The metrics describe "
            "this dataset and setup only. There has been no independent clinical validation, "
            "prospective testing or regulatory approval, and high accuracy does not establish "
            "medical reliability. Do not use this application for diagnosis or treatment decisions."
        )


st.markdown("---")
render_html("""
<div class="footer">
Breast Tumour Classification · Educational machine learning project<br>
Python, scikit-learn, Streamlit, Plotly
</div>
""")