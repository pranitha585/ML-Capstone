import os, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    r2_score, mean_squared_error, mean_absolute_error,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,
    BaggingClassifier, RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import VarianceThreshold

st.set_page_config(page_title="AeroLab | ML Showcase", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: radial-gradient(ellipse at 15% 0%, #172554 0%, #0b1120 38%, #080d18 100%); color: #e5e7eb; }
section[data-testid="stSidebar"] { background: #0b1220; border-right: 1px solid #26334a; }
h1,h2,h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.035em; }
.hero { padding: 1.5rem 1.7rem; border: 1px solid #293852; border-radius: 22px; background: linear-gradient(120deg, rgba(30,64,175,.28), rgba(14,165,233,.08) 55%, rgba(16,185,129,.08)); margin-bottom: 1.2rem; }
.eyebrow { text-transform: uppercase; letter-spacing: .16em; font-size: .72rem; color: #7dd3fc; font-weight: 700; }
.hero h1 { font-size: 2.55rem; margin: .35rem 0 .45rem 0; color: #f8fafc; }
.hero p { color: #b7c5d9; font-size: 1.02rem; max-width: 850px; }
.pill { display:inline-block; border:1px solid #36506f; color:#bae6fd; border-radius:999px; padding:5px 10px; font-size:.75rem; margin:4px 4px 0 0; }
div[data-testid="stMetric"] { background: rgba(15,23,42,.72); border: 1px solid #283750; border-radius: 16px; padding: 15px 17px; }
div[data-testid="stMetricLabel"] { color: #94a3b8; }
div[data-testid="stMetricValue"] { color: #f8fafc; }
.stButton > button { border-radius: 10px; border: 1px solid #38bdf8; background: linear-gradient(90deg,#0284c7,#2563eb); color:white; font-weight:700; }
.stDownloadButton > button { border-radius: 10px; }
div[data-testid="stExpander"] { border:1px solid #263750; border-radius:12px; }
.small-note { color:#9fb0c8; font-size:.88rem; }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("## ◈ AeroLab")
st.sidebar.caption("AIR QUALITY · MACHINE LEARNING")
page = st.sidebar.radio("WORKSPACE", ["Overview", "Classification", "Regression", "Clustering", "Data Explorer", "Setup & rubric"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Project tracks**")
st.sidebar.markdown("`01` AQI category prediction  \n`02` PM2.5 concentration  \n`03` Pollution-pattern clusters")
st.sidebar.caption("Local-first dashboard · your CSV files stay on your machine")

DATA_DIR = "data"
TRACK_DIRS = {
    "classification": os.path.join(DATA_DIR, "classification"),
    "regression": os.path.join(DATA_DIR, "regression"),
    "clustering": os.path.join(DATA_DIR, "clustering"),
}

def csv_files(folder):
    return sorted(glob.glob(os.path.join(folder, "*.csv")))


def load_track_data(track, uploads=None):
    """Load one CSV or combine selected CSV files for an ML workspace."""

    folder = TRACK_DIRS[track]
    os.makedirs(folder, exist_ok=True)

    # Find CSV files already stored in the project's data folder.
    local_files = sorted(glob.glob(os.path.join(folder, "*.csv")))

    # Build a list of available datasets.
    datasets = {}

    for path in local_files:
        label = f"Local: {os.path.basename(path)}"
        datasets[label] = ("local", path)

    for index, uploaded_file in enumerate(uploads or []):
        label = f"Uploaded: {uploaded_file.name}"
        # Avoid duplicate labels if files have the same name.
        while label in datasets:
            index += 1
            label = f"Uploaded ({index}): {uploaded_file.name}"

        datasets[label] = ("upload", uploaded_file)

    if not datasets:
        return None

    st.subheader(f"{track.title()} dataset selection")

    load_mode = st.radio(
        "How would you like to load the data?",
        ["Use one dataset", "Combine selected datasets"],
        horizontal=True,
        key=f"{track}_load_mode"
    )

    if load_mode == "Use one dataset":
        selected_labels = [
            st.selectbox(
                "Choose a CSV file",
                list(datasets.keys()),
                key=f"{track}_single_dataset"
            )
        ]
    else:
        selected_labels = st.multiselect(
            "Choose CSV files to combine",
            list(datasets.keys()),
            default=list(datasets.keys()),
            key=f"{track}_multiple_datasets"
        )

        if not selected_labels:
            st.info("Select at least one CSV file.")
            return None

    frames = []

    for label in selected_labels:
        source_type, source = datasets[label]

        try:
            if source_type == "local":
                frame = pd.read_csv(source, low_memory=False)
            else:
                source.seek(0)
                frame = pd.read_csv(source, low_memory=False)

            # Keep track of the origin of each observation.
            frame["_source_file"] = (
                os.path.basename(source)
                if source_type == "local"
                else source.name
            )

            if track == "clustering":
                frame["_source_station"] = os.path.splitext(
                    frame["_source_file"].iloc[0]
                )[0] if not frame.empty else label

            frames.append(frame)

        except Exception as error:
            st.warning(f"Could not read {label}: {error}")

    if not frames:
        return None

    if len(frames) == 1:
        return frames[0]

    # Align columns by name when combining multiple files.
    return pd.concat(frames, ignore_index=True, sort=False)


def safe_numeric(df):
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            # convert columns that are mostly numeric; preserve categorical columns otherwise
            converted = pd.to_numeric(out[c], errors="coerce")
            if converted.notna().mean() >= 0.82:
                out[c] = converted
    return out

def data_quality(df):
    return pd.DataFrame({
        "Column": df.columns,
        "Data type": [str(x) for x in df.dtypes],
        "Missing": [int(df[c].isna().sum()) for c in df.columns],
        "Missing %": [round(float(df[c].isna().mean()*100), 2) for c in df.columns],
        "Unique values": [int(df[c].nunique(dropna=True)) for c in df.columns],
    }).sort_values("Missing %", ascending=False)

def numeric_columns(df, exclude=None):
    exclude = set(exclude or [])
    return [c for c in df.select_dtypes(include=np.number).columns if c not in exclude and not c.startswith("_")]

def model_preprocessor(X):
    num_cols = list(X.select_dtypes(include=np.number).columns)
    cat_cols = [c for c in X.columns if c not in num_cols]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True))])
    return ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols)
    ], remainder="drop")

def cap_rows(X, y=None, max_rows=16000, seed=42):
    if len(X) <= max_rows:
        return X, y
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(len(X), max_rows, replace=False))
    if y is None:
        return X.iloc[idx].copy(), None
    return X.iloc[idx].copy(), y.iloc[idx].copy() if hasattr(y, "iloc") else np.asarray(y)[idx]

def show_dataset_header(df, track_name, target=None):
    st.markdown(f"""
    <div class="hero">
      <div class="eyebrow">TRACK / {track_name}</div>
      <h1>{track_name} intelligence</h1>
      <p>Explore the raw data, review data-quality signals, inspect distributions and relationships, then compare reproducible baseline models. Metrics shown here are computed from the dataset loaded into this app.</p>
      <span class="pill">Interactive EDA</span><span class="pill">Model comparison</span><span class="pill">Downloadable outputs</span>
    </div>""", unsafe_allow_html=True)
    a,b,c,d = st.columns(4)
    a.metric("Rows", f"{len(df):,}")
    b.metric("Columns", f"{df.shape[1]:,}")
    c.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    d.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
    if target and target in df.columns:
        st.caption(f"Target column detected: `{target}`")

def eda_panel(df, target=None, track=""):
    st.subheader("Data quality & cleaning preview")
    left, right = st.columns([1.15, 1])
    with left:
        st.dataframe(data_quality(df), use_container_width=True, height=300)
    with right:
        st.markdown("**Missingness by column**")
        miss = df.isna().mean().mul(100).sort_values(ascending=False).head(18).reset_index()
        miss.columns = ["Column", "Missing %"]
        st.plotly_chart(px.bar(miss, x="Missing %", y="Column", orientation="h", color="Missing %",
                               color_continuous_scale=["#38bdf8","#818cf8","#f472b6"],
                               template="plotly_dark", title="Columns with most missing values").update_layout(height=340, margin=dict(l=5,r=5,t=45,b=5)),
                        use_container_width=True)
    st.markdown("**Raw preview**")
    st.dataframe(df.head(12), use_container_width=True)
    st.download_button("Download loaded data as CSV", df.to_csv(index=False).encode("utf-8"), file_name=f"{track.lower()}_loaded_data.csv", mime="text/csv")
    numeric = numeric_columns(df)
    if numeric:
        st.subheader("Feature distributions")
        selected = st.multiselect("Choose numeric features", numeric, default=numeric[:min(4,len(numeric))], key=f"dist_{track}")
        if selected:
            c1,c2 = st.columns(2)
            with c1:
                col = st.selectbox("Histogram feature", selected, key=f"hist_{track}")
                st.plotly_chart(px.histogram(df, x=col, nbins=45, marginal="box", template="plotly_dark",
                                             color_discrete_sequence=["#38bdf8"], title=f"Distribution · {col}").update_layout(height=390), use_container_width=True)
            with c2:
                col2 = st.selectbox("Box plot feature", selected, index=min(1,len(selected)-1), key=f"box_{track}")
                st.plotly_chart(px.box(df, y=col2, points="outliers", template="plotly_dark",
                                       color_discrete_sequence=["#a78bfa"], title=f"Spread & potential outliers · {col2}").update_layout(height=390), use_container_width=True)
        if len(numeric) >= 2:
            st.subheader("Correlation map")
            corr_cols = st.multiselect("Correlation features (limit 18 for readability)", numeric, default=numeric[:min(12,len(numeric))], key=f"corr_{track}")
            if len(corr_cols) >= 2:
                corr = df[corr_cols].corr(numeric_only=True)
                st.plotly_chart(px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r",
                                          zmin=-1, zmax=1, template="plotly_dark", title="Pearson correlation"),
                                use_container_width=True)
    if target and target in df.columns:
        st.subheader("Target distribution")
        counts = df[target].astype("string").fillna("Missing target").value_counts().reset_index()
        counts.columns = [target, "Count"]
        st.plotly_chart(px.bar(counts, x=target, y="Count", color=target, template="plotly_dark",
                               title=f"Distribution of {target}", color_discrete_sequence=px.colors.qualitative.Bold),
                        use_container_width=True)

def classification_page(df):
    target_default = "AQI_Bucket" if "AQI_Bucket" in df.columns else None
    show_dataset_header(df, "Classification", target_default)
    eda_panel(df, target_default, "Classification")
    st.divider()
    st.subheader("Train and compare classifiers")
    st.caption("This is an interactive baseline run on the loaded data. It does not overwrite your notebook results. The app removes rows without the target, excludes obvious leakage columns, and fits imputation/encoding/scaling inside each pipeline.")
    possible = list(df.columns)
    default_target = target_default or st.selectbox("Target column", possible, key="cls_target")
    target = st.selectbox("Target column", possible, index=possible.index(default_target), key="cls_target_pick")
    excluded = st.multiselect("Columns to exclude from features (IDs, leakage, notes)", [c for c in df.columns if c != target],
                              default=[c for c in ["AQI","Date","_source_file","_source_station"] if c in df.columns and c != target], key="cls_exclude")
    limit = st.slider("Maximum rows for model training", 2000, 20000, 12000, 1000, key="cls_limit")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1200, class_weight="balanced"),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "Gaussian Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42, class_weight="balanced"),
        "SVM (RBF)": SVC(C=1, kernel="rbf", probability=True, random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=-1, class_weight="balanced_subsample"),
        "AdaBoost": AdaBoostClassifier(n_estimators=60, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=70, random_state=42),
        "Bagging": BaggingClassifier(n_estimators=40, random_state=42, n_jobs=-1),
        "MLP Neural Network": MLPClassifier(hidden_layer_sizes=(64,), max_iter=250, early_stopping=True, random_state=42)
    }
    selected_models = st.multiselect("Models to run", list(models), default=["Logistic Regression","Decision Tree","Random Forest","Gradient Boosting"])
    if st.button("Run classification benchmark", type="primary", key="run_cls"):
        work = safe_numeric(df).dropna(subset=[target]).copy()
        y = work[target].astype(str)
        drop = [target] + [c for c in excluded if c in work.columns]
        X = work.drop(columns=drop, errors="ignore")
        X = X.drop(columns=[c for c in X.columns if X[c].nunique(dropna=True) <= 1], errors="ignore")
        if X.shape[1] == 0 or y.nunique() < 2:
            st.error("Please choose a target with at least two classes and keep at least one feature.")
            return
        X, y = cap_rows(X, y, limit)
        counts = y.value_counts()
        keep = counts[counts >= 2].index
        mask = y.isin(keep)
        X, y = X.loc[mask], y.loc[mask]
        try:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
        except ValueError:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42)
        results, fitted, preds = [], {}, {}
        progress = st.progress(0, text="Preparing model runs…")
        for i, name in enumerate(selected_models):
            try:
                # GaussianNB needs dense input after one-hot encoding.
                pre = model_preprocessor(X)
                if name == "Gaussian Naive Bayes":
                    pre = ColumnTransformer([
                        ("num", Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]), list(X.select_dtypes(include=np.number).columns)),
                        ("cat", Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("encode",OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), [c for c in X.columns if c not in X.select_dtypes(include=np.number).columns])
                    ])
                pipe = Pipeline([("prep", pre), ("model", models[name])])
                pipe.fit(Xtr, ytr)
                pred = pipe.predict(Xte)
                row = {"Model":name, "Accuracy":accuracy_score(yte,pred), "Precision (weighted)":precision_score(yte,pred,average="weighted",zero_division=0),
                       "Recall (weighted)":recall_score(yte,pred,average="weighted",zero_division=0), "Weighted F1":f1_score(yte,pred,average="weighted",zero_division=0)}
                if hasattr(pipe,"predict_proba"):
                    try:
                        proba=pipe.predict_proba(Xte)
                        row["ROC-AUC (OvR)"]=roc_auc_score(yte,proba,labels=pipe.named_steps["model"].classes_,multi_class="ovr",average="weighted")
                    except Exception:
                        row["ROC-AUC (OvR)"]=np.nan
                else: row["ROC-AUC (OvR)"]=np.nan
                results.append(row); fitted[name]=pipe; preds[name]=pred
            except Exception as e:
                st.warning(f"{name} skipped: {e}")
            progress.progress((i+1)/max(len(selected_models),1), text=f"Finished {name}")
        if results:
            res=pd.DataFrame(results).sort_values("Weighted F1",ascending=False)
            st.session_state["cls_results"]=res
            st.session_state["cls_fitted"]=fitted
            st.session_state["cls_preds"]=preds
            st.session_state["cls_yte"]=yte
            st.session_state["cls_Xte"]=Xte
            st.success("Benchmark complete.")
    if "cls_results" in st.session_state:
        res=st.session_state["cls_results"]
        st.markdown("#### Model leaderboard")
        st.dataframe(res.style.format({c:"{:.4f}" for c in res.columns if c!="Model"}),use_container_width=True)
        st.plotly_chart(px.bar(res, x="Model", y="Weighted F1", color="Accuracy", barmode="group", template="plotly_dark",
                               color_continuous_scale="Tealgrn", title="Classifier performance · weighted F1 and accuracy").update_layout(xaxis_tickangle=-25), use_container_width=True)
        names=list(st.session_state["cls_preds"].keys())
        chosen=st.selectbox("Inspect confusion matrix",names,key="cm_pick")
        yt=st.session_state["cls_yte"]; yp=st.session_state["cls_preds"][chosen]
        labels=sorted(pd.Series(yt).unique())
        cm=confusion_matrix(yt,yp,labels=labels)
        st.plotly_chart(px.imshow(cm,x=labels,y=labels,text_auto=True,color_continuous_scale="Blues",template="plotly_dark",
                                  labels=dict(x="Predicted",y="Actual",color="Count"),title=f"Confusion matrix · {chosen}"),use_container_width=True)
        st.download_button("Download model comparison CSV",res.to_csv(index=False).encode(), "classification_model_results.csv","text/csv")
        with st.expander("Classification report"):
            st.code(classification_report(yt,yp,zero_division=0))

def regression_page(df):
    show_dataset_header(df,"Regression","PM2.5" if "PM2.5" in df.columns else None)
    eda_panel(df,"PM2.5" if "PM2.5" in df.columns else None,"Regression")
    st.divider()
    st.subheader("Train and compare regressors")
    numeric = numeric_columns(safe_numeric(df))
    target_default = "PM2.5" if "PM2.5" in numeric else (numeric[-1] if numeric else None)
    if not target_default:
        st.error("No numeric columns found. Please upload a numeric regression dataset.")
        return
    target=st.selectbox("Numeric target",numeric,index=numeric.index(target_default),key="reg_target")
    excluded=st.multiselect("Exclude these feature columns", [c for c in df.columns if c!=target],
                            default=[c for c in ["No","year","month","day","hour","_source_file"] if c in df.columns and c!=target],key="reg_exclude")
    limit=st.slider("Maximum rows for training",5000,50000,20000,5000,key="reg_limit")
    models={
        "Linear Regression":LinearRegression(),
        "Ridge Regression":Ridge(alpha=1.0),
        "Lasso Regression":Lasso(alpha=.01,max_iter=5000),
        "ElasticNet":ElasticNet(alpha=.01,l1_ratio=.5,max_iter=5000,random_state=42),
        "Decision Tree":DecisionTreeRegressor(max_depth=18,random_state=42),
        "Random Forest":RandomForestRegressor(n_estimators=70,max_depth=18,min_samples_leaf=2,random_state=42,n_jobs=-1),
        "Gradient Boosting":GradientBoostingRegressor(n_estimators=80,random_state=42),
        "KNN Regressor":KNeighborsRegressor(n_neighbors=7,n_jobs=-1),
        "SVR (RBF)":SVR(C=10,epsilon=.1),
    }
    selected=st.multiselect("Models to run",list(models),default=["Linear Regression","Ridge Regression","Decision Tree","Random Forest","Gradient Boosting"])
    if st.button("Run regression benchmark",type="primary",key="run_reg"):
        work=safe_numeric(df).copy()
        work=work.dropna(subset=[target])
        y=work[target].astype(float)
        drop=[target]+[c for c in excluded if c in work.columns]
        X=work.drop(columns=drop,errors="ignore")
        X=X.drop(columns=[c for c in X.columns if X[c].nunique(dropna=True)<=1],errors="ignore")
        if X.empty:
            st.error("Keep at least one predictor column.")
            return
        X,y=cap_rows(X,y,limit)
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
        results=[]; preds={}; fitted={}
        progress=st.progress(0,text="Training regressors…")
        for i,name in enumerate(selected):
            try:
                pipe=Pipeline([("prep",model_preprocessor(X)),("model",models[name])])
                # SVR is expensive; fit on a subset but evaluate on the same held-out test split.
                if name=="SVR (RBF)" and len(Xtr)>10000:
                    idx=np.random.RandomState(42).choice(len(Xtr),10000,replace=False)
                    pipe.fit(Xtr.iloc[idx],ytr.iloc[idx])
                else:
                    pipe.fit(Xtr,ytr)
                pred=pipe.predict(Xte)
                results.append({"Model":name,"R²":r2_score(yte,pred),"RMSE":float(np.sqrt(mean_squared_error(yte,pred))),"MAE":mean_absolute_error(yte,pred)})
                preds[name]=pred; fitted[name]=pipe
            except Exception as e: st.warning(f"{name} skipped: {e}")
            progress.progress((i+1)/max(len(selected),1),text=f"Finished {name}")
        if results:
            res=pd.DataFrame(results).sort_values("R²",ascending=False)
            st.session_state["reg_results"]=res; st.session_state["reg_preds"]=preds
            st.session_state["reg_yte"]=yte; st.session_state["reg_fitted"]=fitted
            st.session_state["reg_Xte"]=Xte
            st.success("Regression benchmark complete.")
    if "reg_results" in st.session_state:
        res=st.session_state["reg_results"]
        st.markdown("#### Model leaderboard")
        st.dataframe(res.style.format({c:"{:.4f}" for c in ["R²","RMSE","MAE"]}),use_container_width=True)
        st.plotly_chart(px.bar(res,x="Model",y="R²",color="RMSE",template="plotly_dark",color_continuous_scale="Tealgrn",
                               title="Regression comparison · higher R² is better").update_layout(xaxis_tickangle=-20),use_container_width=True)
        names=list(st.session_state["reg_preds"].keys())
        chosen=st.selectbox("Inspect predictions",names,key="reg_pred_pick")
        yt=st.session_state["reg_yte"]; yp=st.session_state["reg_preds"][chosen]
        sample=pd.DataFrame({"Actual":np.asarray(yt),"Predicted":yp})
        sample=sample.sample(min(3000,len(sample)),random_state=42)
        st.plotly_chart(px.scatter(sample,x="Actual",y="Predicted",opacity=.55,template="plotly_dark",
                                   title=f"Predicted vs actual · {chosen}",trendline=None).add_trace(
                                       go.Scatter(x=[sample.Actual.min(),sample.Actual.max()],y=[sample.Actual.min(),sample.Actual.max()],mode="lines",name="Perfect prediction",line=dict(dash="dash"))),
                        use_container_width=True)
        residual=pd.DataFrame({"Predicted":np.asarray(yp),"Residual":np.asarray(yt)-np.asarray(yp)})
        st.plotly_chart(px.scatter(residual.sample(min(3000,len(residual)),random_state=42),x="Predicted",y="Residual",opacity=.55,template="plotly_dark",
                                   title=f"Residual diagnostic · {chosen}").add_hline(y=0,line_dash="dash"),use_container_width=True)
        st.download_button("Download model comparison CSV",res.to_csv(index=False).encode(),"regression_model_results.csv","text/csv")

def clustering_page(df):
    show_dataset_header(df,"Clustering")
    eda_panel(df,None,"Clustering")
    st.divider()
    st.subheader("Discover pollution patterns")
    data=safe_numeric(df)
    exclude=[c for c in data.columns if c.startswith("_")]
    nums=numeric_columns(data,exclude)
    if len(nums)<2:
        st.error("At least two numeric features are required for clustering. Upload the station-hour CSV files, or choose a dataset with numeric measurements.")
        return
    default=[c for c in nums if c.lower() not in ["no","year","month","day","hour"]][:min(8,len(nums))]
    selected=st.multiselect("Numeric features for clustering",nums,default=default,key="clust_features")
    k=st.slider("Number of clusters (k)",2,10,4,key="cluster_k")
    max_rows=st.slider("Rows to cluster (sampling keeps the browser responsive)",1000,12000,5000,1000,key="cluster_limit")
    if st.button("Run clustering analysis",type="primary",key="run_cluster"):
        X=data[selected].replace([np.inf,-np.inf],np.nan).copy()
        X=X.dropna(axis=1,how="all")
        X=X.fillna(X.median(numeric_only=True))
        X=X.loc[:,X.nunique(dropna=True)>1]
        if X.shape[1]<2:
            st.error("Choose at least two usable, non-constant numeric features.")
            return
        if len(X)>max_rows:
            X=X.sample(max_rows,random_state=42)
        scaler=StandardScaler()
        Xs=scaler.fit_transform(X)
        km=KMeans(n_clusters=k,random_state=42,n_init=10)
        labels=km.fit_predict(Xs)
        metrics={"Algorithm":"K-Means","Silhouette Score":silhouette_score(Xs,labels),"Davies-Bouldin Index":davies_bouldin_score(Xs,labels),"Calinski-Harabasz Index":calinski_harabasz_score(Xs,labels)}
        # Ward linkage is supported for Euclidean features; run on the same bounded sample.
        try:
            hier=AgglomerativeClustering(n_clusters=k,linkage="ward")
            hlabels=hier.fit_predict(Xs)
            metrics2={"Algorithm":"Hierarchical (Ward)","Silhouette Score":silhouette_score(Xs,hlabels),"Davies-Bouldin Index":davies_bouldin_score(Xs,hlabels),"Calinski-Harabasz Index":calinski_harabasz_score(Xs,hlabels)}
        except Exception as e:
            hlabels=None; metrics2=None; st.warning(f"Hierarchical model skipped: {e}")
        pca=PCA(n_components=2,random_state=42)
        xy=pca.fit_transform(Xs)
        out=pd.DataFrame({"PC1":xy[:,0],"PC2":xy[:,1],"K-Means cluster":labels.astype(str)})
        if hlabels is not None: out["Hierarchical cluster"]=hlabels.astype(str)
        st.session_state["cluster_metrics"]=pd.DataFrame([metrics]+([metrics2] if metrics2 else []))
        st.session_state["cluster_plot"]=out
        st.session_state["cluster_elbow"]=(Xs,k)
        st.session_state["cluster_features"]=selected
        st.success("Clustering analysis complete.")
    if "cluster_metrics" in st.session_state:
        st.markdown("#### Cluster quality metrics")
        st.dataframe(st.session_state["cluster_metrics"].style.format({c:"{:.4f}" for c in ["Silhouette Score","Davies-Bouldin Index","Calinski-Harabasz Index"]}),use_container_width=True)
        out=st.session_state["cluster_plot"]
        method=st.selectbox("Visualise algorithm",["K-Means","Hierarchical (Ward)"],key="cluster_viz_method")
        labelcol="K-Means cluster" if method=="K-Means" else "Hierarchical cluster"
        if labelcol in out:
            st.plotly_chart(px.scatter(out,x="PC1",y="PC2",color=labelcol,template="plotly_dark",opacity=.75,
                                       title=f"PCA 2D cluster map · {method}",color_discrete_sequence=px.colors.qualitative.Bold),
                            use_container_width=True)
        Xs,k=st.session_state["cluster_elbow"]
        ks=list(range(2,min(11,max(3,len(Xs)//100)+1)))
        inertias=[]
        for kk in ks:
            inertias.append(KMeans(n_clusters=kk,random_state=42,n_init=10).fit(Xs).inertia_)
        st.plotly_chart(px.line(x=ks,y=inertias,markers=True,template="plotly_dark",labels={"x":"Number of clusters (k)","y":"Inertia"},
                                title="Elbow curve · K-Means").update_traces(line_color="#38bdf8"),use_container_width=True)
        st.subheader("Hierarchical dendrogram")
        st.caption("A dendrogram on a sample of up to 700 observations keeps this view responsive; it illustrates how hierarchical groups merge.")
        try:
            from scipy.cluster.hierarchy import linkage, dendrogram
            from scipy.spatial.distance import pdist
            import matplotlib.pyplot as plt
            dendro_n = min(700, len(Xs))
            dendro_idx = np.random.RandomState(42).choice(len(Xs), dendro_n, replace=False) if len(Xs) > dendro_n else np.arange(len(Xs))
            Z = linkage(Xs[dendro_idx], method="ward")
            fig, ax = plt.subplots(figsize=(12, 4.5))
            dendrogram(Z, truncate_mode="lastp", p=35, leaf_rotation=45, leaf_font_size=8, ax=ax)
            ax.set_title("Ward linkage · truncated dendrogram")
            ax.set_xlabel("Merged sample / cluster")
            ax.set_ylabel("Merge distance")
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
        except Exception as e:
            st.info(f"Dendrogram could not be rendered for this selection: {e}")
        st.subheader("t-SNE projection")
        st.caption("A sample-based 2D projection; t-SNE shows local neighbourhood structure and is not a quality metric.")
        try:
            tsne_n = min(1000, len(Xs))
            tsne_idx = np.random.RandomState(42).choice(len(Xs), tsne_n, replace=False) if len(Xs) > tsne_n else np.arange(len(Xs))
            tsne_xy = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto", perplexity=min(30, max(5, (tsne_n-1)//3))).fit_transform(Xs[tsne_idx])
            tsne_df = pd.DataFrame({"t-SNE 1": tsne_xy[:,0], "t-SNE 2": tsne_xy[:,1], "Cluster": out.iloc[tsne_idx][labelcol].to_numpy() if labelcol in out else out.iloc[tsne_idx]["K-Means cluster"].to_numpy()})
            st.plotly_chart(px.scatter(tsne_df, x="t-SNE 1", y="t-SNE 2", color="Cluster", opacity=.75, template="plotly_dark", title=f"t-SNE · {method}"), use_container_width=True)
        except Exception as e:
            st.info(f"t-SNE could not be rendered for this selection: {e}")
        st.download_button("Download cluster assignments",out.to_csv(index=False).encode(),"cluster_assignments_pca.csv","text/csv")

def get_uploads(track):
    return st.file_uploader(
        f"Add {track} CSV file(s) here (optional if files are already in data/{track}/)",
        type=["csv"], accept_multiple_files=True, key=f"upload_{track}"
    )

if page == "Overview":
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">MACHINE LEARNING CAPSTONE · INTERACTIVE SHOWCASE</div>
      <h1>Air quality, seen through data.</h1>
      <p>A polished workspace for your three ML tracks: predict AQI categories, estimate PM2.5 concentration, and discover hidden pollution patterns. Explore data quality and EDA, benchmark models, inspect errors, and export results.</p>
      <span class="pill">Classification</span><span class="pill">Regression</span><span class="pill">Clustering</span><span class="pill">EDA + Metrics</span>
    </div>
    """, unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1:
        st.markdown("### 01 · Classification")
        st.markdown("**Question:** Which AQI category does a city-day belong to?")
        st.markdown("- Target: `AQI_Bucket`\n- 10 classifiers available\n- Accuracy, weighted F1, ROC-AUC (OvR)\n- Confusion matrix and class report")
    with c2:
        st.markdown("### 02 · Regression")
        st.markdown("**Question:** How much PM2.5 should we expect?")
        st.markdown("- Target: `PM2.5`\n- Regression leaderboard\n- R², RMSE, MAE\n- Residual and predicted-vs-actual plots")
    with c3:
        st.markdown("### 03 · Clustering")
        st.markdown("**Question:** Which observations share similar pollution patterns?")
        st.markdown("- K-Means + hierarchical clustering\n- Silhouette, Davies–Bouldin, Calinski–Harabasz\n- Elbow curve and PCA 2D view")
    st.markdown("## Start here")
    st.info("Use the sidebar to open a track. You can put CSVs in the matching `data/` folder or upload them directly in the app. Your notebooks remain the source of your capstone's documented experiments; this dashboard runs a separate, interactive benchmark on the data you load.")
    st.markdown("### Data placement")
    st.code("air_quality_ml_showcase/\n├── app.py\n├── requirements.txt\n├── README.md\n└── data/\n    ├── classification/  # city_day.csv\n    ├── regression/      # Beijing station CSV files\n    └── clustering/      # AshokVihar_Hourly.csv, DwarkaSec8_Hourly.csv, etc.")
    st.markdown("### Suggested presentation flow")
    st.markdown("**Problem → dataset audit → cleaning and EDA → model comparison → best model diagnostics → limitations and next steps.**")

elif page == "Classification":
    uploads=get_uploads("classification")
    df=load_track_data("classification",uploads)
    if df is None:
        st.markdown("<div class='hero'><div class='eyebrow'>TRACK 01</div><h1>Classification</h1><p>Predict an AQI category from pollutant and contextual features.</p></div>",unsafe_allow_html=True)
        st.warning("No classification CSV found yet. Add `city_day.csv` to `data/classification/` or upload it above.")
    else:
        classification_page(df)

elif page == "Regression":
    uploads=get_uploads("regression")
    df=load_track_data("regression",uploads)
    if df is None:
        st.markdown("<div class='hero'><div class='eyebrow'>TRACK 02</div><h1>Regression</h1><p>Estimate continuous PM2.5 concentration.</p></div>",unsafe_allow_html=True)
        st.warning("No regression CSV found yet. Add your Beijing station CSV files to `data/regression/` or upload them above. Multiple station files are concatenated automatically.")
    else:
        regression_page(df)

elif page == "Clustering":
    uploads=get_uploads("clustering")
    df=load_track_data("clustering",uploads)
    if df is None:
        st.markdown("<div class='hero'><div class='eyebrow'>TRACK 03</div><h1>Clustering</h1><p>Group similar air-quality observations without using target labels.</p></div>",unsafe_allow_html=True)
        st.warning("No clustering CSV found yet. Add your station-hour files to `data/clustering/` or upload them above. Multiple files are combined.")
    else:
        clustering_page(df)

elif page == "Data Explorer":
    st.markdown("<div class='hero'><div class='eyebrow'>DATA LAB</div><h1>Inspect before you model.</h1><p>Upload any CSV and audit its shape, missingness, duplicates, data types, feature distributions, and correlations.</p></div>",unsafe_allow_html=True)
    files=st.file_uploader("Upload one or more CSV files",type=["csv"],accept_multiple_files=True,key="generic_upload")
    if files:
        choices={f.name:f for f in files}
        chosen=st.selectbox("Dataset",list(choices))
        df=pd.read_csv(choices[chosen],low_memory=False)
        st.metric("Rows × columns",f"{df.shape[0]:,} × {df.shape[1]:,}")
        eda_panel(df,None,"Data Explorer")

else:
    st.markdown("<div class='hero'><div class='eyebrow'>PROJECT GUIDE</div><h1>Setup & capstone checklist</h1><p>Use this dashboard as the presentation layer around your three notebooks. It does not claim to reproduce notebook scores unless you run the matching data and models here.</p></div>",unsafe_allow_html=True)
    st.markdown("### Rubric-aligned showcase")
    rubric = pd.DataFrame([
        ["Dataset audit","Shape, types, missing values, duplicates","Included"],
        ["EDA","Feature distributions, correlation heatmap, target distribution","Included where applicable"],
        ["Cleaning visibility","Missingness table, raw preview, downloadable loaded data","Included; explicit cleaning decisions remain documented in notebooks"],
        ["Regression evaluation","R², RMSE, MAE; actual-vs-predicted and residual plots","Included"],
        ["Classification evaluation","Accuracy, weighted precision/recall/F1, ROC-AUC OvR, confusion matrix","Included"],
        ["Clustering evaluation","Silhouette, Davies–Bouldin, Calinski–Harabasz, elbow, PCA view","Included"],
        ["Tuning and CV","Grid/Randomized search and cross-validation","Keep detailed tuning evidence in notebooks; dashboard benchmarks are interactive baselines"],
        ["Reproducibility","Random seed 42 used where applicable","Included in app baseline runs"],
        ["Deployment","Streamlit Community Cloud / Render","Optional next step"],
    ],columns=["Requirement","What to show","Status"])
    st.dataframe(rubric,use_container_width=True,hide_index=True)
    st.markdown("### Run locally")
    st.code("python -m venv .venv\n# Windows PowerShell\n.venv\\Scripts\\Activate.ps1\npip install -r requirements.txt\nstreamlit run app.py",language="powershell")
    st.markdown("### Dataset placement")
    st.code("data/classification/city_day.csv\n\ndata/regression/  # place all 12 Beijing station CSV files here\n\ndata/clustering/  # place AshokVihar_Hourly.csv, DCStadium_Hourly.csv, DwarkaSec8_Hourly.csv, Najafgarh_Hourly.csv, NehruNagar_Hourly.csv, Okhla_Hourly.csv here")
    st.markdown("### Important notes")
    st.markdown("""
- Do not put `AQI` in the classification features when predicting `AQI_Bucket`; it is direct target leakage.
- For regression, choose `PM2.5` as the target and exclude it from features.
- For clustering, do not include known labels/cluster columns as input features. Use them only after clustering for interpretation.
- The app's model runs are a visual interactive benchmark, not a replacement for the exact tuned results and cross-validation evidence in your notebooks.
- Before submission, run all notebook cells top-to-bottom and make sure your reported results match the version you present.
- If the app is deployed publicly, large raw datasets can make deployment slow. Consider keeping datasets locally or using a smaller, clearly documented sample.
""")
