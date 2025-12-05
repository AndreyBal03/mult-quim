import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage
from statsmodels.formula.api import ols
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import seaborn as sns


class DatasetModel:
    def __init__(self):
        self.df = None

    def load_csv(self, uploaded_file):
        try:
            self.df = pd.read_csv(uploaded_file)
            return True, None
        except Exception as e:
            return False, str(e)

    def load_excel(self, uploaded_file):
        try:
            self.df = pd.read_excel(uploaded_file)
            return True, None
        except Exception as e:
            return False, str(e)

    def get_summary_stats(self):
        if self.df is None:
            return None
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return None
        return {
            "mean": numeric_df.mean(),
            "median": numeric_df.median(),
            "mode": numeric_df.mode().iloc[0] if not numeric_df.mode().empty else None,
            "std": numeric_df.std(),
            "var": numeric_df.var(),
            "iqr": numeric_df.quantile(0.75) - numeric_df.quantile(0.25),
        }


class PreprocessingModel:
    def __init__(self, df):
        self.df = df.copy()

    def drop_na(self):
        initial_count = self.df.shape[0]
        self.df = self.df.dropna()
        dropped = initial_count - self.df.shape[0]
        return dropped

    def drop_column(self, column):
        if column in self.df.columns:
            self.df = self.df.drop(columns=[column])
            return True
        return False

    def standardize(self, columns):
        scaler = StandardScaler()
        self.df[columns] = scaler.fit_transform(self.df[columns])
        return True

    def get_numerical_columns(self):
        return self.df.select_dtypes(include=[np.number]).columns.tolist()


class PCAModel:
    def __init__(self, df, columns):
        self.df = df.copy()
        self.columns = columns
        self.pca = None
        self.pc_values = None
        self.explained = None
        self.cumulative = None

    def fit_pca(self):
        X = self.df[self.columns].values
        n_components = min(X.shape)
        self.pca = PCA(n_components=n_components)
        self.pc_values = self.pca.fit_transform(X)
        self.explained = self.pca.explained_variance_ratio_
        self.cumulative = np.cumsum(self.explained)
        return True

    def get_scree_data(self):
        if self.explained is None:
            raise ValueError("PCA not fitted yet")
        df_var = pd.DataFrame(
            {
                "Componente": [f"PC{i + 1}" for i in range(len(self.explained))],
                "Varianza (%)": self.explained * 100,
                "Acumulada (%)": self.cumulative * 100,
            }
        )
        return df_var

    def get_scores_plot(self, pc_x, pc_y, palette):
        pc_x_idx = int(pc_x.replace("PC", "")) - 1
        pc_y_idx = int(pc_y.replace("PC", "")) - 1
        df_plot = pd.DataFrame(
            self.pc_values[:, [pc_x_idx, pc_y_idx]], columns=[pc_x, pc_y]
        )
        df_plot["ID"] = self.df.index.astype(str)
        fig = px.scatter(
            df_plot,
            x=pc_x,
            y=pc_y,
            color="ID",
            title=f"PCA: {pc_x} vs {pc_y}",
            color_discrete_sequence=palette if isinstance(palette, list) else None,
        )
        fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
        fig.update_layout(template="plotly_white")
        return fig

    def get_biplot(self, pc_x, pc_y, palette):
        pc_x_idx = int(pc_x.replace("PC", "")) - 1
        pc_y_idx = int(pc_y.replace("PC", "")) - 1
        fig = go.Figure()
        scores_x = self.pc_values[:, pc_x_idx]
        scores_y = self.pc_values[:, pc_y_idx]
        fig.add_trace(
            go.Scatter(
                x=scores_x,
                y=scores_y,
                mode="markers",
                name="Muestras (Scores)",
                marker=dict(
                    size=10, color="#B0A461", line=dict(width=1, color="white")
                ),
                text=[f"Muestra {i}" for i in self.df.index],
                hovertemplate="<b>%{text}</b><br>%{x:.2f}, %{y:.2f}<extra></extra>",
            )
        )
        loadings_x = self.pca.components_[pc_x_idx, :]
        loadings_y = self.pca.components_[pc_y_idx, :]
        scale_factor = (
            0.8
            * max(np.max(np.abs(scores_x)), np.max(np.abs(scores_y)))
            / max(np.max(np.abs(loadings_x)), np.max(np.abs(loadings_y)))
        )
        for i, var_name in enumerate(self.columns):
            fig.add_trace(
                go.Scatter(
                    x=[0, loadings_x[i] * scale_factor],
                    y=[0, loadings_y[i] * scale_factor],
                    mode="lines+markers+text",
                    name=var_name,
                    line=dict(color="#4A525A", width=2),
                    marker=dict(size=[0, 8], color="#4A525A"),
                    text=["", var_name],
                    textposition="top center",
                    textfont=dict(size=10, color="#4A525A"),
                    hovertemplate=f"<b>{var_name}</b><br>Loading X: {loadings_x[i]:.3f}<br>Loading Y: {loadings_y[i]:.3f}<extra></extra>",
                    showlegend=False,
                )
            )
        fig.update_layout(
            title=f"Biplot: {pc_x} vs {pc_y}",
            xaxis_title=f"{pc_x} ({self.explained[pc_x_idx] * 100:.2f}%)",
            yaxis_title=f"{pc_y} ({self.explained[pc_y_idx] * 100:.2f}%)",
            template="plotly_white",
            hovermode="closest",
            width=800,
            height=600,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        return fig

    def get_loadings_df(self):
        loadings_df = pd.DataFrame(
            self.pca.components_.T,
            columns=[f"PC{i + 1}" for i in range(self.pca.components_.shape[0])],
            index=self.columns,
        )
        return loadings_df

    def get_transformed_df(self):
        pc_df = pd.DataFrame(
            self.pc_values,
            columns=[f"PC{i + 1}" for i in range(self.pc_values.shape[1])],
        )
        return pc_df


class ClusteringModel:
    def __init__(self, df, numerical_cols):
        self.df = df.copy()
        self.numerical_cols = numerical_cols

    def kmeans_cluster(self, k, n_init, init_method):
        kmeans = KMeans(n_clusters=k, n_init=n_init, init=init_method, random_state=42)
        labels = kmeans.fit_predict(self.df[self.numerical_cols])
        sil_score = silhouette_score(self.df[self.numerical_cols], labels)
        self.df["Cluster_KMeans"] = labels
        return labels, sil_score

    def hierarchical_cluster(self, linkage_method, n_clusters):
        model = AgglomerativeClustering(
            n_clusters=n_clusters, linkage=linkage_method, metric="euclidean"
        )
        labels = model.fit_predict(self.df[self.numerical_cols])
        self.df["Cluster_Hier"] = labels
        return labels

    def get_dendrogram(self, linkage_method):
        linked = linkage(self.df[self.numerical_cols], method=linkage_method)
        fig = ff.create_dendrogram(linked, orientation="left")
        return fig


class ANOVAModel:
    def __init__(self, df, y_var, x_var):
        self.df = df.copy()
        self.y_var = y_var
        self.x_var = x_var

    def compute_anova(self):
        df_model = self.df[[self.y_var, self.x_var]].copy()
        df_model = df_model.dropna()
        df_model[self.x_var] = df_model[self.x_var].astype("category")
        if df_model.shape[0] < 3:
            raise ValueError("Not enough data for ANOVA")
        n_groups = df_model[self.x_var].nunique()
        if n_groups < 2:
            raise ValueError("Need at least 2 groups")
        formula = f'Q("{self.y_var}") ~ C(Q("{self.x_var}"))'
        model = ols(formula, data=df_model).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        return anova_table, df_model

    def tukey_test(self, df_model):
        tukey = pairwise_tukeyhsd(
            endog=df_model[self.y_var], groups=df_model[self.x_var], alpha=0.05
        )
        return tukey
