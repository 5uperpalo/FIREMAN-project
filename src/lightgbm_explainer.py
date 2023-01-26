from copy import copy
from typing import List, Literal, Optional
import pandas as pd

import lightgbm as lgb
import numpy as np
import shap
from shap.plots._decision import DecisionPlotResult


class ShapExplainer:
    def __init__(
        self,
        preprocessor: object,
        model: lgb.basic.Booster,
        objective: Literal["binary", "multiclass", "regression"],
        n_class: int = 3,
    ):
        """SHAP explainer.
        Args:
            preprocessor (object): model preprocessor
            model (lgb.basic.Booster): trained model
            objective (str): type of the task/objective
            n_class (int): number of classes in case muticlass objective is selected
        """
        self.preprocessor = preprocessor
        self.model = model
        self.objective = objective
        self.n_class = n_class

    def explain_decision_plot(self, df: pd.DataFrame) -> DecisionPlotResult:
        """Process the data and pick proper decision plot based on model output.
        Args:
            df (pd.DataFrame): input data
        Returns:
            shap_decision_plot (DecisionPlotResult): decision plot
        """
        if (self.objective in ["binary", "multiclass"]) and (df.shape[0] > 1):
            raise ValueError(
                """
                Multioutput decision plots for classification support only per value
                analysis.
                """
            )

        shap_values = self.compute_shap_values(df)
        base_value = self.compute_base_value(df)
        if df.shape[0] == 1:
            legend_labels = self._legend_labels(df)
        else:
            legend_labels = None

        if self.objective in ["binary", "regression"]:
            shap_decision_plot = shap.decision_plot(
                base_value=base_value[:, -1:][0],
                shap_values=shap_values,
                features=df[self.model.feature_name()],
                feature_names=self.model.feature_name(),
                legend_labels=legend_labels,
                link="logit",
                highlight=0,
            )
        elif self.objective == "multiclass":
            shap_decision_plot = shap.multioutput_decision_plot(
                base_values=list(base_value),
                shap_values=list(shap_values),
                features=df[self.model.feature_name()],
                feature_names=self.model.feature_name(),
                legend_labels=legend_labels,
                link="logit",
                row_index=0,
            )

        return shap_decision_plot

    def compute_shap_values(self, df: pd.DataFrame) -> np.ndarray:
        """Helper method to compute SHAP values for other shap plots.
        Args:
            df (pd.DataFrame): input data
        Returns:
            shap_values (pd.DataFrame): computed shap values
        """
        df_p = self.preprocessor.transform(df)
        shap_values = self.model.predict(
            data=df_p[self.model.feature_name()], pred_contrib=True
        )
        if self.objective == "multiclass":
            shap_values = shap_values.reshape(
                self.n_class, df_p.shape[0], len(self.model.feature_name()) + 1
            )[:, :, :-1]
        elif self.objective in ["binary", "regression"]:
            shap_values = shap_values[:, :-1]
        return shap_values

    def compute_base_value(self, df: pd.DataFrame) -> np.ndarray:
        """Helper method to compute SHAP base/exptected value for other shap plots.
        Args:
            df (pd.DataFrame): input data
        Returns:
            base_value (pd.DataFrame): computed base values
        """
        df_p = self.preprocessor.transform(df)
        base_value = self.model.predict(
            data=df_p[self.model.feature_name()], pred_contrib=True
        )

        if self.objective == "multiclass":
            base_value = base_value.reshape(
                self.n_class, df_p.shape[0], len(self.model.feature_name()) + 1
            )[:, :, -1]
        elif self.objective in ["binary", "regression"]:
            base_value = base_value[:, -1:]
        return base_value

    def _legend_labels(self, df: pd.DataFrame) -> List[str]:
        """Helper method to create legend with raw and predicted values.
        Args:
            df (pd.DataFrame): array of values to explain
        Returns:
            labels (list): list of labels with values
        """
        # preds_raw = np.squeeze(self.model.predict(df))
        if self.objective == "binary":
            preds = self.compute_preds(df)[0]
            labels = [
                f"Sample prob. {preds.round(2):.2f})"
            ]  # (raw {preds_raw.round(2):.2f})"]
        elif self.objective == "regression":
            preds = self.compute_preds(df)[0]
            labels = [f"Sample val.: {preds.round(2):.2f}"]
        elif self.objective == "multiclass":
            preds = np.squeeze(self.compute_preds(df))
            labels = [
                f"""
                Class {i} prob. {preds[i].round(2):.2f}) 
                """
                for i in range(len(preds))
            ]
            # Class {i} prob. {preds[i].round(2):.2f} (raw {preds_raw[i].round(2):.2f})
        return labels

    def compute_preds(self, df: pd.DataFrame) -> np.ndarray:
        """Helper method to compute predictions of the model.
        Args:
            df (pd.DataFrame): input data
        Returns:
            preds (pd.DataFrame): preditions
        """

        df_p = self.preprocessor.transform(df)
        preds = self.model.predict(data=df_p[self.model.feature_name()])

        return preds
