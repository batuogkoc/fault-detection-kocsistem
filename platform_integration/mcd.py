import pandas as pd
from sklearn.covariance import MinCovDet
from .base import BaseDetector
from typing import Union

class MCD(BaseDetector):
    """MinConDet based anomaly detector for one dimensional time series data."""
    def __init__(self, 
                 rolling_type: str = "mean",
                 *,
                 date_column: str = "Date",
                 feature_column: str = "Value",
                 contamination: float = 0.1,
                 anomaly_threshold: Union[float, str] = "auto",
                 model_mp_quantile: float = 0.02,
                 feature_rp: str = None,
                 model_rp: str = "60min",
                 range_filter: bool = True,
                 range_rp: str = None,
                 raw_risk_rp: str = None,
                 risk_rp: str = None,
                 **kwargs):

        # Hyperparameters
        self.rolling_type = rolling_type
        self.date_column = date_column
        self.feature_column = feature_column
        self.contamination = contamination
        self.anomaly_threshold = anomaly_threshold
        self.model_mp_quantile = model_mp_quantile
        self.model_rp = pd.Timedelta(model_rp)
        self.feature_rp = pd.Timedelta(feature_rp) if feature_rp is not None else self.model_rp / 2
        self.range_filter = range_filter
        self.range_rp = pd.Timedelta(range_rp) if range_rp is not None else self.model_rp * 3 / 4
        self.raw_risk_rp = pd.Timedelta(raw_risk_rp) if raw_risk_rp is not None else self.model_rp * 2
        self.risk_rp = pd.Timedelta(risk_rp) if risk_rp is not None else self.model_rp * 6

        # Parameters
        self.model_mp: int  
        self.feature_mp: int
        self.raw_risk_mp: int 
        self.range_limit: float
        self.robust_cov: MinCovDet
    
    def __str__(self):
        try:
            temp_dict = vars(self).copy()
            if self.range_filter: temp_dict["range_limit_td"] = pd.to_timedelta(temp_dict["range_limit"], unit="s")
            temp_dict["robust_mean"] = self.robust_cov.location_
            temp_dict["robust_covariance"] = self.robust_cov.covariance_
            attributes = "\n".join(f"{key}: {value}" for key, value in temp_dict.items())
        except:
            print("Model is not trained")
            attributes = "\n".join(f"{key}: {value}" for key, value in temp_dict.items())
        return f"{self.__class__.__name__}:\n{attributes}"
    
    def __repr__(self):
        return self.__str__()

    def fit(self, data: pd.DataFrame):
        """
        Implementation of pipeline_fit_score function that executes feature_fit_transform and model_fit_score functions
        respectively and return the resulting dataframe
        """
        self.fit_min_periods(data)
        data = self.feature_transform(data)
        self.model_fit(data)

        return self

    def anomaly_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Implementation of pipeline_score function, that applied feature_transform and model_score functions, respectively.
        """
        data = self.feature_transform(data)
        data = self.model_score(data)

        return data
    
    def fit_min_periods(self, data:pd.DataFrame):
        """
        Learn min periods
        """
        data = data.set_index(self.date_column).sort_index().dropna()
        data['count'] = data.rolling(self.model_rp).count()
        self.model_mp = int(data['count'].quantile(self.model_mp_quantile))
        self.feature_mp = int(self.model_mp * (self.feature_rp / self.model_rp))
        self.raw_risk_mp = int(self.model_mp * (self.raw_risk_rp / self.model_rp))

        return self

    def feature_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the feature engineering
        """
        data = data.set_index(self.date_column).sort_index()
        data[self.feature_column] = data[self.feature_column].rolling(self.feature_rp, min_periods=self.feature_mp).agg(self.rolling_type)
        data.dropna(inplace=True)

        return data

    def model_fit(self, data: pd.DataFrame):
        """
        """
        # Fit a Minimum Covariance Determinant (MCD) robust estimator to data
        self.robust_cov = MinCovDet().fit(data[[self.feature_column]].dropna())
        data['Raw_Anomaly_Score'] = self.robust_cov.dist_
        del self.robust_cov.dist_
       
        # Apply range_filter
        if self.range_filter: data = self._range_filter(data, fit = True)

        if self.anomaly_threshold == "auto":
            # Apply rolling mean to get anomaly scores
            data["Anomaly_Score"] = data['Raw_Anomaly_Score'].rolling(self.model_rp, min_periods=self.model_mp).mean()
            df_anomaly = data["Anomaly_Score"].reset_index()        
            self.anomaly_threshold = df_anomaly["Anomaly_Score"].quantile(1 - self.contamination)
        
        return self

    def model_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        """
        data = data.copy()

        # Calculate Mahalanobis distance to get raw anomaly scores
        data['Raw_Anomaly_Score'] = self.robust_cov.mahalanobis(data[[self.feature_column]].dropna())

        # Apply range_filter
        if self.range_filter: data = self._range_filter(data, fit = False)

        # Apply rolling mean to get anomaly scores
        data["Anomaly_Score"] = data['Raw_Anomaly_Score'].rolling(self.model_rp, min_periods=self.model_mp).mean()

        df_anomaly = data["Anomaly_Score"].reset_index()

        return df_anomaly
    
    def _get_params(self):
        params = {"feature_mp": self.feature_mp,
                  "model_mp": self.model_mp,
                  "raw_risk_mp": self.raw_risk_mp,
                  "range_limit": str(pd.to_timedelta(self.range_limit, unit="s")) ,
                  "robust_mean": self.robust_cov.location_.tolist(),
                  "robust_cov": self.robust_cov.covariance_.tolist()
                  }
        
        return params


