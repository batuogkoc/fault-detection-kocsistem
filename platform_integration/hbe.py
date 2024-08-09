import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from .base import BaseDetector
from typing import Union

class HBE(BaseDetector):
    """Anomaly detector for one dimensional time series data."""
    def __init__(self, 
                 *,
                 date_column: str = "Date",
                 feature_column: str = "Value",
                 contamination: float = 0.1,
                 anomaly_threshold: Union[float, str] = "auto",
                 bins: int = 100,
                 model_mp_quantile: float = 0.02,
                 model_rp: str = "60min",
                 range_filter: bool = True,
                 range_rp: str = None,
                 raw_risk_rp: str = None,
                 risk_rp: str = None,
                 **kwargs):

        # Hyperparameters
        self.date_column = date_column
        self.feature_column = feature_column
        self.contamination = contamination
        self.anomaly_threshold = anomaly_threshold
        self.bins = bins
        self.model_mp_quantile = model_mp_quantile
        self.model_rp = pd.Timedelta(model_rp)
        self.range_filter = range_filter
        self.range_rp = pd.Timedelta(range_rp) if range_rp is not None else self.model_rp * 3 / 4
        self.raw_risk_rp = pd.Timedelta(raw_risk_rp) if raw_risk_rp is not None else self.model_rp * 2
        self.risk_rp = pd.Timedelta(risk_rp) if risk_rp is not None else self.model_rp * 6

        # Parameters
        self.model_mp: int  
        self.raw_risk_mp: int  
        self.anomaly_threshold: int 
        self.range_limit: float = None
        self.interp_func: interp1d
    
    def __str__(self):
        try:
            temp_dict = vars(self).copy()
            temp_dict["interp_func"] = "scipy.interpolate.interp1d"
            attributes = "\n".join(f"{key}: {value}" for key, value in temp_dict.items())
        except:
            print("Model is not trained")
            attributes = "\n".join(f"{key}: {value}" for key, value in temp_dict.items())
        return f"{self.__class__.__name__}:\n{attributes}"
    
    def __repr__(self):
        return self.__str__()

    def fit(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Implementation of pipeline_fit_score function that executes feature_fit_transform and model_fit_score functions
        respectively and return the resulting dataframe
        """
        self.fit_min_periods(data)
        self.model_fit(data)

        return self

    def anomaly_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Implementation of pipeline_score function, that applied feature_transform and model_score functions, respectively.
        """
        data = self.model_score(data)

        return data

    def fit_min_periods(self, data: pd.DataFrame):
        """
        """
        data = data.set_index(self.date_column).sort_index()        
        data['count'] = data.rolling(self.model_rp).count()
        data['count'] = np.where(data['count'] == 0, np.nan, data['count'])
        self.model_mp = int(data['count'].quantile(self.model_mp_quantile))
        self.raw_risk_mp = self.model_mp 
        # self.raw_risk_mp = int(self.model_mp * (self.raw_risk_rp / self.model_rp))

        return self
    
    def model_fit(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        """
        data = data.set_index(self.date_column).sort_index()
        
        # Create and normalize histogram
        hist, bin_edges = np.histogram(data[self.feature_column].dropna().values, bins=self.bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        pdf = hist / np.sum(hist)

        # Interpolate PDF
        self.interp_func = interp1d(bin_centers, pdf, kind='linear', fill_value="extrapolate")

        # Calculate scores
        data["Probabilities"] = self.interp_func(data[self.feature_column])
        data["Probabilities"] = np.where(data["Probabilities"] < 0, 0, data["Probabilities"])
        data["Raw_Anomaly_Score"] = - np.log10(data["Probabilities"] + 1e-20)
      
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
        data = data.set_index(self.date_column).sort_index()
        
        # Calculate scores
        data["Probabilities"] = self.interp_func(data[self.feature_column])
        data["Probabilities"] = np.where(data["Probabilities"] < 0, 0, data["Probabilities"])
        data["Raw_Anomaly_Score"] = -np.log10(data["Probabilities"] + 1e-20)
      
        # Apply range_filter
        if self.range_filter: data = self._range_filter(data, fit = False)

        # Apply rolling mean to get anomaly scores
        data["Anomaly_Score"] = data['Raw_Anomaly_Score'].rolling(self.model_rp, min_periods=self.model_mp).mean()

        df_anomaly = data["Anomaly_Score"].reset_index()

        return df_anomaly
    
    def _get_params(self):
        params = {"model_mp": self.model_mp,
                  "anomaly_threshold": self.anomaly_threshold,
                  "range_limit": str(pd.to_timedelta(self.range_limit, unit="s")) ,
                  }
        
        return params


