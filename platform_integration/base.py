import abc
import pandas as pd

class BaseDetector(abc.ABC):

    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def fit(self, data):
        pass

    @abc.abstractmethod
    def anomaly_score(self, data):
        pass

    @abc.abstractmethod
    def fit_min_periods(self, data):
        pass
    
    @abc.abstractmethod
    def _get_params(self, data):
        pass

    def _range_filter(self, data: pd.DataFrame, fit: bool = False):
        data = data.copy()
        data["range"] = data.index.to_series().diff().dt.total_seconds()
        data['max_diff'] = data['range'].rolling(self.range_rp).max()
        if fit: self.range_limit = data['max_diff'].quantile(.95)
        return data[data['max_diff'] <= self.range_limit]
    
    def risk_score(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Function that creates risk scores
        """
        data = data.set_index(self.date_column).sort_index()

        # Apply thresholding
        data['Binary_Anomaly_Score'] = (data['Anomaly_Score'] >= float(self.anomaly_threshold)).astype(int)

        # Apply rolling mean to get raw risk scores
        data["Raw_Risk_Score"] = data['Binary_Anomaly_Score'].rolling(self.raw_risk_rp, min_periods = self.raw_risk_mp).mean()

        # Apply rolling mean to get risk scores
        data["Risk_Score"] = data['Raw_Risk_Score'].rolling(self.risk_rp).mean()

        df_risk = data["Risk_Score"].reset_index()

        return df_risk
