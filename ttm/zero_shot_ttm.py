"""
Zero-Shot Granite TTM for EV SoH Forecasting
Improved version returning full forecast horizon
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import tempfile
import numpy as np
import pandas as pd
import torch

from transformers import Trainer, TrainingArguments, set_seed
from tsfm_public import TimeSeriesPreprocessor, get_datasets
from tsfm_public.toolkit.get_model import get_model


# ============================================================
# Configuration
# ============================================================

SEED = 42
set_seed(SEED)

TTM_MODEL_PATH = "ibm-granite/granite-timeseries-ttm-r1"

CONTEXT_LENGTH = 512
PREDICTION_LENGTH = 96


class ZeroShotTTM:
    """
    Zero-Shot Granite TTM wrapper for EV SoH forecasting
    """

    def __init__(self,
                 context_length=CONTEXT_LENGTH,
                 prediction_length=PREDICTION_LENGTH):

        self.context_length = context_length
        self.prediction_length = prediction_length

        self.column_specifiers = {
            "timestamp_column": "time_idx",
            "id_columns": [],
            "target_columns": ["soh"],
            "control_columns": [],
        }

       
        self.model = get_model(
            TTM_MODEL_PATH,
            context_length=context_length,
            prediction_length=prediction_length,
            freq_prefix_tuning=False,
            freq=None,
            prefer_l1_loss=False,
            prefer_longer_context=True,
        )

        self.temp_dir = tempfile.mkdtemp()

        self.trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir=self.temp_dir,
                per_device_eval_batch_size=16,
                seed=SEED,
                report_to="none",
                disable_tqdm=True,
                dataloader_pin_memory=False,
            ),
        )

    # ============================================================
    # Predict future SoH
    # ============================================================

    def predict(self, soh_history):
        """
        Returns full forecast horizon (length = prediction_length)
        """

        if len(soh_history) < self.context_length:
            return np.array([soh_history[-1]] * self.prediction_length)

        df = pd.DataFrame({
            "time_idx": np.arange(len(soh_history)),
            "soh": soh_history
        })

        split_config = {
            "train": [0, len(df) - self.prediction_length],
            "valid": [0, len(df) - self.prediction_length],
            "test": [0, len(df)]
        }

        tsp = TimeSeriesPreprocessor(
            **self.column_specifiers,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            scaling=False,
            encode_categorical=False,
            scaler_type="standard",
        )

        _, _, dset_test = get_datasets(
            tsp,
            df,
            split_config,
            use_frequency_token=self.model.config.resolution_prefix_tuning,
        )

        predictions = self.trainer.predict(dset_test)

        forecast = predictions.predictions[0]

        predicted_future = forecast[0, :, 0]
        
                
        current_soh = soh_history[-1]

        
        predicted_future = np.minimum(predicted_future, current_soh)

        
        predicted_future = np.minimum.accumulate(predicted_future)

        return predicted_future  # ← FULL 96-step forecast

    # ============================================================
    # Convenience: Mean Forecast
    # ============================================================

    def predict_mean(self, soh_history):
        """
        Returns average predicted SoH over forecast horizon
        """
        forecast = self.predict(soh_history)
        return float(np.mean(forecast))