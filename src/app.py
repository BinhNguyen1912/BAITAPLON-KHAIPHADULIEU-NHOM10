import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import gradio as gr


MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'best_tuned_XGBoost_model.joblib')
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, 'feature_names.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    features = None
    if os.path.exists(FEATURE_NAMES_PATH):
        features = joblib.load(FEATURE_NAMES_PATH)
        if not isinstance(features, list):
            features = list(features)
    elif hasattr(model, 'feature_names_in_'):
        features = model.feature_names_in_.tolist()
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    return model, features, scaler


def prepare_features_single(input_data: dict, feature_columns: list[str]) -> pd.DataFrame:
    df = pd.DataFrame([input_data])
    for c in feature_columns:
        if c not in df.columns:
            df[c] = 0
    df = df[feature_columns]
    df = df.fillna(0)
    return df


def predict_single_wrapper(model, feature_columns):
    def fn(
        OverallQual, GrLivArea, GarageCars, FullBath, BedroomAbvGr, YearBuilt, Fireplaces, LotArea,
        OverallCond, TotalBsmtSF, Flr1SF, Flr2SF, LotFrontage, HalfBath, BsmtFullBath, BsmtHalfBath,
        YearRemodAdd, OpenPorchSF, EnclosedPorch, ScreenPorch, TotRmsAbvGrd, GarageArea, ExterQualScore, IsCulDSac
    ):
        row = {
            'OverallQual': OverallQual,
            'GrLivArea': GrLivArea,
            'GarageCars': GarageCars,
            'FullBath': FullBath,
            'BedroomAbvGr': BedroomAbvGr,
            'YearBuilt': YearBuilt,
            'Fireplaces': Fireplaces,
            'LotArea': LotArea,
            'OverallCond': OverallCond,
            'TotalBsmtSF': TotalBsmtSF,
            '1stFlrSF': Flr1SF,
            '2ndFlrSF': Flr2SF,
            'LotFrontage': LotFrontage,
            'HalfBath': HalfBath,
            'BsmtFullBath': BsmtFullBath,
            'BsmtHalfBath': BsmtHalfBath,
            'YearRemodAdd': YearRemodAdd,
            'OpenPorchSF': OpenPorchSF,
            'EnclosedPorch': EnclosedPorch,
            'ScreenPorch': ScreenPorch,
            'TotRmsAbvGrd': TotRmsAbvGrd,
            'GarageArea': GarageArea,
            'ExterQualScore': ExterQualScore,
            'IsCulDSac': int(bool(IsCulDSac))
        }
        X = prepare_features_single(row, feature_columns)
        log_price = float(model.predict(X)[0])
        price = float(np.expm1(log_price))
        rmse_log = 0.1282
        lower = float(np.expm1(log_price - rmse_log))
        upper = float(np.expm1(log_price + rmse_log))
        return f"${price:,.0f} (≈ [{lower:,.0f} ; {upper:,.0f}])"
    return fn


def build_interface():
    model, feature_columns, _ = load_artifacts()

    with gr.Blocks(title="House Price Prediction (XGBoost)") as demo:
        gr.Markdown("### House Price Prediction")
        with gr.Row():
            with gr.Column():
                OverallQual = gr.Slider(1, 10, value=7, step=1, label="OverallQual")
                OverallCond = gr.Slider(1, 10, value=5, step=1, label="OverallCond")
                YearBuilt = gr.Number(value=2005, label="YearBuilt")
                YearRemodAdd = gr.Number(value=2005, label="YearRemodAdd")
                ExterQualScore = gr.Slider(1, 10, value=4, step=1, label="ExterQualScore")
                IsCulDSac = gr.Checkbox(value=False, label="IsCulDSac")
            with gr.Column():
                GrLivArea = gr.Number(value=1800, label="GrLivArea")
                LotArea = gr.Number(value=9000, label="LotArea")
                LotFrontage = gr.Number(value=60, label="LotFrontage")
                TotalBsmtSF = gr.Number(value=1000, label="TotalBsmtSF")
                Flr1SF = gr.Number(value=900, label="1stFlrSF")
                Flr2SF = gr.Number(value=0, label="2ndFlrSF")
            with gr.Column():
                BedroomAbvGr = gr.Slider(0, 8, value=3, step=1, label="BedroomAbvGr")
                FullBath = gr.Slider(0, 4, value=2, step=1, label="FullBath")
                HalfBath = gr.Slider(0, 2, value=0, step=1, label="HalfBath")
                BsmtFullBath = gr.Slider(0, 3, value=0, step=1, label="BsmtFullBath")
                BsmtHalfBath = gr.Slider(0, 2, value=0, step=1, label="BsmtHalfBath")
                TotRmsAbvGrd = gr.Slider(2, 15, value=6, step=1, label="TotRmsAbvGrd")
                Fireplaces = gr.Slider(0, 4, value=1, step=1, label="Fireplaces")
                GarageCars = gr.Slider(0, 5, value=2, step=1, label="GarageCars")
                GarageArea = gr.Number(value=550, label="GarageArea")
                OpenPorchSF = gr.Number(value=0, label="OpenPorchSF")
                EnclosedPorch = gr.Number(value=0, label="EnclosedPorch")
                ScreenPorch = gr.Number(value=0, label="ScreenPorch")

        predict_btn = gr.Button("Predict")
        output = gr.Textbox(label="Predicted SalePrice (with ±1 RMSE band)")

        predict_fn = predict_single_wrapper(model, feature_columns)
        predict_btn.click(
            predict_fn,
            inputs=[OverallQual, GrLivArea, GarageCars, FullBath, BedroomAbvGr, YearBuilt, Fireplaces, LotArea,
                    OverallCond, TotalBsmtSF, Flr1SF, Flr2SF, LotFrontage, HalfBath, BsmtFullBath, BsmtHalfBath,
                    YearRemodAdd, OpenPorchSF, EnclosedPorch, ScreenPorch, TotRmsAbvGrd, GarageArea, ExterQualScore, IsCulDSac],
            outputs=[output]
        )

    return demo


if __name__ == '__main__':
    demo = build_interface()
    demo.launch(share=True, server_name='0.0.0.0', server_port=7860)


