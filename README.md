# House Prices - Advanced Regression Techniques (Modularized)

Dự án đã được tách thành các module trong thư mục `src/` để tái sử dụng cho tiền xử lý, EDA, tạo đặc trưng, huấn luyện, đánh giá, và triển khai bằng Gradio.

## Yêu cầu

- Python 3.9+
- Hệ điều hành: Windows/Linux/macOS

## Cài đặt nhanh (Windows PowerShell)

```bash
cd D:\NHOM10
python -m venv .venv
.\.venv\Scripts\Activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Chuẩn bị dữ liệu

- Đặt `train.csv` và `test.csv` ở thư mục gốc dự án (cùng cấp với `src/`).

## Chạy ứng dụng (Gradio)

```bash
python -m src.app
```

- Truy cập liên kết hiển thị trên console hoặc `http://localhost:7860`.
- Ứng dụng yêu cầu các file model đã huấn luyện:
  - `models/best_tuned_XGBoost_model.joblib`
  - `models/feature_names.pkl`
  - (tuỳ chọn) `models/scaler.joblib`

Nếu chưa có model, bạn có thể tự huấn luyện (tham khảo bên dưới) hoặc yêu cầu tạo file `run_train.py` để tự động train và lưu artifacts.

## Huấn luyện (tham khảo nhanh)

```python
import os
from joblib import dump
from sklearn.model_selection import train_test_split
from src.preprocessing import load_raw, log_transform_target, merge_for_processing, impute_missing_values, one_hot_encode, split_train_test
from src.feature_engineering import engineer_features
from src.model_trainning import get_models, evaluate_models, pick_best

train_df, test_df = load_raw('train.csv', 'test.csv')
_ = log_transform_target(train_df, 'SalePrice')
all_data = merge_for_processing(train_df, test_df)
all_data = impute_missing_values(all_data)
all_data = engineer_features(all_data)
all_data = one_hot_encode(all_data)
X, y, X_test = split_train_test(all_data, len(train_df), train_df, 'SalePrice')

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
models = get_models()
results = evaluate_models(models, X_tr, X_val, y_tr, y_val, cv_folds=5)
best_name, best_model, perf = pick_best(results, metric='rmse')

os.makedirs('models', exist_ok=True)
dump(best_model, f'models/best_tuned_{best_name}_model.joblib')
dump(list(X.columns), 'models/feature_names.pkl')
```

## Cấu trúc dự án

```
D:\NHOM10
├─ src/
│  ├─ __init__.py
│  ├─ preprocessing.py          # Load/merge, imputation, one-hot, split
│  ├─ eda.py                    # Phân phối, thiếu dữ liệu, tương quan
│  ├─ feature_engineering.py    # Tạo features + pipeline engineer_features
│  ├─ model_trainning.py        # Models, đánh giá, chọn best
│  ├─ evaluation.py             # Metrics và biểu đồ đánh giá
│  └─ app.py                    # Ứng dụng Gradio
├─ models/                      # Lưu model và artifacts (tạo sau khi train)
├─ train.csv
├─ test.csv
├─ requirements.txt
└─ README.md
```

## Ghi chú

- XGBoost/LightGBM có thể cần công cụ build (VC++ Build Tools trên Windows hoặc gcc trên Linux) nếu cài đặt gặp lỗi.
- Nếu chạy `src.app` báo thiếu model, hãy huấn luyện và lưu vào thư mục `models/` như hướng dẫn.
