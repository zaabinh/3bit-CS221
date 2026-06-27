# Hướng dẫn mô hình MPNet SentenceTransformer

### 1. Tổng quan

Pipeline phân loại khía cạnh bằng MPNet gồm hai phiên bản encoder:

- **Encoder chưa huấn luyện lại**: mô hình gốc `all-mpnet-base-v2`.
- **Encoder đã huấn luyện lại**: `all-mpnet-base-v2` được fine-tune trên đánh giá nhà hàng bằng các cặp văn bản có độ tương đồng mục tiêu được tính từ mức độ giao nhau của nhãn đa lớp.

Mỗi encoder có một mô hình One-vs-Rest Logistic Regression, bộ ngưỡng phân loại và metadata riêng. Không được dùng encoder của một phiên bản với classifier của phiên bản còn lại.

Thứ tự các nhãn luôn là:

```python
["food", "price", "service", "ambiance", "miscellaneous"]
```

### 2. Vị trí các artifact

```text
models/sbert/
├── encoder_training_metadata.json
├── unretrained_encoder/
├── retrained_encoder/
└── classifiers/
    ├── unretrained/
    │   ├── model.joblib
    │   └── metadata.json
    └── retrained/
        ├── model.joblib
        └── metadata.json
```

Kết quả của quá trình thí nghiệm được lưu tại:

```text
outputs/SBERT/
├── data_split.npz
├── logistic_grid_results.csv
├── classifier_validation_summary.csv
└── encoder_comparison.csv              # Được tạo sau khi chạy notebook đánh giá
```

Các notebook tạo ra những artifact trên:

```text
notebooks/SBERT/01-sbert_retraining.ipynb
notebooks/SBERT/02-sbert_ovr_logistic_regression.ipynb
notebooks/SBERT/03-sbert_evaluate.ipynb
```

Khi huấn luyện lại toàn bộ mô hình, hãy chạy theo đúng thứ tự trên.

### 3. Cấu hình mô hình hiện tại

#### Encoder chưa huấn luyện lại

- Encoder: `models/sbert/unretrained_encoder`
- Classifier: `models/sbert/classifiers/unretrained/model.joblib`
- Embedding được chuẩn hóa L2.
- Logistic Regression: `C=10.0`, `class_weight="balanced"`, `solver="liblinear"`

Ngưỡng phân loại:

```python
{
    "food": 0.37,
    "price": 0.48,
    "service": 0.55,
    "ambiance": 0.69,
    "miscellaneous": 0.21,
}
```

#### Encoder đã huấn luyện lại

- Encoder: `models/sbert/retrained_encoder`
- Classifier: `models/sbert/classifiers/retrained/model.joblib`
- Embedding được chuẩn hóa L2.
- Logistic Regression: `C=1.0`, `class_weight="balanced"`, `solver="lbfgs"`

Ngưỡng phân loại:

```python
{
    "food": 0.22,
    "price": 0.16,
    "service": 0.44,
    "ambiance": 0.72,
    "miscellaneous": 0.32,
}
```

Phiên bản đã huấn luyện lại hiện có macro F1 trên validation cao hơn (`0.8489` so với `0.8261`). Sau khi chạy notebook đánh giá, hãy dùng kết quả test độc lập trong `outputs/SBERT/encoder_comparison.csv` để quyết định phiên bản triển khai cuối cùng.

### 4. Tiền xử lý

MPNet SentenceTransformer pipeline được huấn luyện trực tiếp bằng văn bản tiếng Anh trong:

```text
data/Restaurant_ABSA_processed.csv -> review_en
```

Không thực hiện loại bỏ stopword, lemmatization, loại bỏ dấu câu, chuyển thành chữ thường hoặc gọi `preprocess_review()` trước khi tạo SentenceTransformer embedding.

Trong môi trường triển khai:

1. Kiểm tra đầu vào là chuỗi.
2. Loại bỏ khoảng trắng ở đầu và cuối.
3. Từ chối chuỗi rỗng.
4. Đưa trực tiếp câu tiếng Anh tự nhiên vào MPNet.

```python
def prepare_review(review: str) -> str:
    if not isinstance(review, str):
        raise TypeError("Review must be a string")

    review = review.strip()
    if not review:
        raise ValueError("Review cannot be empty")

    return review
```

Không sử dụng `preprocessing.preprocess_review()` với các MPNet SentenceTransformer classifier đã lưu. Hàm đó loại bỏ stopword và có thể loại bỏ từ phủ định, làm câu `"food was not good"` trở thành văn bản gần giống `"food good"`.

Dữ liệu huấn luyện hiện tại là tiếng Anh. Với dữ liệu đầu vào không phải tiếng Anh, cần dịch sang tiếng Anh trước khi dự đoán hoặc huấn luyện lại toàn bộ pipeline bằng SentenceTransformer đa ngôn ngữ.

### 5. Tải model bundle đã lưu

Đặt `variant` thành `"retrained"` hoặc `"unretrained"`.

```python
from pathlib import Path
import json

import joblib
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path.cwd()
variant = "retrained"

bundle_dir = PROJECT_ROOT / "models" / "sbert" / "classifiers" / variant

with (bundle_dir / "metadata.json").open(encoding="utf-8") as file:
    metadata = json.load(file)

encoder_path = PROJECT_ROOT / metadata["encoder_path"]
encoder = SentenceTransformer(str(encoder_path))
classifier = joblib.load(bundle_dir / "model.joblib")
```

Nếu ứng dụng có thể được chạy từ thư mục khác thư mục gốc của repository, cần cấu hình `PROJECT_ROOT` rõ ràng thay vì phụ thuộc vào `Path.cwd()`.

### 6. Hàm dự đoán

Sử dụng hàm `predict_reviews()` ở phần tiếng Anh. Hàm này thực hiện đầy đủ các bước:

1. Kiểm tra và chuẩn hóa khoảng trắng của đầu vào.
2. Tạo SentenceTransformer embedding với đúng thiết lập `normalize_embeddings`.
3. Lấy xác suất từ classifier.
4. Áp dụng ngưỡng riêng của từng nhãn từ `metadata.json`.
5. Chọn nhãn có xác suất cao nhất nếu không nhãn nào vượt ngưỡng.
6. Trả về nhãn dự đoán và xác suất của tất cả các khía cạnh.

Ví dụ:

```python
predictions = predict_reviews([
    "The food was excellent but the service was slow.",
    "The restaurant has a relaxing atmosphere and reasonable prices.",
])

for prediction in predictions:
    print(prediction)
```

### 7. Lưu ý khi triển khai

- Chỉ tải encoder và classifier một lần khi ứng dụng khởi động.
- Sử dụng batch khi dự đoán nhiều review.
- Giữ nguyên `normalize_embeddings` theo `metadata.json`.
- Đọc thứ tự nhãn và ngưỡng từ metadata, không hard-code trong ứng dụng.
- Chỉ tải file Joblib đáng tin cậy vì quá trình giải tuần tự Joblib có thể thực thi mã.
- Luôn sử dụng encoder, classifier và metadata thuộc cùng một phiên bản.
- Nên đánh version cho mỗi model bundle mới để có thể rollback khi cần.
