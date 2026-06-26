# Hướng dẫn pipeline huấn luyện SBERT

## 1. Mục tiêu

Thư mục `notebooks/SBERT` xây dựng pipeline phân loại đa nhãn các khía cạnh được đề cập trong đánh giá nhà hàng:

```python
["food", "price", "service", "ambiance", "miscellaneous"]
```

Pipeline chính gồm ba giai đoạn và một giai đoạn cải thiện chuyên biệt:

1. Tạo tập dữ liệu cố định và fine-tune SBERT.
2. Huấn luyện, tối ưu Logistic Regression trên embedding của từng encoder.
3. Đánh giá hai pipeline trên cùng một tập test và phân tích lỗi.
4. Thử các chiến lược riêng để cải thiện lớp `miscellaneous`.

Hai pipeline được so sánh:

```text
Review
  ├── SBERT gốc ────────────> OVR Logistic Regression
  └── SBERT đã fine-tune ───> OVR Logistic Regression
                                  │
                                  └── Ngưỡng riêng cho từng aspect
```

Các notebook phải được chạy theo thứ tự:

```text
notebooks/SBERT/01-sbert_retraining.ipynb
notebooks/SBERT/02-sbert_ovr_logistic_regression.ipynb
notebooks/SBERT/03-sbert_evaluate.ipynb
notebooks/SBERT/04-improve_mischellaneous.ipynb
```

## 2. Dữ liệu và tiền xử lý

Nguồn dữ liệu:

```text
data/Restaurant_ABSA_processed.csv
```

Pipeline sử dụng:

```python
TEXT_COL = "review_en"
ASPECT_COLS = [
    "food",
    "price",
    "service",
    "ambiance",
    "miscellaneous",
]
```

Văn bản trong `review_en` được đưa trực tiếp vào SBERT. Pipeline này không thực hiện:

- Loại bỏ stopword.
- Lemmatization.
- Loại bỏ dấu câu.
- Chuyển toàn bộ thành chữ thường.
- Gọi `preprocessing.preprocess_review()`.

Điều này phù hợp với SentenceTransformer vì mô hình cần câu tự nhiên và ngữ cảnh đầy đủ. Ví dụ, loại stopword có thể làm mất từ phủ định `not` và thay đổi ý nghĩa câu.

Embedding được chuẩn hóa L2 bằng:

```python
normalize_embeddings=True
```

Thiết lập này phải giống nhau trong lúc huấn luyện classifier, đánh giá và triển khai.

---

## 3. Notebook 01 — Fine-tune SBERT

File:

```text
notebooks/SBERT/01-sbert_retraining.ipynb
```

### 3.1. Nhiệm vụ chính

Notebook này:

1. Thiết lập seed để có thể tái lập kết quả.
2. Tìm thư mục gốc của project.
3. Đọc và kiểm tra dữ liệu.
4. Chia dữ liệu thành train, validation và test.
5. Tạo các cặp review để fine-tune SBERT.
6. Lưu encoder gốc.
7. Fine-tune và lưu encoder mới.
8. Lưu metadata của quá trình huấn luyện.

### 3.2. Thiết lập seed

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
```

Seed được đặt cho Python, NumPy và PyTorch nhằm giảm sự thay đổi giữa các lần chạy.

Kết quả vẫn có thể khác nhẹ trên GPU do một số phép toán không hoàn toàn deterministic.

### 3.3. Tìm thư mục project

Notebook có thể được mở từ nhiều working directory khác nhau. Đoạn mã sau đi ngược lên các thư mục cha cho đến khi tìm thấy dữ liệu:

```python
path = Path.cwd().resolve()

for _ in range(8):
    if (path / "data" / "Restaurant_ABSA_processed.csv").exists():
        PROJECT_ROOT = path
        break
    path = path.parent
```

Các đường dẫn quan trọng:

```python
DATA_PATH = PROJECT_ROOT / "data" / "Restaurant_ABSA_processed.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "SBERT"
MODEL_DIR = PROJECT_ROOT / "models" / "sbert"

BASE_ENCODER_DIR = MODEL_DIR / "unretrained_encoder"
RETRAINED_ENCODER_DIR = MODEL_DIR / "retrained_encoder"
SPLIT_PATH = OUTPUT_DIR / "data_split.npz"
```

### 3.4. Kiểm tra nhãn

Notebook kiểm tra mỗi cột aspect chỉ chứa `0` hoặc `1`:

```python
invalid_labels = {
    aspect: sorted(df[aspect].dropna().unique().tolist())
    for aspect in ASPECT_COLS
    if not set(df[aspect].dropna().unique()).issubset({0, 1})
}
```

Nếu tồn tại giá trị khác, notebook dừng lại bằng `ValueError`. Việc này tránh huấn luyện trên nhãn sai định dạng.

### 3.5. Chia dữ liệu

Đầu tiên, 20% dữ liệu được dành cho test:

```python
train_val_idx, test_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=SEED,
)
```

Sau đó 20% tổng dữ liệu được dành cho validation:

```python
train_idx, val_idx = train_test_split(
    train_val_idx,
    test_size=0.25,
    random_state=SEED,
)
```

Tỷ lệ cuối cùng:

| Tập | Tỷ lệ |
|---|---:|
| Train | 60% |
| Validation | 20% |
| Test | 20% |

Các index được lưu vào:

```text
outputs/SBERT/data_split.npz
```

Tất cả notebook sau phải dùng lại file này. Không nên chia dữ liệu lại ở từng notebook vì hai mô hình sẽ không còn được đánh giá trên cùng tập mẫu.

### 3.6. Jaccard similarity giữa hai bộ nhãn

Mỗi review có một vector đa nhãn, ví dụ:

```python
[1, 1, 0, 0, 0]  # food + price
```

Hàm:

```python
def jaccard_similarity(left, right):
    intersection = np.logical_and(left, right).sum()
    union = np.logical_or(left, right).sum()
    return float(intersection / union) if union else 0.0
```

Tính:

```text
Jaccard(A, B) = số nhãn chung / tổng số nhãn khác nhau
```

Ví dụ:

```text
A = {food, price}
B = {food, service}

intersection = {food}
union = {food, price, service}
similarity = 1 / 3
```

Giá trị này được dùng làm mục tiêu cho `CosineSimilarityLoss`.

### 3.7. Tạo cặp huấn luyện

Hàm `build_training_pairs()` tạo hai nhóm:

- Positive candidate: hai review có ít nhất một nhãn chung.
- Negative candidate: hai review không có nhãn chung.

Với mỗi review, notebook lấy tối đa sáu cặp:

```python
pairs_per_sample=6
```

Mỗi cặp được chuyển thành:

```python
InputExample(
    texts=[review_a, review_b],
    label=jaccard_similarity,
)
```

Kết quả hiện tại:

```text
2880 training pairs
```

Mục tiêu của quá trình này là đưa các review có aspect tương tự đến gần nhau hơn trong không gian embedding và đẩy các review không liên quan ra xa.

### 3.8. Lưu encoder gốc

```python
BASE_MODEL_NAME = "all-mpnet-base-v2"

base_encoder = SentenceTransformer(BASE_MODEL_NAME)
base_encoder.save(str(BASE_ENCODER_DIR))
```

Encoder chưa fine-tune được lưu tại:

```text
models/sbert/unretrained_encoder/
```

Việc lưu encoder gốc giúp:

- So sánh công bằng với encoder fine-tune.
- Không phụ thuộc Internet khi đánh giá hoặc triển khai.
- Đảm bảo đúng cùng một phiên bản mô hình nền.

### 3.9. Fine-tune encoder

Cấu hình:

```python
EPOCHS = 3
BATCH_SIZE = 16
WARMUP_RATIO = 0.10
```

DataLoader:

```python
train_loader = DataLoader(
    train_examples,
    shuffle=True,
    batch_size=BATCH_SIZE,
)
```

Loss:

```python
train_loss = losses.CosineSimilarityLoss(retrained_encoder)
```

`CosineSimilarityLoss` tối ưu cosine similarity giữa embedding của hai review sao cho gần với Jaccard similarity của bộ nhãn.

Warmup:

```python
warmup_steps = int(
    len(train_loader) * EPOCHS * WARMUP_RATIO
)
```

Warmup tăng learning rate dần trong 10% số bước đầu tiên, giúp quá trình fine-tune ổn định hơn.

Huấn luyện:

```python
retrained_encoder.fit(
    train_objectives=[(train_loader, train_loss)],
    epochs=EPOCHS,
    warmup_steps=warmup_steps,
    show_progress_bar=True,
)
```

Encoder sau fine-tune được lưu tại:

```text
models/sbert/retrained_encoder/
```

### 3.10. Metadata

Thông tin huấn luyện được lưu tại:

```text
models/sbert/encoder_training_metadata.json
```

Metadata hiện tại chứa:

- Base model: `all-mpnet-base-v2`
- Epochs: `3`
- Batch size: `16`
- Warmup ratio: `0.1`
- Training pairs: `2880`
- Seed: `42`
- Cột văn bản và thứ tự nhãn

---

## 4. Notebook 02 — Huấn luyện OVR Logistic Regression

File:

```text
notebooks/SBERT/02-sbert_ovr_logistic_regression.ipynb
```

### 4.1. Nhiệm vụ chính

Notebook này:

1. Tải encoder gốc và encoder đã fine-tune.
2. Tải lại đúng train/validation/test split.
3. Tạo embedding cho train và validation.
4. Grid search Logistic Regression.
5. Chọn cấu hình tốt nhất theo macro F1.
6. Tối ưu ngưỡng riêng cho từng aspect.
7. Lưu classifier và metadata của từng encoder.

Test set chỉ được đọc index để báo kích thước, không được dùng để chọn tham số.

### 4.2. Hai encoder được đánh giá

```python
ENCODER_PATHS = {
    "unretrained": MODEL_DIR / "unretrained_encoder",
    "retrained": MODEL_DIR / "retrained_encoder",
}
```

Mỗi encoder tạo ra một pipeline độc lập:

```text
unretrained encoder + unretrained classifier
retrained encoder   + retrained classifier
```

Không được tráo classifier giữa hai encoder vì không gian embedding đã thay đổi sau fine-tune.

### 4.3. Grid tham số

```python
GRID = {
    "C": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    "class_weight": [None, "balanced"],
    "solver": ["liblinear", "lbfgs"],
}
```

Ý nghĩa:

- `C`: nghịch đảo độ mạnh regularization.
  - `C` nhỏ: regularization mạnh hơn.
  - `C` lớn: mô hình bám sát dữ liệu train hơn.
- `class_weight="balanced"`: tăng trọng số cho lớp dương hiếm.
- `solver`: thuật toán tối ưu Logistic Regression.
- `max_iter=3000`: giới hạn số vòng lặp, đủ lớn để mô hình hội tụ.

Tổng số cấu hình:

```text
8 × 2 × 2 = 32
```

### 4.4. One-vs-Rest

```python
model = OneVsRestClassifier(
    LogisticRegression(...),
    n_jobs=1,
)
```

Với năm aspect, One-vs-Rest huấn luyện năm bài toán nhị phân độc lập:

```text
food vs not-food
price vs not-price
service vs not-service
ambiance vs not-ambiance
miscellaneous vs not-miscellaneous
```

Một review có thể được dự đoán nhiều aspect cùng lúc.

### 4.5. Hàm `fit_one()`

Hàm này:

1. Nhận một bộ tham số.
2. Huấn luyện OVR Logistic Regression.
3. Lấy xác suất trên validation.
4. Tạm thời dùng ngưỡng `0.5`.
5. Tính macro F1, micro F1 và F1 của từng aspect.

Macro F1:

```python
f1_score(
    y_val,
    predictions,
    average="macro",
)
```

Macro F1 cho trọng số bằng nhau cho mọi aspect. Vì vậy `miscellaneous` không bị che lấp bởi các lớp phổ biến như `food`.

Micro F1:

```python
f1_score(
    y_val,
    predictions,
    average="micro",
)
```

Micro F1 gom toàn bộ TP, FP và FN của các lớp. Chỉ số này phản ánh hiệu suất tổng thể nhưng thường thiên về lớp phổ biến.

### 4.6. Grid search bằng multiprocessing

```python
with parallel_config(
    backend="loky",
    n_jobs=N_JOBS,
    inner_max_num_threads=1,
):
```

Trong đó:

```python
N_JOBS = min(4, os.cpu_count() or 1)
```

`loky` chạy mỗi cấu hình trong một process riêng.

`inner_max_num_threads=1` tránh trường hợp mỗi process tự tạo thêm nhiều BLAS threads, gây CPU oversubscription và tốn RAM.

Mỗi classifier cũng dùng:

```python
n_jobs=1
```

để tránh parallelism lồng nhau.

Kết quả được sắp xếp theo:

```python
["macro_f1", "micro_f1"]
```

với macro F1 là tiêu chí chính.

### 4.7. Tối ưu ngưỡng theo từng aspect

Ngưỡng mặc định `0.5` không nhất thiết phù hợp cho mọi lớp, đặc biệt là lớp mất cân bằng.

Hàm `tune_thresholds()` thử:

```python
np.arange(0.05, 0.951, 0.01)
```

Với mỗi aspect:

1. Lấy xác suất của riêng aspect đó.
2. Thử từng ngưỡng.
3. Tính binary F1 trên validation.
4. Chọn ngưỡng có F1 cao nhất.

Ngưỡng chỉ được tối ưu trên validation, không được dùng test set.

### 4.8. Áp dụng ngưỡng

```python
predictions[:, index] = (
    probabilities[:, index] >= thresholds[aspect]
)
```

Nếu không có aspect nào vượt ngưỡng:

```python
predictions[row, np.argmax(probabilities[row])] = 1
```

Fallback này đảm bảo mỗi review luôn có ít nhất một nhãn.

Pipeline hiện tại không giới hạn `top_k` và không loại `miscellaneous` khi xuất hiện cùng lớp khác.

### 4.9. Lưu classifier

Mỗi encoder có một thư mục riêng:

```text
models/sbert/classifiers/unretrained/
├── model.joblib
└── metadata.json

models/sbert/classifiers/retrained/
├── model.joblib
└── metadata.json
```

`model.joblib` chứa mô hình OVR Logistic Regression.

`metadata.json` chứa:

- Đường dẫn encoder.
- Thứ tự aspect.
- Cấu hình chuẩn hóa embedding.
- Batch size.
- Tham số Logistic Regression.
- Ngưỡng từng aspect.
- F1 của từng ngưỡng trên validation.
- Macro F1 và micro F1 trên validation.

### 4.10. Cấu hình tốt nhất hiện tại

Encoder gốc:

```python
{
    "C": 10.0,
    "class_weight": "balanced",
    "solver": "liblinear",
}
```

Encoder fine-tune:

```python
{
    "C": 1.0,
    "class_weight": "balanced",
    "solver": "lbfgs",
}
```

Validation:

| Encoder | Macro F1 | Micro F1 |
|---|---:|---:|
| Unretrained | 0.8261 | 0.9085 |
| Retrained | 0.8489 | 0.9296 |

Các file kết quả:

```text
outputs/SBERT/logistic_grid_results.csv
outputs/SBERT/classifier_validation_summary.csv
```

---

## 5. Notebook 03 — Đánh giá và phân tích lỗi

File:

```text
notebooks/SBERT/03-sbert_evaluate.ipynb
```

### 5.1. Nhiệm vụ chính

Notebook này:

1. Tải hai encoder.
2. Tải hai classifier và metadata tương ứng.
3. Lấy tập test từ split đã lưu.
4. Tạo embedding và dự đoán.
5. So sánh các metric.
6. Vẽ confusion matrix.
7. Phân tích mẫu đúng, sai, tốt nhất và tệ nhất của retrained model.
8. Phân tích false positive và false negative theo aspect.

Notebook này không tối ưu thêm tham số. Test set chỉ được dùng để đánh giá cuối cùng.

### 5.2. Hàm `evaluate_bundle()`

Hàm tải một model bundle:

```python
classifier = joblib.load(bundle_dir / "model.joblib")
metadata = json.load(...)
encoder = SentenceTransformer(str(encoder_path))
```

Sau đó:

1. Tạo embedding cho toàn bộ test set.
2. Lấy xác suất từ classifier.
3. Áp dụng ngưỡng trong metadata.
4. Tính metric.
5. Đo thời gian dự đoán.

Các metric:

- `macro_f1`: trung bình F1 của các aspect.
- `micro_f1`: F1 tổng trên mọi dự đoán nhị phân.
- `samples_f1`: tính F1 riêng cho từng review rồi lấy trung bình.
- `hamming_loss`: tỷ lệ nhãn bị dự đoán sai.
- F1 riêng của từng aspect.
- Thời gian tổng.
- Millisecond trên mỗi review.

### 5.3. Kết quả test hiện tại

File:

```text
outputs/SBERT/encoder_comparison.csv
```

| Metric | Retrained | Unretrained |
|---|---:|---:|
| Macro F1 | 0.8296 | 0.7837 |
| Micro F1 | 0.9231 | 0.8785 |
| Samples F1 | 0.9215 | 0.8790 |
| Hamming loss | 0.0613 | 0.0975 |
| Food F1 | 0.9653 | 0.9569 |
| Price F1 | 0.9873 | 0.9677 |
| Service F1 | 0.9574 | 0.9263 |
| Ambiance F1 | 0.9524 | 0.8889 |
| Miscellaneous F1 | 0.2857 | 0.1786 |

Encoder fine-tune tốt hơn trên toàn bộ metric chính, nhưng `miscellaneous` vẫn có F1 thấp.

### 5.4. Classification report và confusion matrix

Notebook tạo classification report cho mỗi encoder:

```text
outputs/SBERT/retrained_classification_report.csv
outputs/SBERT/unretrained_classification_report.csv
```

`multilabel_confusion_matrix()` tạo một confusion matrix riêng cho từng aspect:

```text
[[TN, FP],
 [FN, TP]]
```

Confusion matrix giúp xác định:

- Aspect bị dự đoán dư nhiều: FP cao.
- Aspect thường bị bỏ sót: FN cao.
- Ngưỡng hiện tại quá thấp hoặc quá cao.

### 5.5. Phân tích theo từng review

Notebook tạo `retrained_samples_df`, mỗi dòng chứa:

- Index trong test set.
- Index trong dataset.
- Nội dung review.
- Nhãn thật.
- Nhãn dự đoán.
- Xác suất từng aspect.
- Exact match.
- Sample-level F1.
- False positive.
- False negative.
- Độ tin cậy trung bình của nhãn được chọn.
- Xác suất sai cao nhất.

### 5.6. Exact match

Một review là exact match khi toàn bộ vector dự đoán giống vector nhãn thật:

```python
np.array_equal(true_row, predicted_row)
```

Ví dụ:

```text
Nhãn thật:    [food, price]
Nhãn dự đoán: [food, price]
=> exact match
```

```text
Nhãn thật:    [food, price]
Nhãn dự đoán: [food]
=> không exact match
```

### 5.7. Sample-level F1

Notebook tính F1 riêng cho từng review:

```text
sample_f1 = 2TP / (2TP + FP + FN)
```

Chỉ số này cho biết mức độ đúng của tập nhãn trên từng mẫu.

Ví dụ:

```text
Nhãn thật:    [food, price]
Nhãn dự đoán: [food, service]

TP = 1
FP = 1
FN = 1
sample F1 = 0.5
```

### 5.8. Best cases

Best cases được chọn từ các exact match và sắp xếp theo:

```python
predicted_confidence
```

Đây là trung bình xác suất của các nhãn được dự đoán dương.

Các mẫu này cho thấy:

- Cách diễn đạt mà mô hình nhận diện rõ.
- Những aspect có từ khóa hoặc ngữ nghĩa ổn định.
- Các tổ hợp nhãn mà mô hình học tốt.

Kết quả được lưu tại:

```text
outputs/SBERT/retrained_best_cases.csv
```

### 5.9. Worst cases

Các mẫu sai được sắp xếp theo:

1. Sample F1 thấp nhất.
2. Số lỗi FP + FN cao nhất.
3. Xác suất dành cho quyết định sai cao nhất.

```python
incorrect_samples.sort_values(
    [
        "sample_f1",
        "error_count",
        "maximum_wrong_confidence",
    ],
    ascending=[True, False, False],
)
```

Những trường hợp này thường cho thấy:

- Nhãn dữ liệu không nhất quán.
- Câu dịch khó hiểu.
- Aspect `miscellaneous` quá rộng.
- Review đề cập gián tiếp.
- Ngưỡng chưa phù hợp.
- Mẫu hiếm hoặc cách diễn đạt chưa xuất hiện đủ trong train set.

Kết quả:

```text
outputs/SBERT/retrained_worst_cases.csv
outputs/SBERT/retrained_sample_predictions.csv
```

### 5.10. Phân tích lỗi theo aspect

Với mỗi aspect, notebook đếm:

- True positive.
- False positive.
- False negative.
- True negative.
- F1.
- Loại lỗi chính.

Quy tắc diễn giải:

- FP cao hơn FN: mô hình dự đoán aspect quá thường xuyên; có thể cân nhắc tăng ngưỡng.
- FN cao hơn FP: mô hình thường bỏ sót aspect; có thể cân nhắc giảm ngưỡng hoặc bổ sung dữ liệu.
- F1 thấp với support nhỏ: ưu tiên kiểm tra chất lượng và số lượng dữ liệu trước khi chỉnh ngưỡng.

Kết quả:

```text
outputs/SBERT/retrained_aspect_error_analysis.csv
```

Notebook cũng vẽ biểu đồ so sánh FP và FN của từng aspect.

---

## 6. Notebook 04 — Cải thiện lớp `miscellaneous`

File:

```text
notebooks/SBERT/04-improve_mischellaneous.ipynb
```

Tên file đang giữ cách viết `mischellaneous` để khớp với notebook hiện tại. Tên nhãn trong dữ liệu và mã nguồn vẫn là:

```python
"miscellaneous"
```

### 6.1. Mục tiêu

Kết quả từ notebook 03 cho thấy lớp `miscellaneous` của retrained model có:

```text
TP = 6
FP = 23
FN = 7
F1 = 0.2857
```

Số false positive cao hơn nhiều false negative. Điều này cho thấy mô hình đang gán `miscellaneous` cho quá nhiều review.

Notebook 04 thử bốn hướng:

1. Chỉ tối ưu lại threshold của classifier hiện tại.
2. Huấn luyện classifier nhị phân riêng với class weight tùy chỉnh.
3. Tăng trọng số cho các hard negative.
4. Dùng xác suất của bốn aspect còn lại làm context feature.

Mọi chiến lược được chọn bằng validation set. Test set chỉ được dùng sau khi chiến lược thắng đã được xác định.

### 6.2. Tải retrained pipeline

Notebook sử dụng:

```text
models/sbert/retrained_encoder/
models/sbert/classifiers/retrained/model.joblib
models/sbert/classifiers/retrained/metadata.json
outputs/SBERT/data_split.npz
```

Encoder và classifier gốc không bị ghi đè. Mô hình cải thiện được lưu thành một component riêng.

Các index:

```python
MISC_INDEX = ASPECT_COLS.index("miscellaneous")

OTHER_INDICES = [
    index
    for index, aspect in enumerate(ASPECT_COLS)
    if aspect != "miscellaneous"
]
```

`MISC_INDEX` xác định vị trí của nhãn cần cải thiện. `OTHER_INDICES` được dùng khi lấy xác suất của `food`, `price`, `service` và `ambiance`.

### 6.3. Tạo embedding và xác suất baseline

Notebook tạo embedding một lần cho train, validation và test:

```python
X_train = encode(train_idx)
X_val = encode(val_idx)
X_test = encode(test_idx)
```

Sau đó classifier retrained hiện tại tạo xác suất:

```python
base_train_probabilities = base_classifier.predict_proba(X_train)
base_val_probabilities = base_classifier.predict_proba(X_val)
base_test_probabilities = base_classifier.predict_proba(X_test)
```

Các xác suất này được dùng cho:

- Baseline threshold.
- Phát hiện hard negative.
- Context feature trong stacked classifier.
- So sánh pipeline đầy đủ.

### 6.4. Hàm tối ưu threshold

Hàm `tune_binary_threshold()` thử:

```python
np.arange(0.05, 0.951, 0.005)
```

Với mỗi threshold, hàm tính:

- F1.
- Precision.
- Recall.

Tiêu chí chính là F1. Nếu hai threshold có F1 gần bằng nhau, notebook ưu tiên threshold có precision cao hơn:

```python
if (
    candidate["f1"] > best["f1"]
    or (
        np.isclose(candidate["f1"], best["f1"])
        and candidate["precision"] > best["precision"]
    )
):
```

Quy tắc tie-break này phù hợp với lỗi hiện tại vì false positive đang chiếm ưu thế.

### 6.5. Baseline threshold

Chiến lược đơn giản nhất không huấn luyện model mới:

```python
baseline_threshold = tune_binary_threshold(
    y_val_misc,
    base_val_probabilities[:, MISC_INDEX],
)
```

Mục tiêu là xác định mức cải thiện chỉ nhờ chọn threshold tốt hơn trên validation.

Nếu chiến lược này thắng, deployment chỉ cần cập nhật threshold trong metadata; không cần thêm classifier.

### 6.6. Out-of-fold probability

Hai chiến lược hard-negative và stacked context cần xác suất trên train set.

Không nên dùng xác suất do model đã fit trên toàn bộ train tạo ra vì chúng là in-sample predictions và thường quá lạc quan.

Notebook sử dụng:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)
```

Mỗi fold:

1. Fit OVR classifier trên bốn phần.
2. Dự đoán phần còn lại.
3. Ghi kết quả vào đúng vị trí trong `oof_probabilities`.

```python
oof_probabilities[holdout_indices] = (
    fold_classifier.predict_proba(
        X_train[holdout_indices]
    )
)
```

Mỗi review train vì vậy được dự đoán bởi một model chưa nhìn thấy review đó. Cách này hạn chế leakage khi xây meta-feature.

### 6.7. Chiến lược 1 — Dedicated weighted classifier

Thay vì dùng estimator `miscellaneous` chung với OVR, notebook huấn luyện một Logistic Regression nhị phân riêng:

```python
model = LogisticRegression(
    C=params["C"],
    class_weight={
        0: 1.0,
        1: params["positive_weight"],
    },
    solver=params["solver"],
    max_iter=3000,
    random_state=SEED,
)
```

Grid:

```python
DEDICATED_GRID = {
    "C": [
        0.01, 0.05, 0.1, 0.5,
        1.0, 2.0, 5.0, 10.0,
    ],
    "positive_weight": [
        1.0, 1.5, 2.0, 3.0,
        4.0, 5.0, 7.0,
    ],
    "solver": ["liblinear", "lbfgs"],
}
```

`positive_weight` điều khiển mức phạt khi bỏ sót mẫu `miscellaneous`.

- Weight quá thấp có thể làm recall giảm.
- Weight quá cao có thể tạo thêm false positive.

Mỗi cấu hình cũng được tối ưu threshold riêng trên validation.

### 6.8. Chiến lược 2 — Hard-negative reweighting

Hard negative là review có nhãn thật bằng `0` nhưng baseline lại gán xác suất `miscellaneous` cao.

```python
hard_negative_mask = np.logical_and(
    y_train_misc == 0,
    oof_probabilities[:, MISC_INDEX]
        >= hard_negative_cutoff,
)
```

Các mẫu này thường giống `miscellaneous` về ngữ nghĩa nhưng thực tế thuộc các lớp cụ thể như:

- Food/menu.
- Cleanliness/ambiance.
- Staff/service.
- Price/value.

Notebook tăng `sample_weight` của hard negative:

```python
sample_weight = np.ones(
    len(y_train_misc),
    dtype=np.float32,
)

sample_weight[hard_negative_mask] = (
    params["hard_negative_weight"]
)
```

Sau đó:

```python
model.fit(
    X_train,
    y_train_misc,
    sample_weight=sample_weight,
)
```

Grid thử nhiều:

- `C`.
- Positive class weight.
- Hard-negative weight.
- Solver.

Mục tiêu chính là giảm false positive mà không làm recall giảm quá mạnh.

### 6.9. Chiến lược 3 — Stacked context

Trong dataset hiện tại, `miscellaneous` luôn xuất hiện cùng ít nhất một aspect khác.

Do đó notebook bổ sung xác suất của bốn aspect còn lại vào SBERT embedding:

```python
def stacked_features(
    embeddings,
    probabilities,
    context_weight,
):
    return np.hstack([
        embeddings,
        probabilities[:, OTHER_INDICES]
            * context_weight,
    ])
```

Vector mới có dạng:

```text
[SBERT embedding,
 P(food),
 P(price),
 P(service),
 P(ambiance)]
```

Train sử dụng `oof_probabilities` để tránh leakage:

```python
train_features = stacked_features(
    X_train,
    oof_probabilities,
    context_weight,
)
```

Validation và test sử dụng xác suất từ classifier retrained đã lưu:

```python
val_features = stacked_features(
    X_val,
    base_val_probabilities,
    context_weight,
)
```

`context_weight` điều chỉnh mức ảnh hưởng của bốn xác suất phụ so với embedding.

Nếu stacked context thắng, deployment phải tải:

1. Retrained SBERT encoder.
2. OVR classifier gốc.
3. Improved miscellaneous classifier.

Ứng dụng phải lấy bốn xác suất aspect từ OVR classifier trước khi gọi improved classifier.

### 6.10. Multiprocessing grid search

Các chiến lược được chạy bằng Joblib:

```python
with parallel_config(
    backend="loky",
    n_jobs=min(4, os.cpu_count() or 1),
    inner_max_num_threads=1,
):
```

`loky` chạy các cấu hình trong process riêng. Giới hạn bốn process giúp tránh sử dụng quá nhiều RAM.

`inner_max_num_threads=1` tránh việc mỗi process tạo thêm nhiều native threads.

Notebook có grid tương đối lớn. Nếu máy thiếu RAM hoặc chạy quá lâu, nên thu hẹp grid theo hai giai đoạn:

1. Chạy grid thô.
2. Giữ vùng tham số tốt nhất và chạy grid nhỏ hơn.

### 6.11. Chọn chiến lược bằng validation

Bốn candidate:

```python
candidate_rows = [
    baseline_row,
    dedicated_best,
    hard_negative_best,
    stacked_best,
]
```

Kết quả được sắp xếp:

```python
validation_results.sort_values(
    ["f1", "precision"],
    ascending=False,
)
```

Chiến lược thắng:

```python
winning_strategy = (
    validation_results.iloc[0]["strategy"]
)
```

Kết quả validation được lưu tại:

```text
outputs/SBERT/miscellaneous_strategy_validation.csv
```

Không được chọn chiến lược dựa trên test F1.

### 6.12. So sánh trên test

Sau khi chiến lược thắng đã được cố định, notebook tính:

- F1.
- Precision.
- Recall.
- Average precision.
- Số dự đoán dương.

Kết quả:

```text
outputs/SBERT/miscellaneous_strategy_test.csv
```

Notebook vẫn ghi kết quả test của tất cả chiến lược để phân tích nghiên cứu. Tuy nhiên, không nên dùng bảng này để tiếp tục chọn tham số vì điều đó sẽ làm test set trở thành validation set.

### 6.13. Ảnh hưởng lên pipeline năm aspect

Notebook giữ nguyên dự đoán của:

- `food`
- `price`
- `service`
- `ambiance`

Chỉ cột `miscellaneous` được thay bằng dự đoán từ chiến lược thắng:

```python
improved_full_predictions[:, MISC_INDEX] = (
    winning_probabilities
        >= winning_candidate["threshold"]
).astype(np.int8)
```

Sau đó so sánh:

- Macro F1 toàn pipeline.
- Micro F1 toàn pipeline.
- Miscellaneous F1.

Kết quả:

```text
outputs/SBERT/miscellaneous_full_pipeline_comparison.csv
```

### 6.14. Precision-recall curve

Vì `miscellaneous` là lớp hiếm, precision-recall curve có ý nghĩa hơn accuracy.

Notebook vẽ đường cong cho từng chiến lược và tính:

```python
average_precision_score(
    y_test_misc,
    probabilities,
)
```

Một chiến lược tốt nên:

- Có average precision cao.
- Giữ precision tốt khi recall tăng.
- Không chỉ đạt F1 cao tại một threshold quá nhạy.

### 6.15. Phân tích lỗi còn lại

Các dự đoán sai của chiến lược thắng được lưu tại:

```text
outputs/SBERT/miscellaneous_remaining_errors.csv
```

Mỗi dòng chứa:

- Dataset index.
- Review.
- Nhãn thật.
- Nhãn dự đoán.
- Xác suất.
- Loại lỗi: false positive hoặc false negative.
- Các aspect thật của review.

File này nên được dùng để:

- Kiểm tra nhãn sai.
- Xác định subtopic thiếu dữ liệu.
- Tìm các hard negative mới.
- Quyết định có nên tách `miscellaneous` thành nhiều lớp cụ thể hay không.

### 6.16. Lưu improved component

Mô hình thắng được lưu riêng tại:

```text
models/sbert/classifiers/retrained_miscellaneous_improved/
├── model.joblib
└── metadata.json
```

Classifier gốc không bị ghi đè.

Metadata chứa:

- Tên chiến lược.
- Encoder path.
- Base classifier path.
- Miscellaneous index.
- Threshold.
- Tham số của chiến lược.
- Validation F1, precision và recall.

Nếu baseline threshold thắng, notebook chỉ cần lưu metadata vì không có classifier mới.

---

## 7. Tóm tắt luồng artifact

```text
data/Restaurant_ABSA_processed.csv
        │
        ▼
01-sbert_retraining.ipynb
        ├── outputs/SBERT/data_split.npz
        ├── models/sbert/unretrained_encoder/
        ├── models/sbert/retrained_encoder/
        └── models/sbert/encoder_training_metadata.json
        │
        ▼
02-sbert_ovr_logistic_regression.ipynb
        ├── models/sbert/classifiers/unretrained/
        ├── models/sbert/classifiers/retrained/
        ├── outputs/SBERT/logistic_grid_results.csv
        └── outputs/SBERT/classifier_validation_summary.csv
        │
        ▼
03-sbert_evaluate.ipynb
        ├── outputs/SBERT/encoder_comparison.csv
        ├── outputs/SBERT/*_classification_report.csv
        ├── outputs/SBERT/retrained_best_cases.csv
        ├── outputs/SBERT/retrained_worst_cases.csv
        ├── outputs/SBERT/retrained_sample_predictions.csv
        └── outputs/SBERT/retrained_aspect_error_analysis.csv
        │
        ▼
04-improve_mischellaneous.ipynb
        ├── outputs/SBERT/miscellaneous_strategy_validation.csv
        ├── outputs/SBERT/miscellaneous_strategy_test.csv
        ├── outputs/SBERT/miscellaneous_full_pipeline_comparison.csv
        ├── outputs/SBERT/miscellaneous_remaining_errors.csv
        └── models/sbert/classifiers/retrained_miscellaneous_improved/
```

## 8. Cách chạy lại pipeline

### Bước 1: Cài dependencies

```powershell
pip install -r requirements.txt
```

Các package chính:

- `sentence-transformers`
- `transformers[torch]`
- `torch`
- `scikit-learn`
- `joblib`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`

### Bước 2: Chạy notebook 01

Notebook này có chi phí cao nhất vì fine-tune SBERT.

Sau khi chạy, kiểm tra:

```text
models/sbert/unretrained_encoder/
models/sbert/retrained_encoder/
outputs/SBERT/data_split.npz
```

### Bước 3: Chạy notebook 02

Notebook tạo embedding và chạy 32 cấu hình Logistic Regression cho từng encoder.

Sau khi chạy, kiểm tra:

```text
models/sbert/classifiers/unretrained/model.joblib
models/sbert/classifiers/retrained/model.joblib
```

### Bước 4: Chạy notebook 03

Notebook tạo bảng so sánh cuối cùng và các file phân tích lỗi.

Không sử dụng kết quả test để quay lại chọn tham số nếu muốn giữ test set là đánh giá độc lập. Nếu cần thử cấu hình mới, nên chọn bằng validation hoặc tạo một experiment split mới.

### Bước 5: Chạy notebook 04

Notebook thử các chiến lược riêng cho `miscellaneous` và chọn chiến lược bằng validation F1.

Sau khi chạy, kiểm tra:

```text
outputs/SBERT/miscellaneous_strategy_validation.csv
outputs/SBERT/miscellaneous_strategy_test.csv
outputs/SBERT/miscellaneous_full_pipeline_comparison.csv
outputs/SBERT/miscellaneous_remaining_errors.csv
models/sbert/classifiers/retrained_miscellaneous_improved/
```

Grid của notebook 04 lớn hơn notebook 02. Nên bắt đầu với `n_jobs=2` nếu máy có ít RAM và chỉ tăng lên `4` sau khi kiểm tra mức sử dụng bộ nhớ.

## 9. Hạn chế hiện tại

### 9.1. Chưa dùng multilabel-stratified split

Pipeline đang dùng `train_test_split()` thông thường. Với lớp hiếm như `miscellaneous`, số mẫu dương có thể phân bố không đều.

Có thể cải thiện bằng iterative multilabel stratification.

### 9.2. Dữ liệu nhỏ

Dataset chỉ có khoảng 800 review. Fine-tune mô hình lớn như MPNet có nguy cơ overfit.

Nên:

- Thử nhiều seed.
- So sánh kết quả trung bình và độ lệch chuẩn.
- Dùng early stopping hoặc evaluator trong quá trình fine-tune.
- Giảm epoch nếu validation giảm.

### 9.3. Mục tiêu fine-tune gián tiếp

SBERT được fine-tune để học similarity giữa các bộ aspect, không trực tiếp tối ưu classification loss.

Đây là một dạng metric learning. Có thể thử thêm:

- Multiple Negatives Ranking Loss.
- Contrastive Loss.
- Triplet Loss.
- Fine-tune Transformer trực tiếp bằng multi-label BCE loss.

### 9.4. `miscellaneous` khó học

`miscellaneous` có ít mẫu và chứa nhiều chủ đề không đồng nhất, ví dụ:

- Vị trí.
- Chính sách nhà hàng.
- Marketing.
- Công nghệ.
- Trải nghiệm chung.

Giải pháp tốt hơn chỉ điều chỉnh ngưỡng là:

- Sửa nhãn sai.
- Bổ sung dữ liệu.
- Tách `miscellaneous` thành các aspect cụ thể.
- Xây classifier riêng cho lớp này.

### 9.5. Classifier cuối chỉ fit trên train

Classifier được lưu sau khi fit trên train và tối ưu bằng validation. Validation chưa được gộp vào quá trình fit cuối.

Nếu đã hoàn tất toàn bộ lựa chọn tham số và chấp nhận đóng băng cấu hình, có thể:

1. Giữ thresholds đã chọn trên validation.
2. Gộp train và validation.
3. Fit lại classifier cuối trên tập gộp.
4. Đánh giá duy nhất một lần trên test.

Việc này cần được thực hiện cẩn thận để không tối ưu lại dựa trên test.

## 10. Kết luận

Pipeline hiện tại cho thấy fine-tune SBERT bằng similarity của bộ nhãn giúp cải thiện rõ rệt:

- Macro F1 tăng từ `0.7837` lên `0.8296` trên test.
- Micro F1 tăng từ `0.8785` lên `0.9231`.
- Hamming loss giảm từ `0.0975` xuống `0.0613`.
- F1 của `food`, `price`, `service` và `ambiance` đều tăng.

Tuy nhiên, `miscellaneous` vẫn là điểm yếu chính. Các bước tiếp theo nên tập trung vào chất lượng nhãn, multilabel stratification và bổ sung dữ liệu cho lớp này thay vì chỉ tiếp tục mở rộng grid Logistic Regression.

Notebook 04 cung cấp một bước thử nghiệm có kiểm soát trước khi thay đổi annotation schema. Dedicated classifier, hard-negative weighting và stacked context có thể cải thiện precision/F1 trong ngắn hạn, nhưng không thay thế được việc chuẩn hóa định nghĩa và bổ sung dữ liệu cho `miscellaneous`.
