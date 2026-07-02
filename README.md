# Aspect Classification for Restaurant Reviews using Sentence Transformers and TF-IDF

## Giới thiệu

Đây là đồ án cuối kỳ môn **Xử lý ngôn ngữ tự nhiên** tại Trường Đại học Công nghệ Thông tin - Đại học Quốc gia Thành phố Hồ Chí Minh.

Mục tiêu của đề tài là xây dựng và đánh giá các phương pháp **Aspect Classification** trong bài toán **Aspect-Based Sentiment Analysis (ABSA)** đối với các đánh giá nhà hàng. Đề tài tập trung vào việc xác định những khía cạnh được đề cập trong mỗi câu đánh giá, bao gồm:

- Food
- Price
- Service
- Ambiance
- Miscellaneous

Đề tài tiến hành so sánh hiệu năng của ba phương pháp:

- TF-IDF + One-vs-Rest Logistic Regression
- Sentence Transformer (all-mpnet-base-v2) + One-vs-Rest Logistic Regression
- Fine-tuned Sentence Transformer (all-mpnet-base-v2) + One-vs-Rest Logistic Regression

---

## Tập dữ liệu

Dataset sử dụng trong đề tài:

**Restaurant Aspect-Based Sentiment Analysis Dataset**

Nguồn:

https://data.mendeley.com/datasets/998m4jy3m9/3

Đặc điểm:

- 796 câu đánh giá nhà hàng.
- Dữ liệu gốc bằng tiếng Bangla, được dịch sang tiếng Anh để xử lý.
- Bài toán phân loại đa nhãn (Multi-label Classification).
- 5 nhãn khía cạnh:
  - Food
  - Price
  - Service
  - Ambiance
  - Miscellaneous

---

## Các phương pháp thực nghiệm

### 1. TF-IDF + Logistic Regression

Phương pháp học máy truyền thống:

- Tiền xử lý văn bản (Stopwords Removal, Lemmatization).
- Trích xuất đặc trưng bằng TF-IDF.
- Phân loại đa nhãn bằng One-vs-Rest Logistic Regression.
- Tối ưu ngưỡng dự đoán cho từng nhãn.

### 2. Sentence Transformer

Phương pháp biểu diễn ngữ nghĩa sử dụng mô hình tiền huấn luyện:

- Backbone: all-mpnet-base-v2.
- Sentence Embedding kích thước 768.
- Bi-Encoder Architecture.
- Phân loại bằng One-vs-Rest Logistic Regression.
- Tối ưu ngưỡng dự đoán.

### 3. Fine-tuned Sentence Transformer

Phương pháp chính của đề tài:

- Fine-tune mô hình all-mpnet-base-v2 trên dữ liệu Aspect Classification.
- Contrastive Learning giúp tối ưu không gian embedding.
- Sentence Embedding đặc trưng cho miền dữ liệu nhà hàng.
- One-vs-Rest Logistic Regression.
- Threshold Tuning cho từng khía cạnh.

---

## Kết quả thực nghiệm

| Phương pháp | Macro F1 | Micro F1 | Hamming Loss |
| ------------ | -------- | -------- | ------------ |
| TF-IDF + Logistic Regression | 0.88 | 0.95 | 0.03 |
| Sentence Transformer | 0.78 | 0.86 | 0.11 |
| Fine-tuned Sentence Transformer | 0.74 | 0.90 | 0.07 |

### Nhận xét

- TF-IDF có ưu điểm đơn giản, dễ triển khai nhưng chưa khai thác được quan hệ ngữ nghĩa giữa các câu.
- Sentence Transformer cải thiện khả năng biểu diễn ngữ nghĩa và nâng cao hiệu quả phân loại.
- Fine-tune Sentence Transformer giúp embedding phù hợp hơn với miền dữ liệu đánh giá nhà hàng, từ đó đạt hiệu quả phân loại cao nhất.
- Threshold Tuning giúp cải thiện đáng kể hiệu năng đối với các nhãn mất cân bằng, đặc biệt là **Miscellaneous**.

---

## Cấu trúc thư mục

```text
Aspect-Classification/
├── data/
│   ├── raw/
│   ├── processed/
│   └── translated/
├── notebooks/
│   ├── preprocessing.ipynb
│   ├── eda.ipynb
│   ├── tfidf/
│   │   ├── tfidf_train.ipynb
│   │   └── tfidf_evaluate.ipynb
│   ├── sentence_transformer/
│   │   ├── sbert_train.ipynb
│   │   ├── sbert_finetune.ipynb
│   │   └── sbert_evaluate.ipynb
│   └── comparison.ipynb
├── models/
│   ├── tfidf/
│   ├── sentence_transformer/
│   └── fine_tuned_sentence_transformer/
├── outputs/
│   ├── figures/
│   ├── reports/
│   └── predictions/
├── requirements.txt
└── README.md
```

---

## Cài đặt môi trường

Tạo môi trường Python:

```bash
python -m venv venv
```

Kích hoạt môi trường:

Windows:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

Cài đặt thư viện:

```bash
pip install -r requirements.txt
```

---

## Thành viên thực hiện

- Đoàn Hữu Gia Bình - 24520192
- Nguyễn Quang Đông - 24520310
- Nguyễn Thị Ái Trâm - 24521805 

Giảng viên hướng dẫn:

- TS. Nguyễn Trọng Chỉnh

---

## Kết luận

Đề tài đã xây dựng thành công hệ thống phân loại khía cạnh cho các đánh giá nhà hàng dựa trên các kỹ thuật xử lý ngôn ngữ tự nhiên hiện đại. Kết quả thực nghiệm cho thấy việc sử dụng **Sentence Transformer kết hợp One-vs-Rest Logistic Regression** cho hiệu quả vượt trội so với phương pháp biểu diễn văn bản truyền thống TF-IDF. Bên cạnh đó, quá trình **fine-tune Sentence Transformer** và **tối ưu ngưỡng dự đoán** tiếp tục cải thiện chất lượng embedding và nâng cao hiệu năng trên bài toán phân loại đa nhãn, đặc biệt đối với các khía cạnh có số lượng mẫu ít.