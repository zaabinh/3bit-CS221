# Tóm tắt EDA cho `Restaurant_ABSA_processed.csv`

Tài liệu này tổng hợp lại kết quả mới nhất từ `notebooks/03-EDA.ipynb` và các file trong `outputs/EDA/`. Dữ liệu đã được làm sạch/cập nhật so với bản EDA trước: hiện còn **796 dòng**, không còn duplicate theo `review_en`, và file correction đã được căn lại đúng index với dữ liệu hiện tại.

## 1. Tổng quan dữ liệu

- Dataset hiện có **796 dòng** và **7 cột**: `review_en`, `review_cleaned`, `food`, `price`, `service`, `ambiance`, `miscellaneous`.
- Không có giá trị thiếu ở tất cả các cột.
- Tất cả nhãn aspect đều là nhị phân hợp lệ `0/1`.
- Không còn duplicate theo `review_en`.
- Không còn duplicate toàn dòng.
- Số giá trị unique:
  - `review_en`: **796 / 796**
  - `review_cleaned`: **789 / 796**

Nhận xét:

- Dữ liệu hiện đã sạch hơn bản trước vì duplicate review đã được xử lý.
- `review_cleaned` vẫn có ít unique hơn `review_en`, nghĩa là một số câu khác nhau bị gom về cùng chuỗi sau preprocessing. Điều này bình thường với dữ liệu đã loại dấu câu/stopword, nhưng có thể làm mất một phần sắc thái ngữ nghĩa.
- Vấn đề chính còn lại không nằm ở missing value hay duplicate, mà nằm ở **mất cân bằng nhãn**, **định nghĩa nhãn `miscellaneous`**, và **khả năng nhiễu/thiếu nhãn theo heuristic**.

## 2. Phân bố nhãn aspect

| Aspect | Số mẫu positive | Tỷ lệ |
|---|---:|---:|
| `food` | 656 | 82.41% |
| `price` | 404 | 50.75% |
| `service` | 227 | 28.52% |
| `ambiance` | 201 | 25.25% |
| `miscellaneous` | 49 | 6.16% |

Nhận xét:

- `food` chiếm ưu thế rất mạnh, xuất hiện trong hơn 82% dữ liệu.
- `price` xuất hiện ở khoảng một nửa dataset.
- `service` và `ambiance` ở mức trung bình thấp.
- `miscellaneous` cực hiếm, chỉ có **49 mẫu positive**, là nhãn khó nhất cho mô hình.

Đề xuất:

- Không dùng accuracy làm metric chính vì nhãn mất cân bằng mạnh.
- Luôn báo cáo `macro-F1`, `micro-F1`, `samples-F1`, và F1 riêng từng aspect.
- Với Logistic Regression / One-vs-Rest, tiếp tục dùng `class_weight="balanced"` và threshold riêng cho từng nhãn.
- Với `miscellaneous`, cần xử lý riêng: tune threshold riêng, binary classifier riêng, hoặc bổ sung dữ liệu.

## 3. Đặc điểm multi-label

Phân bố số aspect positive trên mỗi review:

| Số aspect positive | Số review |
|---:|---:|
| 1 | 59 |
| 2 | 733 |
| 3 | 4 |

Các tổ hợp aspect phổ biến nhất:

| Tổ hợp aspect | Số review |
|---|---:|
| `food, price` | 332 |
| `food, service` | 155 |
| `food, ambiance` | 105 |
| `service, ambiance` | 41 |
| `food` | 36 |
| `price, ambiance` | 33 |
| `food, miscellaneous` | 24 |
| `price, service` | 18 |
| `price, miscellaneous` | 13 |

Nhận xét:

- Dataset gần như được thiết kế theo dạng **2 aspect/review**: 733/796 dòng có đúng 2 nhãn.
- Chỉ có 59 dòng một nhãn và 4 dòng ba nhãn.
- Điều này có thể làm model học bias rằng mỗi review thường nên có 2 aspect. Nếu review thật khi deploy có phân bố khác, model có thể dự đoán thừa hoặc thiếu nhãn.

Đề xuất:

- Giữ bài toán ở dạng multi-label, không chuyển thành single-label.
- Khi deploy, cần kiểm tra threshold trên dữ liệu thật vì phân bố số aspect/review ngoài thực tế có thể khác dataset train.
- Nên phân tích thêm số nhãn dự đoán trung bình trên validation/test để phát hiện bias “luôn dự đoán 2 nhãn”.

## 4. Co-occurrence và correlation giữa các aspect

Co-occurrence đáng chú ý:

- `food` + `price`: **336**
- `food` + `service`: **155**
- `food` + `ambiance`: **109**
- `service` + `ambiance`: **41**
- `price` + `ambiance`: **37**
- `food` + `miscellaneous`: **24**
- `price` + `miscellaneous`: **13**

Correlation đáng chú ý:

- `price` và `service`: **-0.54**
- `food` và `ambiance`: **-0.43**
- `price` và `ambiance`: **-0.38**
- `food` và `service`: **-0.23**
- `food` và `price`: gần như không tương quan, khoảng **0.02**

Nhận xét:

- `food` đi kèm `price` nhiều nhất theo count, nhưng correlation thấp vì `food` xuất hiện quá phổ biến.
- `price` và `service` có tương quan âm mạnh, nghĩa là trong dataset này hai nhãn này ít đi cùng nhau.
- `miscellaneous` có tương quan thấp/âm nhẹ với tất cả nhãn khác, phản ánh việc lớp này vừa hiếm vừa không có pattern đồng xuất hiện rõ.

Đề xuất:

- Không áp rule cứng dựa trên correlation, ví dụ không nên nói “có `price` thì không có `service`”, vì vẫn có 18 dòng `price, service`.
- Có thể dùng xác suất của các aspect khác làm feature phụ cho chiến lược stacked model, nhưng phải chọn bằng validation.
- Với `miscellaneous`, không nên kỳ vọng co-occurrence giúp nhiều; cần cải thiện định nghĩa nhãn và dữ liệu positive.

## 5. Độ dài văn bản

Thống kê độ dài mới nhất:

| Trường | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| `review_word_count` | 14.21 | 14 | 4 | 31 |
| `cleaned_word_count` | 6.47 | 6 | 2 | 14 |
| `review_char_count` | 72.12 | 70 | 24 | 164 |
| `cleaned_char_count` | 41.64 | 40 | 15 | 101 |

Theo aspect:

| Aspect | Mean words gốc | Median words gốc | Mean words cleaned |
|---|---:|---:|---:|
| `food` | 14.04 | 14 | 6.41 |
| `price` | 14.23 | 14 | 6.46 |
| `service` | 14.22 | 14 | 6.53 |
| `ambiance` | 14.64 | 14 | 6.66 |
| `miscellaneous` | 14.90 | 14 | 6.78 |

Nhận xét:

- Review ngắn và khá đồng đều, phù hợp với SBERT và Logistic Regression.
- `miscellaneous` có xu hướng dài hơn một chút, nhưng khác biệt nhỏ; độ dài không đủ để phân biệt lớp này.
- Sau preprocessing, số từ giảm hơn một nửa. Điều này hữu ích cho TF-IDF nhưng có rủi ro mất ngữ nghĩa.

Đề xuất:

- Với SBERT: dùng `review_en` gốc, không dùng `review_cleaned`.
- Với TF-IDF: có thể dùng `review_cleaned`, nhưng cần giữ phủ định nếu mục tiêu có sentiment hoặc ngữ nghĩa chi tiết.
- Với cả train và deployment, preprocessing phải thống nhất tuyệt đối.

## 6. Từ vựng và tín hiệu lexical

Top terms trong `review_cleaned`:

- `food`: 342
- `good`: 306
- `price`: 298
- `taste`: 150
- `restaurant`: 150
- `service`: 122
- `high`: 115
- `quality`: 102
- `also`: 100
- `bad`: 80
- `atmosphere`: 57
- `environment`: 53

Nhận xét:

- Các từ khóa aspect rất rõ: `food`, `price`, `service`, `atmosphere`, `environment`.
- Đây là lý do TF-IDF + Logistic Regression có thể là baseline mạnh.
- Tuy nhiên các từ như `good`, `bad`, `high`, `low`, `expensive` dễ bị hiểu sai nếu preprocessing xóa mất phủ định.

Ví dụ rủi ro:

- `not good` có thể thành `good`.
- `not expensive` có thể thành `expensive`.
- `not clean` có thể thành `clean`.

Đề xuất:

- Không dùng stopword removal mặc định nếu nó xóa `not`, `no`, `never`, `without`.
- Nếu dùng TF-IDF, nên thử `ngram_range=(1, 2)` hoặc `(1, 3)` để giữ cụm phủ định và cụm aspect như `food price`, `good service`, `bad environment`.
- Nếu dùng SBERT, câu gốc đã đủ ngắn; không cần cleaning mạnh.

## 7. Ký tự đặc biệt / uncommon characters

File `outputs/EDA/special_character_rows.csv` phát hiện ký tự đặc biệt:

- `U+200B` — zero-width space
- Xuất hiện trong **3 dòng** ở `review_en`
- Các dòng bị ảnh hưởng: `150`, `403`, `490`

Đề xuất:

- Loại bỏ `\u200b` trong preprocessing và deployment normalization:

```python
text = text.replace("\u200b", "")
```

- Sau khi normalize, nên lưu lại dữ liệu để tránh ký tự vô hình làm lệch matching, hashing, hoặc lookup theo text.

## 8. Nghi vấn thiếu nhãn theo heuristic

File `outputs/EDA/suspicious_missing_aspect_labels.csv` phát hiện **113 dòng nghi vấn thiếu nhãn** bằng keyword heuristic:

| Aspect nghi thiếu | Số dòng |
|---|---:|
| `ambiance` | 64 |
| `food` | 47 |
| `service` | 2 |

Nhận xét:

- Đây là heuristic nên có false positive. Ví dụ một câu có cụm `food price` có thể chỉ nói về giá, không nhất thiết đánh giá món ăn.
- Tuy vậy, 113 dòng là tỷ lệ đáng kể trên dataset 796 dòng, nên vẫn cần review thủ công.
- File correction `Data/temp/Restaurant_ABSA_aspect_corrections.csv` hiện có **103 dòng** và đã được căn đúng `source_row` với dữ liệu hiện tại. Sau khi đối chiếu, các label correction hiện không khác dữ liệu processed nữa, tức là processed CSV nhiều khả năng đã được cập nhật theo correction.

Đề xuất:

- Tiếp tục review `outputs/EDA/suspicious_missing_aspect_labels.csv`, nhưng không tự động sửa toàn bộ.
- Ưu tiên review các dòng nghi thiếu `ambiance` vì nhiều nhất.
- Sau mỗi vòng sửa label, chạy lại `03-EDA.ipynb` để xác nhận:
  - số dòng
  - duplicate
  - phân bố nhãn
  - suspicious rows
- Nếu giữ correction file, nên dùng nó như audit trail: `source_row`, label mới, và `explain`.

## 9. Vấn đề lớn nhất: `miscellaneous`

`miscellaneous` hiện có **49 positive samples**, tương đương **6.16%** dataset.

Vấn đề:

- Số mẫu positive quá ít so với các nhãn khác.
- Lớp này có khả năng gom nhiều ý nghĩa khác nhau: location, overall experience, occasion, marketing, reputation, customer preference, hoặc các nhận xét không thuộc 4 aspect chính.
- Vì lớp này không có từ khóa rõ như `food`, `price`, `service`, `ambiance`, model dễ bị recall thấp hoặc precision thấp tùy threshold.

Đề xuất cải thiện:

- Viết guideline rõ cho `miscellaneous`: khi nào gán, khi nào không gán.
- Nếu có thể, bổ sung dữ liệu positive cho lớp này. Mục tiêu thực tế nên là ít nhất **100-150 positive samples**.
- Nếu không có thêm dữ liệu, giữ chiến lược riêng trong `notebooks/SBERT/04-improve_mischellaneous.ipynb`:
  - threshold riêng cho `miscellaneous`
  - classifier nhị phân riêng
  - hard-negative reweighting
  - stacked context từ xác suất các aspect khác
- Khi đánh giá, cần xem riêng false positive/false negative của `miscellaneous`, không chỉ xem macro-F1 tổng.

## 10. Đề xuất thay đổi cho solution

### Data

- Giữ bản processed hiện tại vì đã sạch hơn: 796 dòng, không missing, không duplicate.
- Loại bỏ `U+200B` khỏi `review_en` nếu chưa được normalize trong source.
- Duy trì `Data/temp/Restaurant_ABSA_aspect_corrections.csv` như audit file cho các sửa nhãn.
- Review thêm 113 dòng suspicious còn lại trước khi train bản cuối.

### Preprocessing

- Với SBERT:
  - dùng `review_en`
  - chỉ normalize nhẹ: lowercase nếu đã thống nhất, strip whitespace, xóa `\u200b`
  - không dùng stopword removal/lemmatization mạnh
- Với TF-IDF:
  - có thể dùng `review_cleaned`
  - giữ phủ định
  - dùng n-gram `(1, 2)` hoặc `(1, 3)`

### Modeling

- Tiếp tục dùng multi-label One-vs-Rest.
- Tune threshold riêng từng aspect trên validation set.
- Dùng `class_weight="balanced"` cho Logistic Regression.
- Với `miscellaneous`, dùng pipeline riêng hoặc model phụ vì dữ liệu quá ít.
- Không dùng correlation để loại trừ nhãn sau dự đoán; chỉ dùng như tín hiệu phân tích.

### Evaluation

- Giữ split cố định để so sánh công bằng giữa các phiên bản.
- Báo cáo:
  - `macro-F1`
  - `micro-F1`
  - `samples-F1`
  - per-aspect precision/recall/F1
  - confusion matrix từng aspect
  - exact-match ratio
  - right/wrong samples
- Với `miscellaneous`, bắt buộc phân tích riêng vì nó là bottleneck.

## 11. Kết luận

Dữ liệu hiện tại đã tốt hơn bản trước: không còn duplicate, không missing value, nhãn là `0/1` hợp lệ, và correction file đã khớp index với processed CSV. Tuy nhiên, solution vẫn bị giới hạn bởi:

1. Mất cân bằng nhãn rất mạnh, đặc biệt `miscellaneous`.
2. 113 dòng suspicious theo heuristic vẫn cần review thủ công.
3. `review_cleaned` làm mất nhiều token và có thể mất phủ định, nên không phù hợp làm input chính cho SBERT.
4. Một vài dòng còn zero-width space trong `review_en`, cần normalize trước deployment.

Thứ tự ưu tiên nên làm:

1. Normalize `\u200b` trong dữ liệu và deployment preprocessing.
2. Review 113 dòng suspicious, cập nhật correction file nếu cần.
3. Rà lại guideline cho `miscellaneous`.
4. Train lại model với split cố định.
5. Đánh giá bằng per-class F1 và phân tích lỗi riêng cho `miscellaneous`.

