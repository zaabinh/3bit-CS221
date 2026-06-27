# So sánh cuối cùng: MPNet unretrained, MPNet retrained và TF-IDF

Tài liệu này tóm tắt kết quả từ `notebooks/final_comparison.ipynb` và các output trong `outputs/final_comparison/`. Tất cả mô hình được đánh giá trên cùng test set cố định từ `outputs/SBERT/data_split.npz`.

## 1. Kết quả tổng quan

| Model | Macro-F1 | Micro-F1 | Samples-F1 | Exact Match | Hamming Loss | ms/review |
|---|---:|---:|---:|---:|---:|---:|
| MPNet retrained | **0.8873** | **0.9554** | **0.9535** | **0.8688** | **0.0350** | 55.27 |
| MPNet unretrained | 0.7819 | 0.8645 | 0.8727 | 0.6125 | 0.1125 | 61.65 |
| TF-IDF | 0.7736 | 0.9064 | 0.8992 | 0.7313 | 0.0713 | **0.0114** |

Nhận xét chính:

- **MPNet retrained là mô hình tốt nhất tổng thể**: cao nhất ở `macro_f1`, `micro_f1`, `samples_f1`, `exact_match_ratio`, và thấp nhất ở `hamming_loss`.
- **TF-IDF nhanh hơn rất nhiều**: khoảng `0.011 ms/review`, trong khi MPNet mất khoảng `55-62 ms/review`.
- **TF-IDF tốt hơn MPNet unretrained ở micro-F1, samples-F1, exact match và hamming loss**, nhưng kém ở macro-F1 do yếu ở `miscellaneous`.
- **MPNet unretrained không nên dùng làm model cuối**, vì retraining giúp cải thiện rất lớn.

## 2. F1 theo từng aspect

| Aspect | MPNet retrained | MPNet unretrained | TF-IDF |
|---|---:|---:|---:|
| `food` | **0.9675** | 0.9273 | 0.9368 |
| `price` | **1.0000** | 0.9581 | 0.9486 |
| `service` | **0.9398** | **0.9398** | 0.9383 |
| `ambiance` | **0.9459** | 0.7619 | 0.7941 |
| `miscellaneous` | **0.5833** | 0.3226 | 0.2500 |

Giải thích:

- `price` là lớp dễ nhất với MPNet retrained: đạt F1 `1.0`, không có false positive và false negative trên test set.
- `service` gần như ngang nhau giữa ba mô hình, cho thấy tín hiệu lexical của lớp này khá rõ.
- `ambiance` cải thiện mạnh sau retraining MPNet: từ `0.7619` lên `0.9459`.
- `miscellaneous` vẫn là lớp yếu nhất ở mọi mô hình, nhưng MPNet retrained tốt hơn rõ rệt.

## 3. Phân tích lỗi

| Model | Aspect | TP | FP | FN | TN |
|---|---|---:|---:|---:|---:|
| MPNet retrained | `food` | 134 | 9 | 0 | 17 |
| MPNet retrained | `price` | 85 | 0 | 0 | 75 |
| MPNet retrained | `service` | 39 | 2 | 3 | 116 |
| MPNet retrained | `ambiance` | 35 | 4 | 0 | 121 |
| MPNet retrained | `miscellaneous` | 7 | 3 | 7 | 143 |
| TF-IDF | `food` | 126 | 9 | 8 | 17 |
| TF-IDF | `price` | 83 | 7 | 2 | 68 |
| TF-IDF | `service` | 38 | 1 | 4 | 117 |
| TF-IDF | `ambiance` | 27 | 6 | 8 | 119 |
| TF-IDF | `miscellaneous` | 2 | 0 | 12 | 146 |

Nhận xét:

- MPNet retrained có recall rất tốt ở `food`, `price`, `ambiance`: không bỏ sót mẫu positive ở ba lớp này.
- TF-IDF bỏ sót nhiều hơn ở `food`, `ambiance`, và đặc biệt là `miscellaneous`.
- TF-IDF có precision `miscellaneous = 1.0` nhưng recall chỉ `0.1429`, nghĩa là mô hình rất ít khi dự đoán `miscellaneous`; khi dự đoán thì đúng, nhưng bỏ sót hầu hết mẫu thật.
- MPNet unretrained bị lỗi nghiêm trọng ở `miscellaneous`: 38 false positives, precision chỉ `0.2083`.

## 4. Exact-match overlap

| MPNet unretrained đúng | MPNet retrained đúng | TF-IDF đúng | Số mẫu |
|---|---|---|---:|
| True | True | True | 82 |
| False | True | True | 26 |
| False | True | False | 21 |
| True | True | False | 10 |
| False | False | False | 9 |
| False | False | True | 6 |
| True | False | True | 3 |
| True | False | False | 3 |

Diễn giải:

- Có **82 mẫu** cả ba mô hình đều dự đoán đúng toàn bộ tập nhãn.
- Có **31 mẫu** MPNet retrained đúng nhưng TF-IDF sai: `21 + 10`.
- Có **9 mẫu** TF-IDF đúng nhưng MPNet retrained sai: `6 + 3`.
- Điều này cho thấy MPNet retrained không chỉ tốt hơn trên metric tổng, mà còn sửa được nhiều case mà TF-IDF sai hơn chiều ngược lại.

## 5. Mẫu MPNet retrained đúng nhưng TF-IDF sai

Một số ví dụ từ `outputs/final_comparison/mpnet_retrained_correct_tfidf_wrong.csv`:

| source_row | Review | True | MPNet retrained | TF-IDF |
|---:|---|---|---|---|
| 750 | the food here is not very good and the workers are lazy | `food, service` | `food, service` | `food` |
| 625 | their kitchen is not very nice so people cannot enjoy the taste of the food as much | `food, ambiance` | `food, ambiance` | `food` |
| 285 | there is no extra decoration and the service is decent | `service, ambiance` | `service, ambiance` | `service` |
| 630 | the biryani rice was fluffy but the meat was tough | `food` | `food` | `food, price` |

Giải thích:

- TF-IDF thường bỏ sót aspect nếu từ khóa không đủ rõ hoặc bị nhiễu bởi cụm từ khác.
- Với câu về “kitchen”, “decoration”, “workers”, MPNet hiểu ngữ cảnh tốt hơn nên bắt được `ambiance` hoặc `service`.
- TF-IDF có thể dự đoán thừa `price` ở câu không nói về giá vì các pattern lexical trong train bị nhiễu hoặc từ ngữ liên quan món ăn/quantity dễ đi kèm price.

## 6. Mẫu TF-IDF đúng nhưng MPNet retrained sai

Một số ví dụ từ `outputs/final_comparison/tfidf_correct_mpnet_retrained_wrong.csv`:

| source_row | Review | True | MPNet retrained | TF-IDF |
|---:|---|---|---|---|
| 634 | schizlingter is a type of food that is served hot and fresh | `food, service` | `food` | `food, service` |
| 109 | the location of the restaurant is good and the price of the food is also low | `price, miscellaneous` | `price, ambiance, miscellaneous` | `price, miscellaneous` |
| 31 | as for the beef the test was very good but the name kalabuna was not right | `food` | `food, miscellaneous` | `food` |
| 344 | all the staff looked very professional but their service was not that good | `service` | `food, service` | `service` |

Giải thích:

- MPNet retrained đôi khi dự đoán thừa aspect do ngữ cảnh rộng, ví dụ thấy `restaurant`, `location`, `staff`, hoặc tên món lạ và suy ra thêm nhãn.
- TF-IDF có lợi thế ở một số câu có từ khóa aspect trực tiếp, ví dụ `served` liên quan `service`.
- Một số lỗi liên quan đến ranh giới nhãn: `location` được gán `miscellaneous`, nhưng MPNet có thể hiểu gần với `ambiance`.

## 7. Mẫu cả MPNet retrained và TF-IDF đều sai

Có **12 mẫu** mà cả MPNet retrained và TF-IDF đều sai exact-match. Một số ví dụ từ `outputs/final_comparison/sample_level_predictions.csv`:

| source_row | Review | True | MPNet retrained | TF-IDF |
|---:|---|---|---|---|
| 260 | a restaurant and location is also very good to eat a full stomach with less money | `price, miscellaneous` | `food, price, ambiance` | `food, price` |
| 599 | the pizza was delicious but the experience was horrible | `food, miscellaneous` | `food, service` | `food, service` |
| 335 | the reviews of the restaurant online were bad but i found the quality of the food to be good | `food, miscellaneous` | `food` | `food, ambiance` |
| 569 | they are giving free toys to children which is good for marketing but the price of food is high and the price of toys is being collected | `price, miscellaneous` | `food, price` | `food, price` |
| 593 | food of normal price is very scarce here and they are not treating all the customers well | `price, service` | `food, price` | `food, price` |
| 412 | it is hard to find eggs in eggchop and everything is rotten | `food` | `food, ambiance` | `food, price` |

Giải thích các lỗi chung:

- Nhiều case có `miscellaneous` nhưng cách diễn đạt không rõ ràng, ví dụ `experience`, `reviews online`, `marketing`, `location`, `favorite`, hoặc “nothing else”. Đây là nhóm ý nghĩa rộng, không có pattern lexical ổn định.
- Cụm `food price`, `price of food`, `less money`, `overpriced` làm cả hai mô hình dễ thêm hoặc giữ `food`, dù label thật đôi khi chỉ là `price`.
- Một số câu có ranh giới mờ giữa `miscellaneous` và `ambiance`, ví dụ `location` có thể bị MPNet hiểu là không gian/địa điểm nhà hàng.
- Một số câu service được diễn đạt gián tiếp như `treating customers well`, `waiters experience worse`; TF-IDF dễ miss nếu keyword không khớp đủ mạnh, còn MPNet đôi khi vẫn ưu tiên `food` vì câu có nhiều từ liên quan món ăn/giá.

Kết luận từ các case cả hai đều sai:

- Lỗi không chỉ do mô hình, mà còn do **định nghĩa label chưa đủ sắc nét**, đặc biệt `miscellaneous`.
- Cần review thủ công nhóm này trước khi train bản cuối vì đây là các mẫu có khả năng gây nhiễu label hoặc nằm sát ranh giới giữa các aspect.
- Nếu cần cải thiện thêm, ưu tiên viết guideline cho `miscellaneous`, `location`, `experience`, `marketing`, `online review`, và các câu chỉ nói về giá của món ăn.

## 8. Best/worst cases theo từng mô hình

Best/worst case được lấy từ các file:

- `outputs/final_comparison/mpnet_retrained_best_cases.csv`
- `outputs/final_comparison/mpnet_retrained_worst_cases.csv`
- `outputs/final_comparison/mpnet_unretrained_best_cases.csv`
- `outputs/final_comparison/mpnet_unretrained_worst_cases.csv`
- `outputs/final_comparison/tfidf_best_cases.csv`
- `outputs/final_comparison/tfidf_worst_cases.csv`

Tiêu chí:

- **Best cases**: dự đoán đúng exact-match và xác suất trung bình trên nhãn thật cao.
- **Worst cases**: sai exact-match, ưu tiên các mẫu có nhiều lỗi nhãn, xác suất cao cho nhãn sai, và xác suất thấp cho nhãn thật.

Tóm tắt case-level:

| Model | Exact Match | Mean label errors | Max label errors | Mean true probability |
|---|---:|---:|---:|---:|
| MPNet retrained | **139 / 160** | **0.1750** | 3 | **0.9471** |
| MPNet unretrained | 98 / 160 | 0.5625 | 3 | 0.8554 |
| TF-IDF | 117 / 160 | 0.3563 | 2 | 0.8047 |

### MPNet retrained

Best cases tiêu biểu:

| source_row | Review | True/Predicted | Mean true probability |
|---:|---|---|---:|
| 779 | the restaurant is very clean and the ambience is very nice | `ambiance` | 0.9999 |
| 628 | it is usually a little less salty than other dishes the taste is unmatched | `food` | 0.9999 |
| 30 | the biryani rice was fluffy and fluffy but the meat was tough | `food` | 0.9998 |

Worst cases tiêu biểu:

| source_row | Review | True | Predicted | Lỗi chính |
|---:|---|---|---|---|
| 260 | a restaurant and location is also very good to eat a full stomach with less money | `price, miscellaneous` | `food, price, ambiance` | thêm `food`, `ambiance`; miss `miscellaneous` |
| 424 | the food here tastes a little better but nothing else is quite customerfriendly | `food, miscellaneous` | `food, service` | thêm `service`; miss `miscellaneous` |
| 599 | the pizza was delicious but the experience was horrible | `food, miscellaneous` | `food, service` | thêm `service`; miss `miscellaneous` |

Giải thích:

- MPNet retrained rất mạnh với các câu có aspect rõ ràng như `food`, `price`, `ambiance`.
- Các worst cases chủ yếu liên quan `miscellaneous`, đặc biệt khi `experience`, `location`, `customerfriendly`, hoặc `marketing` không được diễn đạt bằng keyword nhất quán.
- Mô hình đôi khi map các tín hiệu chung về nhà hàng sang `ambiance` hoặc `service`.

### MPNet unretrained

Best cases tiêu biểu:

| source_row | Review | True/Predicted | Mean true probability |
|---:|---|---|---:|
| 628 | it is usually a little less salty than other dishes the taste is unmatched | `food` | 1.0000 |
| 31 | as for the beef the test was very good but the name kalabuna was not right | `food` | 0.9999 |
| 344 | all the staff looked very professional but their service was not that good | `service` | 0.9997 |

Worst cases tiêu biểu:

| source_row | Review | True | Predicted | Lỗi chính |
|---:|---|---|---|---|
| 532 | from today it is my favorite restaurant because the service here is very bad and the price of everything is very high | `price, service` | `food, service, miscellaneous` | thêm `food`, `miscellaneous`; miss `price` |
| 568 | it is necessary to sell the land and go here to eat the price of food is so high but everyone goes here to get the touch of a good environment | `price, ambiance` | `food, price, miscellaneous` | thêm `food`, `miscellaneous`; miss `ambiance` |
| 531 | the time was good because of the atmosphere of the restaurant and their service was also amazing | `service, ambiance` | `food, service, miscellaneous` | thêm `food`, `miscellaneous`; miss `ambiance` |

Giải thích:

- MPNet unretrained vẫn xử lý tốt các câu đơn giản, trực tiếp, đặc biệt single-label như `food` hoặc `service`.
- Worst cases cho thấy mô hình gốc có xu hướng over-predict `food` và `miscellaneous`.
- Không retrain khiến embedding chưa học tốt ranh giới nội bộ của dataset, đặc biệt giữa `ambiance`, `miscellaneous`, và các câu nói chung về nhà hàng.

### TF-IDF

Best cases tiêu biểu:

| source_row | Review | True/Predicted | Mean true probability |
|---:|---|---|---:|
| 344 | all the staff looked very professional but their service was not that good | `service` | 0.9977 |
| 779 | the restaurant is very clean and the ambience is very nice | `ambiance` | 0.9920 |
| 365 | food prices are lower than other nearby places but the food is very expensive | `price` | 0.9875 |

Worst cases tiêu biểu:

| source_row | Review | True | Predicted | Lỗi chính |
|---:|---|---|---|---|
| 417 | the timing of the meal was just right and the total cost was very less per person | `price, service` | `food, price` | thêm `food`; miss `service` |
| 479 | ordering lower priced items made the waiters experience worse and most of the food seemed overpriced | `price, service` | `food, service` | thêm `food`; miss `price` |
| 335 | the reviews of the restaurant online were bad but i found the quality of the food to be good | `food, miscellaneous` | `food, ambiance` | thêm `ambiance`; miss `miscellaneous` |

Giải thích:

- TF-IDF rất tốt khi câu có keyword rõ như `service`, `clean`, `ambience`, `price`, `expensive`.
- Worst cases xuất hiện khi nghĩa cần hiểu theo ngữ cảnh: `timing of the meal`, `waiters experience`, `reviews online`, hoặc các câu có `miscellaneous`.
- TF-IDF dễ bị kéo về `food` khi câu có nhiều từ liên quan món ăn, ngay cả khi aspect chính là giá hoặc dịch vụ.

## 9. Vì sao MPNet retrained tốt nhất?

MPNet retrained tốt hơn vì:

- Encoder đã được fine-tune theo dữ liệu ABSA hiện tại, nên embedding phản ánh tốt hơn ranh giới giữa các aspect.
- Mô hình hiểu ngữ cảnh tốt hơn TF-IDF, đặc biệt ở các câu không chỉ dựa vào từ khóa trực tiếp.
- Threshold riêng theo aspect giúp giảm lỗi over-predict/under-predict.
- Retraining cải thiện mạnh `ambiance` và `miscellaneous`, hai lớp mà representation gốc hoặc TF-IDF xử lý kém hơn.

Ví dụ cải thiện rõ:

- `ambiance`: từ `0.7619` ở MPNet unretrained lên `0.9459`.
- `miscellaneous`: từ `0.3226` ở MPNet unretrained và `0.2500` ở TF-IDF lên `0.5833`.
- Exact match: từ `0.6125` ở MPNet unretrained và `0.7313` ở TF-IDF lên `0.8688`.

## 10. Khi nào nên dùng model nào?

### Nên dùng MPNet retrained nếu ưu tiên chất lượng

Dùng cho deployment chính nếu:

- cần F1 cao nhất
- cần bắt ngữ cảnh tốt
- chấp nhận inference chậm hơn
- `miscellaneous` và `ambiance` quan trọng

Đây là lựa chọn khuyến nghị cho solution cuối cùng.

### Nên dùng TF-IDF nếu ưu tiên tốc độ hoặc baseline đơn giản

Dùng nếu:

- cần inference cực nhanh
- tài nguyên deployment hạn chế
- muốn baseline dễ debug
- không quá quan trọng `miscellaneous`

TF-IDF vẫn là baseline mạnh, nhưng không nên dùng làm model cuối nếu mục tiêu là macro-F1 cao.

### Không nên dùng MPNet unretrained làm final

MPNet unretrained chậm như MPNet retrained nhưng chất lượng thấp hơn đáng kể. Nó chỉ nên dùng để chứng minh hiệu quả của retraining.

## 11. Hạn chế còn lại

- `miscellaneous` vẫn thấp nhất dù MPNet retrained đã tốt hơn. F1 chỉ đạt `0.5833`.
- Test set chỉ có 14 mẫu `miscellaneous`, nên metric của lớp này dễ dao động.
- Một số lỗi là do ranh giới nhãn chưa thật rõ, đặc biệt giữa `ambiance` và `miscellaneous`.
- TF-IDF dùng `review_cleaned`, còn MPNet dùng `review_en`; nếu preprocessing làm mất phủ định, TF-IDF có thể bị bất lợi ở một số câu.

## 12. Đề xuất tiếp theo

1. Chọn **MPNet retrained** làm model chính.
2. Giữ **TF-IDF** làm baseline tốc độ và fallback đơn giản.
3. Tiếp tục cải thiện `miscellaneous` bằng notebook `notebooks/SBERT/04-improve_mischellaneous.ipynb`.
4. Review các file disagreement:
   - `outputs/final_comparison/mpnet_retrained_correct_tfidf_wrong.csv`
   - `outputs/final_comparison/tfidf_correct_mpnet_retrained_wrong.csv`
   - `outputs/final_comparison/sample_level_predictions.csv`
   - `outputs/final_comparison/worst_cases_all_models.csv`
5. Làm rõ guideline cho `miscellaneous`, đặc biệt các case liên quan `location`, `experience`, `restaurant reputation`, `marketing`, và `occasion`.

## 13. File output liên quan

- `outputs/final_comparison/final_model_comparison.csv`
- `outputs/final_comparison/per_aspect_f1_comparison.csv`
- `outputs/final_comparison/classification_reports.csv`
- `outputs/final_comparison/aspect_error_counts.csv`
- `outputs/final_comparison/exact_match_overlap_summary.csv`
- `outputs/final_comparison/mpnet_retrained_correct_tfidf_wrong.csv`
- `outputs/final_comparison/tfidf_correct_mpnet_retrained_wrong.csv`
- `outputs/final_comparison/case_analysis_summary.csv`
- `outputs/final_comparison/best_cases_all_models.csv`
- `outputs/final_comparison/worst_cases_all_models.csv`
- `outputs/final_comparison/mpnet_retrained_best_cases.csv`
- `outputs/final_comparison/mpnet_retrained_worst_cases.csv`
- `outputs/final_comparison/mpnet_unretrained_best_cases.csv`
- `outputs/final_comparison/mpnet_unretrained_worst_cases.csv`
- `outputs/final_comparison/tfidf_best_cases.csv`
- `outputs/final_comparison/tfidf_worst_cases.csv`
- `outputs/final_comparison/overall_metrics_comparison.png`
- `outputs/final_comparison/per_aspect_f1_comparison.png`
- `outputs/final_comparison/confusion_matrices.png`
- `outputs/final_comparison/mean_label_errors_by_model.png`
