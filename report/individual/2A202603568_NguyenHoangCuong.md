# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603568
- **Họ và tên:** Nguyễn Hoàng Cương
- **SEED** (in ở ô đầu notebook): 2639

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☐ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☐ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☐ |

## 2. Kết quả chính

*Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn.*
                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            1.000    0.78      0.876        0.788   
2. BoW + MLP (deep learning)         1.000    0.89      0.942        0.947   
3a. Encoder (MiniLM) + kNN           0.845    0.93      0.886        0.927   
3b. Encoder (MiniLM) + LogReg        0.969    0.93      0.949        0.947   
3c. Encoder (MiniLM) + MLP head      0.969    0.94      0.954        0.947   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            3.523      27921.362       0.047   
2. BoW + MLP (deep learning)         4.651      48076.380       0.159   
3a. Encoder (MiniLM) + kNN          65.508         29.362      35.476   
3b. Encoder (MiniLM) + LogReg       65.912         32.387      19.548   
3c. Encoder (MiniLM) + MLP head     66.358         32.463      22.670   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.075           0.0  
2. BoW + MLP (deep learning)          0.214           0.0  
3a. Encoder (MiniLM) + kNN           57.452           0.0  
3b. Encoder (MiniLM) + LogReg        30.842           0.0  
3c. Encoder (MiniLM) + MLP head      31.848           0.0  

## 3. Dự đoán và thực tế

*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*
1. `llm_models`
- Dự đoán: Terra cao hơn Luna khoảng 0.05 F1 và thắng Embedding.
- Thực tế: Chênh lệch nhỏ hơn dự đoán, không thắng rõ Embedding.
- Vì sao lệch: Bài toán spam SMS khá đơn giản, embedding đã chứa nhiều thông tin ngữ nghĩa nên lợi thế của model lớn không phát huy mạnh.

2. `tradeoff`
- Dự đoán: Embedding + MLP nằm trên đường Pareto.
- Thực tế: Đường Pareto chỉ gồm BoW + LogReg và BoW + MLP.
- Vì sao lệch: Embedding có F1 tốt hơn nhưng độ trễ tăng đủ nhiều để bị lấn át trong đồ thị lỗi - độ trễ.

3. `emb_tricky`
- Dự đoán: Encoder + LogReg sẽ vượt BoW rõ rệt trên các câu đánh lừa.
- Thực tế: Cả hai đều đúng 5/6.
- Vì sao lệch: Bộ ví dụ quá nhỏ nên chưa thể hiện rõ hết lợi thế ngữ nghĩa của embedding.

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*
Model tụt nhiều nhất là BoW + LogReg vì phụ thuộc mạnh vào từ khóa và coi văn bản như tập token rời rạc.
Case 1: ham_looks_spam

VD: Congrats!!! You WON the bet lol, I owe you a beer.

- Nhãn thật: ham
- BoW dễ dự đoán spam vì thấy các token như WON, Congrats.
- Cơ chế lỗi: model coi các từ khóa này là tín hiệu spam mạnh mà không hiểu ngữ cảnh cuộc trò chuyện.

Case 2: obfuscation

VD: C0ngr4tulat10ns! F R E E ...

- Nhãn thật: spam
- BoW dễ dự đoán ham.
- Cơ chế lỗi: từ khóa spam bị biến dạng nên không còn khớp với vocabulary.

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*

Đã dùng AI để:
- Giải thích các khái niệm:
  - Data leakage
  - Sigmoid và sigmoid saturation
  - Precision, Recall, F1-score
  - Gradient Descent và Backpropagation
  - Embedding, cosine similarity và kNN

- Hỗ trợ debug các hàm:
  - `vectorize`
  - `metrics`
  - `predict_proba`
  - `knn_predict`
  - `llm_classify_batch`
  - `parse_label`

Những lỗi đã sửa:
1. Đặt `return` bên trong vòng `for` của `llm_classify_batch`, làm hàm chỉ xử lý phần tử đầu tiên.
2. Hiểu sai ý nghĩa của logits, gradient và hidden layer trong MLP.

Điều hiểu rõ hơn sau bài:
Hiểu rõ hơn vì sao data leakage làm điểm test bị "ảo"; hiểu gradient và sigmoid saturation; hiểu cách implement một số thuật toán training.
```**````**
