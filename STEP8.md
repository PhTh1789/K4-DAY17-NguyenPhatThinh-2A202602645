# Bước 8 — Phân tích kết quả

File này trả lời bốn câu hỏi phân tích của Bước 8 trong `Guide.md`. Các số liệu
được tạo bằng benchmark offline deterministic, không cần API key. Token được ước
lượng bằng heuristic ổn định thay vì tokenizer của một provider cụ thể.

## Kết quả benchmark

### Standard Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 1,521 | 16,250 | 0.000 | 0.200 | 0 | 0 |
| Advanced | 1,907 | 25,592 | 1.000 | 1.000 | 370 | 0 |

### Long-Context Stress Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 305 | 22,681 | 0.000 | 0.200 | 0 | 0 |
| Advanced | 457 | 13,791 | 1.000 | 1.000 | 258 | 3 |

## 1. Vì sao Advanced có recall tốt hơn Baseline?

Baseline chỉ giữ message theo `thread_id`, nên recall question ở thread mới
không nhìn thấy fact của thread cũ. Advanced tách các fact ổn định thành field có
cấu trúc trong `User.md` và nạp profile theo `user_id` ở thread mới. Khi có
correction, các field đơn như `location` và `profession` được ghi đè bằng giá trị
mới thay vì giữ đồng thời fact cũ sai.

Do đó Advanced đạt cross-session recall `1.000`, trong khi Baseline đạt `0.000`
trên cả hai suite. Chênh lệch này phản ánh đúng thiết kế: Baseline là mốc không
có long-term memory, còn Advanced có persistent memory.

## 2. Vì sao Advanced có thể tốn hơn ở hội thoại ngắn?

Trong Standard Benchmark, Advanced xử lý `25,592` prompt tokens, cao hơn
Baseline `16,250` khoảng **57.5%**. Mỗi lượt Advanced phải mang thêm persistent
profile và metadata của compact context. Các thread tiêu chuẩn chưa đủ dài để
kích hoạt compact (`Compactions = 0`), nên chi phí bổ sung chưa được bù lại bằng
việc nén lịch sử.

Advanced cũng sinh nhiều agent tokens hơn vì phản hồi chứa fact cụ thể và tuân
theo response style đã lưu. Persistent memory vì vậy không miễn phí; với hội
thoại ngắn, lợi ích recall có thể đi kèm prompt và output cost cao hơn.

## 3. Vì sao compact giúp Advanced có lợi thế ở hội thoại dài?

Baseline kéo lại toàn bộ lịch sử của thread, nên prompt context tăng theo từng
lượt. Advanced giữ một số message gần nhất và nén phần cũ thành summary khi vượt
ngưỡng. Trong stress suite, compact xảy ra 3 lần và giảm prompt tokens processed
từ `22,681` xuống `13,791`, tương đương khoảng **39.2%**, trong khi recall vẫn
đạt `1.000`.

Compact tối ưu chủ yếu lượng prompt context phải xử lý; nó không nhất thiết giảm
`Agent tokens only`, vì độ dài câu trả lời còn phụ thuộc nội dung và response
style. Compact cũng có rủi ro làm mất chi tiết tạm thời nếu summary không giữ
đúng thông tin quan trọng.

## 4. Memory file tăng trưởng ra sao và có rủi ro gì?

Advanced tạo thêm 370 byte persistent memory ở Standard Benchmark và 258 byte ở
stress suite. Baseline không có persistent file nên memory growth bằng 0.
Persistent memory cải thiện recall nhưng có thể:

- lưu nhầm fact hoặc ghi đè bằng phát biểu mơ hồ;
- giữ thông tin cá nhân quá lâu;
- làm profile và prompt phình dần;
- đưa dữ liệu cũ hoặc không phù hợp vào câu trả lời về sau;
- gây rò rỉ ngữ cảnh nếu hệ thống không cô lập user hoặc tenant đúng cách.

Bản triển khai giảm các rủi ro trên bằng extraction ưu tiên precision, bỏ qua
recall question và noise đã biết, giới hạn mỗi fact 500 ký tự, giới hạn profile
64 KiB, giữ tối đa 16 preference tích lũy và ghi file theo cơ chế atomic replace.

## Bonus: confidence threshold và guardrail

`PROFILE_CONFIDENCE_THRESHOLD` mặc định là `0.8`. Fact chỉ được ghi khi
confidence heuristic đạt ngưỡng; correction rõ ràng cho nơi ở hoặc nghề nghiệp
có confidence cao hơn preference chung như `interests`.

Confidence threshold giúp bảo vệ **recall precision** bằng cách giảm nguy cơ fact
mơ hồ ghi đè fact đúng. Giới hạn profile và preference giúp chặn memory growth,
từ đó giới hạn lượng persistent context có thể được đưa lại vào prompt. Tuy
nhiên, benchmark hiện tại không phải ablation study nên chưa thể quy một con số
recall hoặc token improvement riêng cho confidence threshold.

Trade-off là threshold quá cao có thể bỏ sót fact hợp lệ. Điểm confidence hiện
là heuristic tĩnh, chưa được calibration bằng dữ liệu thực tế. Trong production
cần thêm consent, retention/deletion policy, theo dõi false-positive và
false-negative, cùng tenant isolation trước khi lưu dữ liệu nhạy cảm.

## Giới hạn của phép đo

- `Cross-session recall` dùng substring matching sau Unicode normalization,
  chưa đánh giá tương đương ngữ nghĩa.
- `Response quality` là heuristic deterministic dựa trên factual coverage và
  readability, không phải đánh giá từ người dùng hoặc LLM judge.
- Token là ước lượng theo số ký tự, không đại diện chính xác billing của từng
  provider.
- Stress recall tập trung vào persistent profile facts, chưa đo đầy đủ khả năng
  giữ transient information bên trong compact summary.
- Kết quả xác nhận offline behavior; live mode còn phụ thuộc model, provider,
  network và prompt behavior thực tế.

## Cách tái tạo kết quả

Chạy từ root repository:

```bash
python src/benchmark.py
pytest src/test_agents.py -v
```

Benchmark tạo state tạm riêng cho mỗi suite nên không đọc profile cũ trong
`state/` và cho kết quả lặp lại được giữa các lần chạy.
