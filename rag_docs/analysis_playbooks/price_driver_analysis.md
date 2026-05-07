# Price Driver Analysis Playbook

intent_tags: [data_analysis, price_driver]
columns: [Price, Area, Floors, Bedrooms, Bathrooms, Frontage]

question_examples:
- "Diện tích và cấu trúc ảnh hưởng như thế nào đến giá?"
- "Nhà to hơn có luôn đắt hơn không?"
- "Số tầng, phòng ngủ, phòng tắm có làm giá tăng không?"

analysis_framework:
1. Xác định biến mục tiêu: Price.
2. Ước lượng mức liên hệ giữa Price với Area/Floors/Bedrooms/Bathrooms/Frontage.
3. So sánh giá theo nhóm cấu trúc (group median/mean).
4. Kiểm tra trường hợp ngoại lệ và kết luận có điều kiện.

chart_recommendations:
- scatter: Area vs Price (xu hướng tổng quát)
- box/bar: Price theo Floors/Bedrooms/Bathrooms
- bar: median Price theo bin diện tích

answer_guidelines:
- Luôn nêu bằng chứng định lượng (corr, median, mean, hoặc chênh lệch theo nhóm).
- Luôn nêu ít nhất một ngoại lệ hoặc trường hợp không theo xu hướng chung.
- Kết luận theo hướng xác suất/điều kiện, tránh khẳng định tuyệt đối.

safe_conclusion_pattern:
- "Nhìn chung diện tích lớn hơn thường đi kèm giá cao hơn, nhưng không phải luôn luôn."
- "Các yếu tố cấu trúc có ảnh hưởng, song mức độ ảnh hưởng thay đổi theo khu vực/pháp lý."

