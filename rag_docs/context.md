DATASET
Ngữ cảnh: Bất động sản Việt Nam
1. Dataset Được Đề Xuất
Tên: House Price Prediction Dataset Vietnam — 2024
Nguồn: kaggle.com/datasets/nguyentiennhan/vietnam-housing-dataset-2024
Bộ dữ liệu này chứa thông tin về nhiều bất động sản nhà ở tại Việt Nam. Nó bao gồm các thuộc tính chi tiết của từng bất động sản, như vị trí, đặc điểm vật lý, tình trạng pháp lý, tình trạng nội thất, cùng với giá bán. Bộ dữ liệu này có thể được sử dụng cho phân tích bất động sản, xây dựng mô hình dự đoán giá và phân tích xu hướng thị trường.
Thu thập dữ liệu: Dữ liệu được thu thập từ trang web batdongsan.vn.
Số dòng: 30000+

2. Backstory — Câu Chuyện Liên Quan
2.1 Làn Sóng Sốt Đất 2020–2022 và Hậu Quả Để Lại
Giai đoạn 2020–2022, thị trường BĐS Việt Nam trải qua nhiều đợt tăng giá bất thường, đặc biệt là đất nền vùng ven. Nhiều khu vực huyện ven Hà Nội và các tỉnh như Bắc Giang, Bắc Ninh, Hòa Bình, Hưng Yên ghi nhận giá đất tăng cục bộ 40–50% so với trước dịch. Cuối năm 2022, thị trường "đóng băng", nhà đầu tư buộc phải hạ giá 20–30% để thanh khoản.
Dataset được thu thập năm 2024 — sau chu kỳ sốt đất và sau giai đoạn đóng băng — nên phản ánh mặt bằng giá đã được điều chỉnh, không phải đỉnh sốt. Tuy nhiên, dấu vết của chu kỳ này vẫn có thể quan sát được: Hưng Yên xuất hiện dày đặc trong dataset với các dự án Vinhomes Ocean Park quy mô lớn, và mức Price_per_m2 tại đây có thể cao bất thường so với các tỉnh lân cận cùng quy mô kinh tế.

2.2 Thị Trường Hai Phân Khúc — Dự Án vs Nhà Phố
Thị trường BĐS Việt Nam tồn tại song song hai phân khúc có hành vi giá rất khác nhau. Nhà dự án (Vinhomes, Him Lam, Sun Casa...) có giá được định sẵn theo chính sách chủ đầu tư, hạ tầng đồng bộ và pháp lý rõ ràng. Nhà phố / đất thổ cư có giá phụ thuộc vào vị trí, mặt tiền và khả năng thương lượng — biến động cao hơn và khó dự đoán hơn.
Trong dataset, hai phân khúc này được phân biệt qua cột Is_Project, cho phép so sánh trực tiếp hành vi giá, mức độ hoàn thiện pháp lý (Has_certificate) và tương quan giữa đặc điểm vật lý với giá bán giữa hai nhóm.

2.3 Nguồn Tham Khảo Chính Thống
VnExpress: Nhà nước can thiệp khi giá nhà đất tăng hơn 20% trong 3 tháng
https://vnexpress.net/co-hoi-va-thach-thuc-cua-bat-dong-san-2022-4422754.html
https://vneconomy.vn/gia-dat-vung-ven-co-xu-huong-tiem-can-gia-khu-vuc-trung-tam.htm
https://vov.vn/kinh-te/bat-dong-san-hung-yen-tung-sot-nhat-mien-bac-gio-ra-sao-post1215445.vov

3. Tính Ngoại Lệ & Bất Ngờ Trong Dữ Liệu

1. Phân phối giá tập trung — không phải right-skewed như kỳ vọng Trái với kỳ vọng ban đầu, phân phối Price trong dataset khá đối xứng, tập trung chủ yếu trong khoảng 1–10 tỷ đồng. Điều này cho thấy dữ liệu thu thập từ batdongsan.vn có thể đã bị giới hạn ngầm ở phân khúc nhà ở phổ thông — các giao dịch biệt thự, đất dự án giá trăm tỷ gần như vắng mặt. Đây là giới hạn cần lưu ý khi suy rộng kết quả phân tích.
2. Chênh lệch giá/m² giữa các tỉnh — hai thế giới trong cùng dataset Giá/m² tại các quận trung tâm TP.HCM có thể cao hơn tỉnh vùng sâu 50–100 lần. Phân phối địa lý tạo ra hai cụm hoàn toàn tách biệt, dễ quan sát qua box plot hoặc scatter plot theo Province.
3. Nghịch lý diện tích — nhà nhỏ nhưng đắt hơn/m² Bất động sản diện tích nhỏ đôi khi có Price_per_m2 cao hơn nhà lớn, do vị trí trung tâm hoặc tiện ích đặc biệt. Hành vi mua bán tại Việt Nam không tuân theo mô hình tuyến tính thuần túy, và mối quan hệ Area vs Price_per_m2 thể hiện rõ điều này.
4. Dự án đôi khi đắt hơn nhà phố cùng diện tích Phản ánh kỳ vọng vào hạ tầng, tiện ích và thương hiệu chủ đầu tư. Đây là nghịch lý thú vị có thể kiểm chứng trực tiếp bằng cách so sánh Price_per_m2 theo cột Is_Project trong dataset.



