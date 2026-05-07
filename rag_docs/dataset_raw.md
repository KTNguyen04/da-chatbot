# Thông tin Kỹ thuật Bộ dữ liệu Bất động sản Việt Nam 2024 (Dữ liệu gốc)

Tài liệu này chứa các thông số kỹ thuật, nguồn gốc và đặc tả thuộc tính thô của bộ dữ liệu trước khi xử lý.

---

### 1. Tổng quan bộ dữ liệu (Metadata)
- **Tên bộ dữ liệu:** House Price Prediction Dataset Vietnam 2024.
- **Nguồn:** Kaggle (https://www.kaggle.com/datasets/nguyentiennhan/vietnam-housing-dataset-2024).
- **Nguồn gốc dữ liệu gốc:** Được thu thập bằng phương pháp crawl từ website batdongsan.vn.
- **Quy mô:** Bao gồm hơn 30.000 dòng dữ liệu (records).
- **Nội dung:** Thông tin chi tiết về vị trí, đặc điểm vật lý, pháp lý, nội thất và giá bán của bất động sản nhà ở tại Việt Nam.

---

### 2. Mô tả các thuộc tính (Schema) - Vị trí & Pháp lý
- **Address (String):** Địa chỉ đầy đủ ban đầu. Chứa các cấp hành chính: dự án, tên đường, phường/xã, quận/huyện, thành phố. Ví dụ: `Xã Long Hưng, Văn Giang...`
- **Project (String):** Tên dự án bất động sản (nếu có). Để trống nếu là nhà dân/không thuộc dự án. Ví dụ: `Vinhomes Ocean Park 2`.
- **Legal status (String):** Tình trạng pháp lý thô. Các giá trị thường gặp: Sổ đỏ, sổ hồng, hợp đồng mua bán, giấy tờ tay... Ví dụ (sau chuẩn hóa): `Have certificate`.

---

### 3. Mô tả các thuộc tính (Schema) - Đặc điểm Vật lý
- **Area (Decimal):** Tổng diện tích bất động sản (đơn vị: m²). Có thể là diện tích thông thủy hoặc diện tích đất. Ví dụ: `84.0`.
- **Frontage (Decimal):** Chiều rộng mặt tiền tiếp giáp đường (đơn vị: mét). Ví dụ: `5.0`.
- **Access Road (Decimal):** Độ rộng của đường hoặc hẻm dẫn vào nhà (đơn vị: mét). Ví dụ: `13.0`.
- **House direction (String):** Hướng chính của ngôi nhà. Gồm 8 hướng chính: Đông, Tây, Nam, Bắc và các hướng chéo. Ví dụ: `Đông - Bắc`.
- **Balcony direction (String):** Hướng của ban công (áp dụng nhiều cho căn hộ chung cư). Ví dụ: `Tây - Nam`.

---

### 4. Mô tả các thuộc tính (Schema) - Cấu trúc & Nội thất
- **Floors (Integer):** Tổng số tầng của công trình. Ví dụ: `4`.
- **Bedrooms (Integer):** Tổng số lượng phòng ngủ hiện có. Ví dụ: `6`.
- **Bathrooms (Integer):** Tổng số lượng phòng tắm/phòng vệ sinh. Ví dụ: `4`.
- **Furniture state (String):** Mức độ trang bị nội thất gốc. Các trạng thái: Đầy đủ (Full), cơ bản (một phần), không nội thất (nhà trống). Ví dụ: `Full`.

---

### 5. Mô tả các thuộc tính (Schema) - Kinh tế
- **Price (Decimal):** Giá bán niêm yết của bất động sản.
  - **Đơn vị:** Tỷ VNĐ (Tỷ đồng). Ví dụ: `8.6`.
  - **Lưu ý:** Đây là giá trị thô, cần kết hợp với diện tích để phân tích đơn giá.

---

**Quy tắc truy vấn RAG cho dữ liệu thô:**
- Sử dụng tài liệu này khi người dùng hỏi về nguồn gốc dữ liệu hoặc các trường dữ liệu chưa qua xử lý (ví dụ: Furniture state, Balcony direction).
- Đối với các phân tích về District, Province hoặc Price_per_m2, hãy ưu tiên tham chiếu sang file `dataset_cleaned.md`.