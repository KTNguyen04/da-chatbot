# Data Dictionary - Vietnam Housing Dataset (Cleaned)

Tài liệu này cung cấp định nghĩa chi tiết cho toàn bộ các trường dữ liệu trong bộ dữ liệu bất động sản Việt Nam đã qua xử lý.

---

## 1. Nhóm thông tin vị trí

### Trường dữ liệu: Address (Địa chỉ)
- **Kiểu dữ liệu:** String
- **Ý nghĩa:** Địa chỉ chi tiết của bất động sản.
- **Nội dung:** Bao gồm số nhà, tên đường, phường/xã, quận/huyện.
- **Ví dụ:** `"Xã Long Hưng, Văn Giang..."`
- **Ghi chú:** Đây là dữ liệu gốc dùng để trích xuất các thông tin vị trí cụ thể khác.

---

### Trường dữ liệu: Ward_Street (Phường/Xã hoặc tên đường)
- **Kiểu dữ liệu:** String
- **Ý nghĩa:** Đơn vị hành chính cấp nhỏ hoặc tên tuyến đường liên quan đến bất động sản.
- **Ví dụ:** `"Phường 11"`
- **Ghi chú:** Dùng để phân tích chi tiết theo khu vực nhỏ hơn quận/huyện.

---

### Trường dữ liệu: District (Quận/Huyện)
- **Kiểu dữ liệu:** String
- **Ý nghĩa:** Quận hoặc huyện nơi bất động sản tọa lạc.
- **Ví dụ:** `"Gò Vấp"`, `"Quận 7"`

---

### Trường dữ liệu: Province (Tỉnh/Thành phố)
- **Kiểu dữ liệu:** String
- **Ý nghĩa:** Tỉnh hoặc thành phố trực thuộc trung ương.
- **Ví dụ:** `"Hồ Chí Minh"`, `"Hà Nội"`

---

## 2. Nhóm thông tin diện tích và cấu trúc

### Trường dữ liệu: Area (Diện tích)
- **Kiểu dữ liệu:** Float
- **Ý nghĩa:** Diện tích sử dụng hoặc diện tích đất của bất động sản.
- **Đơn vị:** mét vuông (m²).
- **Ví dụ:** `84.0`
- **Ghi chú:** Có thể là diện tích thông thủy hoặc diện tích đất tùy loại hình nhà ở.

---

### Trường dữ liệu: Area_group (Nhóm diện tích)
- **Kiểu dữ liệu:** String
- **Ý nghĩa:** Phân nhóm diện tích nhằm phục vụ thống kê và trực quan hóa dữ liệu.
- **Ví dụ:** `"80-120m²"`
- **Ghi chú:** Các nhóm diện tích thường được chia theo khoảng giá trị như:
  - `< 40m²`
  - `40-80m²`
  - `80-120m²`
  - `> 120m²`

---

### Trường dữ liệu: Frontage (Mặt tiền)
- **Kiểu dữ liệu:** Float
- **Ý nghĩa:** Độ rộng mặt tiền của bất động sản.
- **Đơn vị:** mét (m).
- **Ví dụ:** `5.0`
- **Ghi chú:** Là chiều ngang tiếp giáp mặt đường chính.

---

### Trường dữ liệu: Access Road (Đường vào)
- **Kiểu dữ liệu:** Float
- **Ý nghĩa:** Độ rộng đường hoặc hẻm dẫn vào nhà.
- **Đơn vị:** mét (m).
- **Ví dụ:** `13.0`
- **Ghi chú:** Thông số quan trọng để đánh giá khả năng di chuyển và giá trị bất động sản.

---

### Trường dữ liệu: Floors (Số tầng)
- **Kiểu dữ liệu:** Integer
- **Ý nghĩa:** Tổng số tầng của căn nhà.
- **Ví dụ:** `4`

---

### Trường dữ liệu: Bedrooms (Phòng ngủ)
- **Kiểu dữ liệu:** Integer
- **Ý nghĩa:** Tổng số phòng ngủ của bất động sản.
- **Ví dụ:** `6`

---

### Trường dữ liệu: Bathrooms (Phòng tắm / WC)
- **Kiểu dữ liệu:** Integer
- **Ý nghĩa:** Tổng số phòng vệ sinh hoặc phòng tắm.
- **Ví dụ:** `4`

---

## 3. Nhóm thông tin hướng nhà

### Trường dữ liệu: House direction (Hướng nhà)
- **Kiểu dữ liệu:** String
- **Ý nghĩa:** Hướng cửa chính hoặc mặt chính của ngôi nhà.
- **Ví dụ:** `"Đông - Bắc"`
- **Các giá trị phổ biến:**
  - Đông
  - Tây
  - Nam
  - Bắc
  - Đông Bắc
  - Đông Nam
  - Tây Bắc
  - Tây Nam

---

### Trường dữ liệu: Balcony direction (Hướng ban công)
- **Kiểu dữ liệu:** String
- **Ý nghĩa:** Hướng của ban công căn hộ hoặc nhà ở.
- **Ví dụ:** `"Tây - Nam"`
- **Ghi chú:** Chủ yếu áp dụng cho chung cư hoặc nhà có ban công lớn.

---

## 4. Nhóm thông tin pháp lý và nội thất

### Trường dữ liệu: Legal status (Tình trạng pháp lý)
- **Kiểu dữ liệu:** String
- **Ý nghĩa:** Trạng thái pháp lý của bất động sản.
- **Ví dụ:** `"Have certificate"`
- **Các giá trị phổ biến:**
  - Sổ đỏ
  - Sổ hồng
  - Hợp đồng mua bán
  - Giấy tay
  - Chờ cấp sổ

---

### Trường dữ liệu: Has_certificate (Có giấy chứng nhận)
- **Kiểu dữ liệu:** Boolean (0/1)
- **Ý nghĩa:** Xác định bất động sản đã có giấy chứng nhận quyền sở hữu hay chưa.
- **Ví dụ:** `1`
- **Quy ước:**
  - `1`: Đã có sổ
  - `0`: Chưa có sổ hoặc chưa rõ pháp lý

---

### Trường dữ liệu: Furniture state (Tình trạng nội thất)
- **Kiểu dữ liệu:** String
- **Ý nghĩa:** Mức độ hoàn thiện nội thất của bất động sản.
- **Ví dụ:** `"Full"`
- **Các giá trị phổ biến:**
  - Full nội thất
  - Nội thất cơ bản
  - Nhà thô
  - Không nội thất

---

## 5. Nhóm thông tin giá

### Trường dữ liệu: Price (Giá bán)
- **Kiểu dữ liệu:** Float
- **Ý nghĩa:** Tổng giá trị bất động sản.
- **Đơn vị:** Tỷ VNĐ.
- **Ví dụ:** `8.6`
- **Diễn giải:** `8.6` tương đương khoảng `8 tỷ 600 triệu VNĐ`.

---

### Trường dữ liệu: Price_per_m2 (Đơn giá theo diện tích)
- **Kiểu dữ liệu:** Float
- **Ý nghĩa:** Giá bán trung bình trên mỗi mét vuông.
- **Đơn vị:** Tỷ VNĐ/m².
- **Ví dụ:** `0.1024`
- **Diễn giải:** `0.1024` tương đương khoảng `102.4 triệu VNĐ/m²`.

#### Công thức tính

\[
Price\_per\_m2 = \frac{Price}{Area}
\]

---

## 6. Nhóm thông tin dự án

### Trường dữ liệu: Project (Tên dự án)
- **Kiểu dữ liệu:** String
- **Ý nghĩa:** Tên dự án bất động sản mà căn nhà/căn hộ thuộc về.
- **Ví dụ:** `"Vinhomes Ocean Park 2"`
- **Ghi chú:** Để trống nếu là nhà dân riêng lẻ hoặc không thuộc dự án.

---

### Trường dữ liệu: Is_Project (Thuộc dự án)
- **Kiểu dữ liệu:** Boolean (0/1)
- **Ý nghĩa:** Xác định bất động sản có thuộc dự án hay không.
- **Ví dụ:** `1`
- **Quy ước:**
  - `1`: Có thuộc dự án
  - `0`: Nhà riêng lẻ / không thuộc dự án

---

# Tổng quan kiểu dữ liệu

| Tên cột | Kiểu dữ liệu | Ví dụ |
|---|---|---|
| Address | String | Xã Long Hưng |
| Area | Float | 84.0 |
| Frontage | Float | 5.0 |
| Access Road | Float | 13.0 |
| House direction | String | Đông - Bắc |
| Balcony direction | String | Tây - Nam |
| Floors | Integer | 4 |
| Bedrooms | Integer | 6 |
| Bathrooms | Integer | 4 |
| Legal status | String | Have certificate |
| Furniture state | String | Full |
| Price | Float | 8.6 |
| Project | String | Vinhomes Ocean Park 2 |
| Ward_Street | String | Phường 11 |
| District | String | Gò Vấp |
| Province | String | Hồ Chí Minh |
| Price_per_m2 | Float | 0.1024 |
| Is_Project | Boolean | 1 |
| Area_group | String | 80-120m² |
| Has_certificate | Boolean | 1 |

---

# Quy tắc phản hồi của Chatbot

1. Nếu người dùng hỏi `"Cột [tên cột] là gì?"`, chatbot phải trả về đúng phần mô tả của trường dữ liệu tương ứng.

2. Khi đề cập đến giá:
   - Luôn nhấn mạnh đơn vị là **Tỷ VNĐ**.
   - Nếu cần, có thể diễn giải sang triệu VNĐ để dễ hiểu.

3. Khi người dùng hỏi về `"pháp lý"`:
   - Tổng hợp thông tin từ cả:
     - `Legal status`
     - `Has_certificate`

4. Khi người dùng hỏi về `"diện tích"`:
   - Phân biệt rõ:
     - `Area` → diện tích thực tế
     - `Area_group` → nhóm dùng cho thống kê/phân tích.

5. Khi phân tích dữ liệu:
   - `Price_per_m2` nên được ưu tiên để so sánh giá trị bất động sản giữa các khu vực khác nhau.