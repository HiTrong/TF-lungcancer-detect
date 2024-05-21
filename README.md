- Ngày bắt đầu dự án: 02/04/2024

# Tensorflow - Lung Cancer Detect - Project

![Ảnh minh họa](https://www.verywellhealth.com/thmb/yp3xEa3kKfCFAQDvKOXJT9BAmr8=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/non-small-cell-lung-cancer-2249281_final-ea85b1b20eb748fb806d5ed11284dfd8.png)

# 1. Tìm hiểu các loại ung thư phổi

- Để có thể xây dựng được một mô hình nhận dạng ung thư phổi thông qua ảnh chụp CT-Scan ở ngực thì với vị trí là đại diễn kĩ thuật ta rất cần hiểu sơ lược về kiến thức y khoa đối với loại bệnh này. Điều này sẽ giúp ích cho chúng ta trong quá trình xây dựng mô hình, kiểm nghiệm, hiểu rõ bản chất hơn. Ung thư phổi phổ biến bao gồm 3 loại: Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma.

> 1. Adenocarcinoma – Ung thư biểu mô tuyến
Đây là dạng ung thư phổi phổ biến nhất, chiếm khoảng 30% trong tổng số các trường hợp ung thư phổi và khoảng 40% trong các trường hợp ung thư phổi không phải tế bào nhỏ (NSCLC). Thường được tìm thấy ở vùng ngoại biên của phổi, trong các tuyến tiết ra chất nhầy và giúp chúng ta thở.
> 2. Large Cell Carcinoma – Ung thư tế bào lớn
Loại ung thư phổi này phát triển và lan rộng nhanh chóng, chiếm khoảng 10-15% trong tổng số các trường hợp ung thư phổi không phải tế bào nhỏ (NSCLC). Có thể xuất hiện ở bất kỳ vị trí nào trong phổi.
> 3. Squamous Cell Carcinoma – Ung thư biểu mô tế bào vảy
Chiếm khoảng 30% trong tổng số các trường hợp ung thư phổi không phải tế bào nhỏ (NSCLC). Thường liên quan đến việc hút thuốc lá và được tìm thấy ở trung tâm của phổi, nơi các phế quản lớn nối với khí quản và phổi.

# 2. Tìm hiểu về tập dữ liệu

- Nguồn: [Chest CT-Scan images Dataset - Kaggle](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images/data)
- Mô tả:

> - Đây là tập dữ liệu về hình ảnh của 3 loại ung thư phổi đã trình bày ở phần 1. Mỗi loại ung thư phổi có một folder trong đó là ảnh chụp CT-Scan ngực của những người mắc bệnh. Ngoài ra dữ liệu còn chứa một folder ảnh chụp CT-Scan ngực của những người bình thường khác.
> - Toàn bộ ảnh được chia trong 3 folder: training set chiếm 70%, testing set chiếm 20%, validation set chiếm 10%. Bên trong mỗi folder này chứa 4 folder về tình trạng loại bệnh ung thư phổi
> - Tập dữ liệu được sử dụng để cho mục địch huấn luyện mô hình học máy và học sâu (CNN)
> - Tất cả hình ảnh được sắp xếp theo đúng sơ đồ mà Tensorflow có thể load thành dữ liệu hiệu quả

# 3. TensorFlow và mô hình nơ-ron tích chập (CNN)

## 3.1. TensorFlow

- TensorFlow là một thư viện mã nguồn mở của Google, được sử dụng rộng rãi để xây dựng và huấn luyện các mô hình học sâu. Một số điểm nổi bật của - TensorFlow bao gồm:

> - Đa nền tảng: TensorFlow có thể chạy trên nhiều nền tảng khác nhau, bao gồm CPU, GPU và TPU, giúp tối ưu hóa hiệu suất xử lý.
> - Thao tác tensor: TensorFlow làm việc chủ yếu với các tensors, là các mảng dữ liệu đa chiều. Điều này giúp xử lý các phép toán đại số tuyến tính phức tạp một cách hiệu quả.
> - Thư viện phong phú: TensorFlow cung cấp nhiều API cấp cao như Keras để dễ dàng xây dựng và huấn luyện các mô hình học sâu.
> - Cộng đồng và tài liệu: Với một cộng đồng người dùng lớn và nhiều tài liệu hỗ trợ, TensorFlow rất thân thiện với người mới bắt đầu cũng như những chuyên gia.

## 3.2. Mô hình Nơ-ron Tích chập (CNN)

- Mô hình nơ-ron tích chập (CNN) là một loại mạng nơ-ron nhân tạo đặc biệt mạnh mẽ trong các bài toán về xử lý ảnh và nhận diện đối tượng. CNN có một số đặc điểm chính sau:
> - Lớp tích chập (Convolutional Layer): Đây là lớp cốt lõi của CNN, nơi các bộ lọc (filters) hoặc các hạt nhân (kernels) được sử dụng để phát hiện các đặc trưng cục bộ từ dữ liệu đầu vào. Bộ lọc này trượt qua ảnh và thực hiện phép tích chập để tạo ra bản đồ đặc trưng (feature map).
> - Lớp gộp (Pooling Layer): Sau lớp tích chập, lớp gộp thường được sử dụng để giảm kích thước của bản đồ đặc trưng và giảm thiểu số lượng tham số và tính toán. Lớp gộp phổ biến là gộp cực đại (max pooling).
> - Lớp kết nối đầy đủ (Fully Connected Layer): Các lớp cuối cùng trong CNN thường là các lớp kết nối đầy đủ, nơi các neuron kết nối với tất cả các neuron trong lớp trước đó. Các lớp này giúp kết hợp các đặc trưng đã trích xuất để đưa ra quyết định cuối cùng (ví dụ: phân loại ảnh).
> - Tính không gian: CNN khai thác tính không gian của dữ liệu, tức là các mối quan hệ cục bộ giữa các điểm dữ liệu (như các điểm ảnh trong một hình ảnh), làm cho nó đặc biệt hiệu quả trong các bài toán liên quan đến thị giác máy tính.

- Ứng dụng
> - Xử lý ảnh: Nhận diện đối tượng, phân loại ảnh, phát hiện biên, v.v.
> - Thị giác máy tính: Nhận diện khuôn mặt, phân tích video, tự lái xe, v.v.
> - Xử lý ngôn ngữ tự nhiên: Mặc dù ít phổ biến hơn, CNN cũng được áp dụng trong các bài toán như phân loại văn bản và phân tích cảm xúc.

- Ở trong project này ta sẽ sử dụng TensorFlow để xây dựng một mô hình Nơ-ron tích chập CNN để nhận dạng ung thư phổi thông qua hình ảnh CT-Scan ở ngực

# 4. Xây dựng mô hình

## 4.1. Một số thử nghiệm, sai lầm và biện pháp giải quyết

### Thử nghiệm 1:

#### Ghi chép thử nghiệm:

- Số lần chạy thử: 53
- Số lần thay đổi kết cấu setup: 36

A. Sai lầm 1
- Trong bước 4.3 Import dữ liệu và tiền xử lý dữ liệu, dữ liệu được load vào điều chỉnh kích thước, scale và **color mode là grey**
> - Với suy nghĩ rằng nếu đưa color mode về grey ta có thể giảm số kênh xuống từ 3 (RGB) về 1 (grey) sẽ giúp mô hình giảm phức tạp tính toán dễ dàng hơn. Khi chuyển về color mode là grey các đặc trưng vẫn sẽ được giữ nguyên và dễ dàng khám phá bởi mô hình hơn.
> - Nhưng đây là một sự sai lầm vì khi thử nghiệm mô hình đã bị sai lệch và rơi vào mọi trường hợp cần tránh (Underfitting/Overfitting). Tuy nhiên ta cũng không đủ cơ sở khẳng định sai lầm này hoàn toàn gây nên Underfitting/Overfitting. Một loạt điều chỉnh cho thử nghiệm 1 và ta tiếp tục tìm ra sai lầm 2.

B. Sai lầm 2
- Trong bước 4.5 Xây dựng mô hình cấu trúc của mạng như sau với sequential (resize and rescale) và sequential_1 (Ngẫu nhiên lật hoặc xoay ảnh):

| Layer (type)                    | Output Shape           |       Param # |
|---------------------------------|------------------------|---------------|
| sequential (Sequential)         | (32, 256, 256, 1)      |             0 |
| sequential_1 (Sequential)       | (32, 256, 256, 1)      |             0 |
| conv2d (Conv2D)                 | (32, 254, 254, 32)     |           320 |
| max_pooling2d (MaxPooling2D)    | (32, 127, 127, 32)     |             0 |
| conv2d_1 (Conv2D)               | (32, 125, 125, 64)     |        18,496 |
| max_pooling2d_1 (MaxPooling2D)  | (32, 62, 62, 64)       |             0 |
| conv2d_2 (Conv2D)               | (32, 60, 60, 128)      |        73,856 |
| max_pooling2d_2 (MaxPooling2D)  | (32, 30, 30, 128)      |             0 |
| conv2d_3 (Conv2D)               | (32, 28, 28, 256)      |       295,168 |
| max_pooling2d_3 (MaxPooling2D)  | (32, 14, 14, 256)      |             0 |
| flatten (Flatten)               | (32, 50176)            |             0 |
| dense (Dense)                   | (32, 64)               |     3,211,328 |
| dense_1 (Dense)                 | (32, 4)                |           260 |

> - Nhìn chung thì đây là một cấu trúc mạng CNN đơn giản với bộ lọc tăng dần số lượng từ 32 đến 256 giúp mạng học được các đặc trưng từ đơn giản đến phức tạp một cách hiệu quả. Có áp dụng tăng cường dữ liệu để tránh overfitting cũng như một số kĩ thuật khác,...
> - Tuy nhiên kết quả cho ra đáng quan ngại với hiệu suất rất cao trên tập train cực kì tệ trên tập valid (Overfitting). Phương hướng giải quyết cho sai lầm này là các Layers giảm sự phức tạp nơ-ron như: Dropout, Regularization,... Ngoài ra thêm Data Augmentation ở tập dữ liệu, điều chỉnh learning rate,... Tất cả những nổ lực này đều không cải thiện được tình hình overfitting thậm chí còn dẫn tới Underfitting và Vanishing Gradient.

#### Phương hướng giải quyết:

- Các vấn đề từ hàng loạt lần chạy thử trong thử nghiệm 1 bao gồm: Overfitting nặng, Underfitting ở vài trường hợp và Vanishing Gradient diễn ra ở rất nhiều Epoch.
- Phương pháp giải quyết:
1. Xem xét lại cách tiền xử lý tập dữ liệu: Ta cần xác định được liệu kích thước, scale, color mode được xử lý có thật sự đúng đắn giữ được đặc trưng của ảnh giúp máy học chính xác
2. Xem xét lại kiến trúc mạng CNN đã xây dựng: Liệu mô hình có quá phức tạp, nhiều tham số ? Hay các tham số không được điều chính tối ưu xảy ra hiện tượng Vanishing Gradient? Cần thử nghiệm thêm trong việc thay đổi cấu trúc mạng cũng như tham khảo các mạng phổ biến để xử lý vấn đề Vanishing Gradient
3. Áp dụng các kĩ thuật như Early Stopping, LR Reducing Rate,... 

### Thử nghiệm 2:

#### Ghi chép thử nghiệm:

- Từ thử nghiệm 1 ta rút ra được nhiều kết quả cũng như hạn chế cần giải quyết trong lần thử nghiệm 2 này!
- Số lần chạy thử: 81
- Số lần thay đổi kết cấu setup: 49

A. Giải quyết vấn đề 1: Xem xét lại cách tiền xử lý tập dữ liệu
- Không có ảnh hưởng đáng kể của kích thước và scale đối với mô hình sau nhiều một vài lần thay đổi và chạy thử
- Xem xét đến sự cần thiết trong việc tăng cường dữ liệu thêm
- Thứ ta nghi ngờ nhất đó chính là color mode, nhưng thay đổi lại color mode sẽ kéo theo sự thay đổi trong cấu trúc mạng hiện tại chưa giải quyết vấn đề ở thử nghiệm 1. Chính vì vậy ta sẽ giải quyết nó cùng với vấn đề 2.

B. Giải quyết vấn đề 2: Xem xét lại kiến trúc mạng CNN đã xây dựng
- Xem xét lại sự nghi ngờ về mối quan hệ giữa các vấn đề: Overfitting nặng, Underfitting ở vài trường hợp và Vanishing Gradient diễn ra ở rất nhiều Epoch. Với kiến thức chưa dày dặn kinh nghiệm, ta cần tham khảo một số cấu trúc có thể giải quyết Vanishing Gradient trước từ đó cải thiện lại vấn đề fitting.
- ResNet50 là một mô hình có khả năng giải quyết vấn đề Vanishing Gradient nhờ vào các khối residual (Residual Blocks). Việc sử dụng các kết nối tắt (skip connections) trong ResNet-50 giúp truyền gradient một cách hiệu quả hơn qua các lớp, từ đó giảm thiểu hiện tượng vanishing gradient.
- Cấu trúc ResNet50:

![ResNet50](https://miro.medium.com/v2/resize:fit:1400/0*tH9evuOFqk8F41FG.png)

- Sau rất nhiều thử nghiêm với các lần thử trong thử nghiệm 2 các mảnh chìa khóa dần kết nối với nhau tạo ra một giải pháp toàn diện giải quyết mọi vấn đề đã đề cập:
> - Với sự thay đổi về color mode RGB: Mô hình lại gặp vấn đề xử lý phức tạp, số lượng tham số rất nhiều tuy nhiên Overfitting/Underfitting và Vanishing Gradient vẫn xảy ra (Chưa áp dụng ResNet50) dù áp dụng các phương pháp sử dụng thêm các lớp khác hoặc lược bỏ đi một số lớp .
> - ResNet50 với khả năng trích xuất các đặc trưng phức tạp từ hình ảnh. Các lớp sâu hơn có khả năng học các đặc trưng trừu tượng và cao cấp hơn, giúp mô hình nắm bắt được nhiều thông tin hơn từ dữ liệu. Giải quyết được vấn đề Vanishing Gradient vừa giải quyết được tốt sự phức tạp khi ta thay đổi từ grey mode sang RGB. Các vấn đề lần lượt được giải quyết song vẫn có khuyết điểm trong phương pháp này đó là: Kích thước mô hình và yêu cầu tài nguyên, Khó khăn trong việc điều chỉnh và tối ưu hóa, Latency cao trong ứng dụng thời gian thực,... Nhưng với kết quả đem lại trên thử nghiệm ta sẽ chấp nhận phương án này!

#### Phương án cuối cùng:

- Từ rất nhiều sự tìm hiểu và thử nghiệm mài mò, ta rút kết được phương án cụ thể và việc xây dựng mô hình dựa trên phương án đó sẽ được thực hiện bên dưới:

## Các phần tiếp theo:

- Hãy mở file Notebook tại github này!