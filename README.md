# ML_Traffic_sign_detection

## 0. Giới thiệu

Đây là source code đồ án môn <b>Máy học</b> CSS114 trường UIT đề tài <b>Phát hiện và phân loại biển báo</b> gồm 02 thành viên:
- Bùi Cao Doanh 19521366
- Lê Phước Nhật Nam

Souce code được xây dựng từ platform <b>Detectron2</b>. Ở đây sử dụng 02 backbone chính để phân loại biển báo đó là <b>ResNet101</b> và <b>ResNet50</b>.

## 1. Dữ liệu thực hiện

Chúng tôi sử dụng dữ liệu VNTSDB của nhóm tác giả Hoàng Hữu Tín (UIT K10) để phát hiện biển báo. Dữ liệu có thể tải về tại đây: https://github.com/Flavius1996/VNTS-faster-rcnn

Ở dữ liệu phân loại biển báo, chúng tôi sử dụng dữ liệu Traffic sign detection của Zalo AI Challenge 2020. Được công bố ở https://challenge.zalo.ai/portal/traffic-sign-detection

## 2. Cài đặt

Chúng tôi thực hiện cài đặt source code trên máy ảo GPU `Tesla K80` . Chúng ta có thể cài đặt source code bằng các bước đơn giản:

### 2.1. Tải về source code

```
git clone https://github.com/caodoanh2001/ML_Traffic_sign_detection
```

### 2.2. Cài đặt
cd vào thư mục vừa clone về
```
cd ML_Traffic_sign_detection
```
Cài đặt môi trường cần thiết
```
pip install -r requirements.txt
```
Cài đặt source code:
```
python -m pip install -e .
```
## 3. Sử dụng

### 3.1 Huấn luyện mô hình

Chúng ta có thể huấn luyện lại mô hình bằng cách sau:

```
python train.py --train_dir <Thư mục ảnh train> \
              --name 'VNTSDB' \
              --json_dir <Đường dẫn file json annotation> \
              --iter <Số lượng iter> \
              --batch <Batch size> \
              --lr <learning rate> \
              --resume <Train tiêp tục từ weight trước> \
              --config <config file>
```

file config ở `--config` có thể xem ở `configs/faster_rcnn`

Ví dụ:

```
python train.py --train_dir '/train' \
              --name 'VNTSDB' \
              --json_dir '/train.json' \
              --iter 5000 \
              --batch 256 \
              --lr 0.001 \
              --resume 1 \
              --config 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'
```

### 3.2. Chạy test mAP

Để thử nghiệm model trên tập test, ta thực hiện như sau:

```
python test_map.py --test_dir <thư mục test> --json_dir <thư mục annotate> --config <thư mục config> --weight <thư mục weight>
```

### 3.3. Chạy ra detection

> Tải model đã train sẵn tại đây: (Bỏ vào thư mục `model/detect`)
> - Faster R-CNN ResNet50: https://drive.google.com/file/d/1E-NkWzld6D9BHNMhZPe9WoAkP9hGonoX/view?usp=sharing
> - Faster R-CNN ResNet101: https://drive.google.com/file/d/10b1QV9-wiPEdCMfn3e83jKJubpv4Tv-M/view?usp=sharing
> - Yolov4-tiny: updating...
> - Yolov4 custom: updating...

Chúng tôi lưu các detection vào các file `.npy` mặc định được lưu ở folder `detect`.

Chạy detection bằng cách:
```
python detect.py --test_dir <thư mục ảnh test> \
                  --weight <file model> \
                  --config <config> \
```
Ví dụ:

```
python detect.py --test_dir 'val' \
                  --weight 'model/model_final.pth' \
                  --config 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml' \
```
### 3.4. Phân loại lại bằng model classifier

> Tải model đã train sẵn tại đây: (Bỏ vào thư mục `model/detect`)
> - LR: https://drive.google.com/file/d/1O8zbwcGQhc8z8_Xge3Ma1N1Gcn50lyQg/view?usp=sharing
> - SVM: https://drive.google.com/file/d/1-38g-y_SPlwXJp_dvuibMJsqaNj3Af0f/view?usp=sharing
> - XGB: https://drive.google.com/file/d/1-2pf6xDSy0LVG1OCEE1IpxoJT8KUqByt/view?usp=sharing

Thư mục `detect` chỉ chứa các file npy chứa bbox của các đối tượng biển báo. Để phân loại biển báo đó là biển báo gì chúng ta cần đưa qua 1 model classifer. Ở đây chúng tôi đã huấn luyện 3 model: `SVM`, `XGBoost`, `Logistic Regression`.

```
python classify_bbox.py --test_dir <thư mục ảnh test> \
                         --model <tên model>
```
Tên model ở flag `--model`: `svm`, `xgb`, `lr`.

### 3.5. Visualize kết quả

Để visualize kết quả chúng ta chạy lệnh:

```
python visualize.py --test_dir <thư mục ảnh test> \
                     --json <thư mục json sau khi chạy bước 3.4. Mặc định là result/output.json> \
                     --outdir 'thư mục xuất ảnh'
```

### 3.6 Demo trên 1 ảnh hoặc 1 video

```
updating...
```

Ví dụ một số kết quả:

![](https://i.imgur.com/l4eTvYT.jpg)

![](https://i.imgur.com/16nAD76.jpg)

![](https://i.imgur.com/xX58uUa.jpg)

## 4. Một số kết quả thử nghiệm

Chúng tôi đã tiến hành huấn luyện và thu được các kết quả:

### 4.1. Phát hiện vị trí biển báo

Chúng tôi tiến hành chạy 4 phương pháp Yolov4-tiny, Yolov4 custom, Faster R-CNN backbone Resnet50, Faster R-CNN backbone Resnet101 để thử nghiệm kết quả.

![](https://i.imgur.com/d8mw7Id.png)


### 4.2. Phân loại biển báo

Dữ liệu biển báo chia thành 7 class:
- Cấm vào
- Cấm rẽ
- Cấm đỗ
- Giới hạn tốc độ
- Các biển báo còn lại
- Nguy hiểm
- Hiệu lệnh

![](https://i.imgur.com/oak42DK.png)

## 5. Colab notebook cho bộ phân lớp

Để huấn luyện 3 bộ phân loại `SVM`, `XGBoost`, `Logistic Regression` chúng tôi sử dụng link colab dưới đây
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_T01bgLd2e3qwxL3Jg-ltWFx0-6Ocxpe)
