## Download data
RGB: https://drive.google.com/drive/folders/147TOz5VUnCGRlDeLfwkzquEfuni7XktX?usp=sharing
Optical flow: https://drive.google.com/drive/folders/1RnkFjzQR6wX2tePrbCa5CZNVX1RtdZu1?usp=sharing

Giải nén các file vừa tải trong thư mục này.
## Cài đặt thư viện
```pip install -r requirements.txt```

## HUấn luyện và đánh giá
Huấn luyện luồng rgb:
```python train_rgb.py```

Huấn luyện luồng optical flow:
```python train_flow.py```

Huấn luyện mô hình MARS-Mean:
```python train_MARS_mean.py```

Huấn luyện mô hình MARS-Conv6a:
```python train_MARS_conv6a.py```

HUấn luyện mô hình MARS-Avg-Pooling:
```python train_MARS_avgpooling.py```

Huấn luyện các tham số w1 và w2 trong phương pháp kết hợp có trọng số:
```python train_weighting_combine.py```

Đánh giá kết hợp luồng RGB và optical flow:
```python rgb_flow_combine.py```

Đánh giá kết hợp luồng RGB tăng cường (MARS-Conv 6a) và RGB:
```python MARS_rgb_combine.py```

Đánh giá kết hợp luồng RGB tăng cường (MARS-Conv 6a) và optical flow:
```python MARS_flow_combine.py```