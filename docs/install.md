# Step-by-step installation instructions


## Installation
**a.** Create conda environment.
```
conda create -n vlm python=3.8 -y
conda activate vlm
```

**b.** Install [pytorch](https://pytorch.org/)(v2.0.1).
```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

**c.** Install yolov10
```
pip install ultralytics
pip install -q git+https://github.com/THU-MIG/yolov10.git
```

**d.** Install requirements.
```
pip install -r requirements.txt
```
