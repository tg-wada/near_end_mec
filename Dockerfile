# ベースイメージの設定
FROM pytorch/pytorch:2.0.0

# タイムゾーンの設定
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 必要なパッケージのインストール
RUN apt update && \
    apt install -y wget curl && \
    apt install --no-install-recommends -y \
    git make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
    jq libsm6 libxext6 libgl1 gfortran libatlas-base-dev liblapacke-dev vim

# Pythonのセットアップ
RUN apt install --no-install-recommends -y python3 python3-pip python3-setuptools
RUN pip3 install --upgrade pip

# OpenCVのインストール
RUN apt install --no-install-recommends -y g++ gcc python3-dev
RUN pip3 install opencv-python==4.6.0.66

# 作業ディレクトリの設定
ARG WORKDIR="/app"
ENV WORKDIR=${WORKDIR}
WORKDIR ${WORKDIR}

# 依存関係のインストール
#COPY requirements.txt .
#RUN pip3 install --no-cache-dir -r requirements.txt
#RUN pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
RUN pip install torch=2.0.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torchvision=0.15.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install tqdm==4.66.2 Pillow==10.2.0 openpyxl==3.1.2 filterpy==1.4.5 scikit-image==0.21.0 scikit-learn==1.4.0 xlsxwriter==3.1.9 einops==0.7.0 kornia==0.7.1 imutils==0.5.4 joblib==1.2.0 numpy==1.24.4 pyyaml==6.0.1 mmcv-full==1.6.2

# mm設定
WORKDIR /qpp/space
RUN python -m pip install -U openmim
RUN mim install mmengine

# scipyのインストール
RUN pip install scipy

# mmdetectionのセットアップ
RUN git clone -b v2.28.2 https://github.com/open-mmlab/mmdetection.git
WORKDIR /qpp/space/mmdetection
RUN pip install --no-cache-dir -e . && pip install pre-commit && pre-commit install

# mmtrackingのセットアップ
WORKDIR /qpp/space
RUN git clone -b v0.14.0 https://github.com/open-mmlab/mmtracking.git
WORKDIR /qpp/space/mmtracking
RUN apt install build-essential libatlas-base-dev gfortran
RUN pip install -r requirements/build.txt
RUN pip cache remove scipy
#RUN pip install numpy==1.22  matplotlib
RUN pip install matplotlib
RUN pip install -e .

WORKDIR /app
