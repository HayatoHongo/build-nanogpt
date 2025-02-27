# ベースイメージはUbuntu（例として20.04）
FROM ubuntu:20.04

# 非対話モードの設定
ENV DEBIAN_FRONTEND=noninteractive

# システムパッケージの更新とインストール
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを設定
WORKDIR /app

# プロジェクトファイル一式をコンテナにコピー
COPY . /app

# pipをアップグレード
RUN python3 -m pip install --upgrade pip

# 必要なPythonライブラリをインストール
# タイムアウト: 3000秒
RUN pip3 install --default-timeout=3000 numpy tiktoken datasets tqdm requests torch transformers matplotlib

# コンテナ起動時のデフォルトコマンド（例：fineweb.pyを実行）
CMD ["python3", "fineweb.py"]
