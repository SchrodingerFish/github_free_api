# 使用较小的基础镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 更改 apt 源为清华源
RUN echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian bookworm main non-free contrib" > /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian bookworm-updates main non-free contrib" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.tuna.tsinghua.edu.cn/debian-security bookworm-security main" >> /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 设置环境变量
ENV GITHUB_TOKEN=''

# 暴露端口
EXPOSE 8080

# 启动应用
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]
