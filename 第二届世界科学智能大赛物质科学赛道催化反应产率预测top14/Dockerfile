# FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:2.0.0-py3.9.12-cuda11.8.0-u22.04
FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:1.12-py3.9.12-cuda11.3.1-u20.04


# 如有安装其他软件的需求
#RUN apt-get update && apt-get install curl
#RUN apt-get update &&  apt-get install libxrender1 -y
#RUN sed -i 's#http://archive.ubuntu.com/#http://mirrors.tuna.tsinghua.edu.cn/#' /etc/apt/sources.list;
#RUN apt-get update --fix-missing && apt-get install -y libxrender1 --fix-missing
# 如果安装其他python包的情况
RUN pip install numpy --index-url=http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
RUN pip install lightgbm --index-url=http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
RUN pip install pandas --index-url=http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
RUN pip install rdkit==2024.03.3 --index-url=http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
RUN pip install scikit_learn==1.4.2 --index-url=http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
RUN pip install tqdm --index-url=http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
RUN pip install xgboost --index-url=http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
RUN pip install catboost --index-url=http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
RUN pip install opencv-python --index-url=http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com


# 复制代码到镜像仓库
COPY app /app

# 指定工作目录
WORKDIR /app

# 容器启动运行命令
CMD ["bash", "run.sh"]
