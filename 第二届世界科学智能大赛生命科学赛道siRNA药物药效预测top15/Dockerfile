# FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:2.0.0-py3.9.12-cuda11.8.0-u22.04
FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:1.12-py3.9.12-cuda11.3.1-u20.04


# 如有安装其他软件的需求
#RUN apt-get update && apt-get install curl

# 如果安装其他python包的情况
#pip3 install numpy --index-url=http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
run pip install easydict==1.13            --index-url=http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
run pip install matplotlib==3.8.4         --index-url=http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
run pip install numpy==1.24.3             --index-url=http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
run pip install pandas==2.0.0             --index-url=http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
run pip install rich==13.7.1              --index-url=http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
run pip install scikit_learn==1.4.2       --index-url=http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
#run pip install torch==2.2.2              --index-url=http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
run pip install tqdm==4.66.4              --index-url=http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
# 复制代码到镜像仓库
COPY app /app

# 指定工作目录
WORKDIR /app

# 容器启动运行命令
CMD ["bash", "run.sh"]
