COPY tasks_distribute.py /app/tasks_distribute.pyFROM python:3.10 as builder

WORKDIR /app
#RUN pip install --upgrade pip
#安装需要的库
COPY ./requirements.txt /app/
RUN pip install --upgrade pip && \
        pip install -r requirements.txt -i https://pypi.org/simple
#RUN apt-get update
#RUN apt-get install ffmpeg libsm6 libxext6  -y


# 两阶段构建 利用缓存
FROM python:3.10
WORKDIR /app
#RUN apt-get update
#RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

# 将当前项目目录下的所有文件添加到容器的 /app 目录
COPY tasks_distribute.py /app/tasks_distribute.py

EXPOSE 9450
#CMD ["python","input.py"]
CMD ["python","tasks_distribute.py"]
