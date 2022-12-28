FROM ubuntu:18.04

COPY . ./

RUN apt-get update && apt-get install -y \
    software-properties-common
RUN add-apt-repository universe
RUN apt-get install -y \
    python3.8 \
    python3-pip


RUN pip3 install --upgrade pip
RUN apt install ffmpeg libsm6 libxext6 libgl1-mesa-dev -y


RUN pip install -r requirements.txt

RUN cd yolov5

RUN pip install -r requirements.txt

RUN cd ..

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]