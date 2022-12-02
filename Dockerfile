FROM ubuntu:20.04

RUN apt-get update

RUN apt-get install -y gcc musl-dev python3-pip libgl1


COPY . ./

RUN cd yolov5

RUN pip install   -r requirements.txt 

RUN cd ..

RUN pip install  -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]