FROM python:3.11

 

WORKDIR /usr/src/app 

 

COPY requirements.txt ./ 

RUN apt-get update && \
    apt-get install -y portaudio19-dev libportaudio2 libasound-dev build-essential && \
    pip install --no-cache-dir -r requirements.txt


 

COPY . . 

 

CMD [ "python", "./app.py" ] 