FROM ubuntu:18.04

RUN apt-get update && apt-get install -y python3-numpy python3-scipy python3-pip build-essential git axel wget
RUN wget https://aka.ms/downloadazcopy-v10-linux && mv downloadazcopy-v10-linux azcopy.tgz && tar xzf azcopy.tgz --transform 's!^[^/]\+\($\|/\)!azcopy_folder\1!' 
RUN cp azcopy_folder/azcopy /usr/bin

WORKDIR /home/app
RUN pip3 install -U pip
RUN pip3 config set global.index-url http://pip.baidu.com/root/baidu/+simple/
RUN pip3 config set global.index http://pip.baidu.com/root/baidu/
RUN pip3 config set global.trusted-host pip.baidu.com
COPY requirements.txt run_algorithm.py ./
RUN pip3 install -r requirements.txt --default-timeout=1200

ENTRYPOINT ["python3", "-u", "run_algorithm.py"]
