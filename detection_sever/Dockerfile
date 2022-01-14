FROM anibali/pytorch:1.7.0-cuda11.0-ubuntu20.04
WORKDIR "/app/detection/"
COPY --chown=user:user . $WORKDIR
ENV LD_LIBRARY_PATH=/app/detection/x86_64-linux-gnu:$LD_LIBRARY_PATH
RUN conda install pip
RUN pip install -r requirements.txt -i https://pypi.douban.com/simple/
CMD ["python", "app_detect.py"]
