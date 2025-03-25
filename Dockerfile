FROM sdcpp.azurecr.io/slimdiffusers:v3 AS runtime

COPY requirements.txt /home

WORKDIR /home

RUN pip install -r requirements.txt

COPY hdapp.py /home

COPY pipelines /home/pipelines

ENTRYPOINT ["python", "hdapp.py"]