FROM python:3.8

COPY . /app

WORKDIR /app/src

RUN pip install -r ../requirements.txt

#Exposing the default streamlit port
EXPOSE 8501

#Running the streamlit app
ENTRYPOINT ["streamlit", "run", "--server.maxUploadSize=5"]
CMD ["app.py"]