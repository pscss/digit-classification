FROM python:3.9.17
COPY . /digits/
WORKDIR /digits
RUN pip install -r requirements.txt
RUN chmod +x entrypoint.sh
VOLUME /digits/models
CMD ["./entrypoint.sh"]
