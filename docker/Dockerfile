FROM python:3.9.17
COPY . /digits/
WORKDIR /digits
RUN pip install -r requirements.txt
RUN chmod +x entrypoint.sh
# VOLUME /digits/models
EXPOSE 8000
CMD ["./entrypoint.sh"]

# export variable
# flask run
# flask --app main run --host=1.1.1.1:5000