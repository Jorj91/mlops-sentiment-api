# use lightweight python image
FROM python:3.10-slim

# set wd inside the container
WORKDIR /app

# copy dependency file
COPY requirements.txt .

# upgrade pip + install dependecies
RUN pip install --upgrade pip && pip install -r requirements.txt

# copy rest of the app code into the container
COPY . /app

# define port used by fastapi app. HF Space expects to listen on port 7860
ENV PORT=7860
EXPOSE 7860

# start fastapi app using Uvicorn
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
