# Base image
FROM python:3.7-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# copy stuff
COPY requirements.txt requirements.txt
COPY final_exercise/ final_exercise/
COPY data/ data/

# workdir
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

# entrypoint
ENTRYPOINT ["python", "-u", "final_exercise/main.py", "evaluate", "model_chkpt.pt"]


