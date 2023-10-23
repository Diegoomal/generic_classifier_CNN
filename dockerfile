FROM tensorflow/tensorflow:latest-gpu

# Needed for openGL(open-cv)
RUN apt-get update && apt-get install -y mesa-utils libgl1-mesa-glx

# clone dataset from github
RUN apt-get update && apt-get install -y git

# RUN git clone https://github.com/<user_github>/<name_repo_dataset>.git
# RUN mv <name_repo_dataset> dataset

# RUN git clone https://github.com/Diegoomal/ebeer_dataset.git
# RUN mv ebeer_dataset dataset

RUN git clone https://github.com/Diegoomal/dog_dataset.git
RUN mv dog_dataset dataset

RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt
WORKDIR /app
COPY app/ .
CMD [ "python", "main.py" ]