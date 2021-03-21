FROM nvcr.io/nvidia/pytorch:20.11-py3


WORKDIR /src
COPY Pipfile.docker Pipfile

RUN pip install pipenv
RUN pipenv lock -r > requirements.txt
RUN pip install -r requirements.txt

COPY . ./
CMD python -u main.py --verbose
