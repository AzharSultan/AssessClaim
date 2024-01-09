FROM python:3.10

WORKDIR /opt/app

COPY . .

RUN pip install -r requirements.txt

ENTRYPOINT [ "python3", "assess_claim.py" ]