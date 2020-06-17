FROM python:3.7-slim

ADD . /app

RUN set -ex \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
	&& apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false \
	&& rm -rf /var/lib/apt/lists/* \
  && curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python \
  && cd /app \
  && /root/.poetry/bin/poetry config virtualenvs.create false \
  && /root/.poetry/bin/poetry install

CMD ["jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
