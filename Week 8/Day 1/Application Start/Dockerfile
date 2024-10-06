FROM python:3.9
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app
COPY --chown=user . $HOME/app
COPY ./requirements.txt ~/app/requirements.txt
RUN pip install -r requirements.txt
COPY . .
RUN mkdir -p $HOME/app/VECTOR_STORE_CACHE
RUN mkdir -p $HOME/app/E2E_CACHE
EXPOSE 7860
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]