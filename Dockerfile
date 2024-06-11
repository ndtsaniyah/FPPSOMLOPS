FROM python:3.8.19-slim-bookworm

# disable suid and guid
# install build-essential and git
RUN find /usr/bin \( -perm /4000 -o -perm /2000 \) -type f -exec chmod a-s {} + || true && \
    apt-get update -y && \
    apt-get install --no-install-recommends build-essential git -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*
    
RUN groupadd -r appgroup && useradd -r -g appgroup -d /fpmlops -s /sbin/nologin appuser


WORKDIR /fpmlops
RUN chown appuser:appgroup /fpmlops




# Copy requirements.txt
COPY requirements.txt ./

# install uwsgi and the requirement
# upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip==24.0 && \
    python -m pip install --no-cache-dir uwsgi==2.0.26 && \
    python -m pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_sm

# copy all file
# change owner and group
# change permission
COPY . .
RUN chown -R appuser:appgroup /fpmlops && \
    chmod 755 entrypoint.sh

# expose port 5000
EXPOSE 5000

# set user to nobody
USER appuser

# set entrypoint
ENTRYPOINT [ "./entrypoint.sh" ]
