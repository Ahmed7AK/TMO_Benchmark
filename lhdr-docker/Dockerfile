FROM ubuntu:24.04
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository universe && \
    apt-get update && \
    apt-get install -y luminance-hdr=2.6.1.1+dfsg-3build3 && \
    rm -rf /var/lib/apt/lists/*
ENTRYPOINT ["luminance-hdr-cli"]