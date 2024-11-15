FROM eclipse-temurin:17
COPY --from=python:3.12.4-slim / /

ENV CONTAINER=true

RUN mkdir /experiment
WORKDIR /experiment
#
#RUN apt-get update && apt-get install -y \
#    python3 python3-venv python3-pip python3-wheel \
#    && apt-get clean && \
#    rm -rf /var/lib/apt/lists/*

# Copy simulation files
COPY effects effects
COPY gradle gradle
COPY src src
COPY docker docker
COPY *.kts ./
COPY *.properties ./
COPY gradlew* ./
COPY requirements.txt ./
COPY python python

# Debloat build.gradle.kts in order to avoid downloading unnecessary JVM for testing
RUN sed -i '/alias(libs.plugins.gitSemVer)/d' build.gradle.kts
RUN sed -i '/alias(libs.plugins.multiJvmTesting)/d' build.gradle.kts
RUN sed -i '/multiJvm {/,/}/d' build.gradle.kts

RUN pip3 install -r requirements.txt

RUN python3 python/setup.py sdist bdist_wheel
RUN pip3 install -e python
#RUN CI=true ./gradlew runAllGraphic --info
#RUN rm -rf data
#RUN ./gradlew --stop

CMD ./gradlew runAllBatch
