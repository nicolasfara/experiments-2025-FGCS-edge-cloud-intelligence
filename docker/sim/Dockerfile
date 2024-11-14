FROM eclipse-temurin:17

RUN mkdir /experiment
WORKDIR /experiment

RUN apt-get update && apt-get install -y \
    python3 python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

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
RUN pip install -r requirements.txt --break-system-packages
RUN CI=true ./gradlew runAllGraphic --info
RUN rm -rf data
RUN ./gradlew --stop

CMD ./gradlew runAllBatch
