FROM continuumio/miniconda:4.5.12

RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends --fix-missing \
    build-essential \
    ca-certificates \
    cmake \
    git \
    gcc \
    vim

RUN conda install -y -c bioconda --force nextflow scipy pybedtools pyfaidx pandas numpy bbmap && \
    conda install -y -c anaconda --force networkx click biopython matplotlib scikit-learn seaborn pyamg && \
    conda install -y -c plotly plotly && \
    conda install -y -c conda-forge deap hdbscan multicore-tsne && \
    conda clean -a --yes

RUN pip install polycracker==1.0.3

RUN mkdir -p workdir
ENV HOME /workdir
WORKDIR ${HOME}
COPY . ${HOME}
WORKDIR ${HOME}/polycracker
