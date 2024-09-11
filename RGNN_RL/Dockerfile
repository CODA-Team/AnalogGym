FROM continuumio/miniconda3

WORKDIR /rl
VOLUME [ "/rl" ]

COPY . /rl
COPY environment.yml /rl/
COPY torch-1.13.1+cpu-cp310-cp310-linux_x86_64.whl /rl/
COPY torch_cluster-1.6.1+pt113cpu-cp310-cp310-linux_x86_64.whl /rl/
COPY torch_scatter-2.1.1+pt113cpu-cp310-cp310-linux_x86_64.whl /rl/
COPY torch_sparse-0.6.17+pt113cpu-cp310-cp310-linux_x86_64.whl /rl/
COPY torch_spline_conv-1.2.2+pt113cpu-cp310-cp310-linux_x86_64.whl /rl/

RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "analoggym-env", "/bin/bash", "-c"]

RUN pip install --no-cache-dir /rl/torch-1.13.1+cpu-cp310-cp310-linux_x86_64.whl \
    /rl/torch_cluster-1.6.1+pt113cpu-cp310-cp310-linux_x86_64.whl \
    /rl/torch_scatter-2.1.1+pt113cpu-cp310-cp310-linux_x86_64.whl \
    /rl/torch_sparse-0.6.17+pt113cpu-cp310-cp310-linux_x86_64.whl \
    /rl/torch_spline_conv-1.2.2+pt113cpu-cp310-cp310-linux_x86_64.whl

CMD ["conda", "run", "-n", "analoggym-env", "python", "main_AMP.py"]




