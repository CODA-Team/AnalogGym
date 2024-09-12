## Running the Simulation

There are three ways to run the simulation:

1. **Using Docker with Local Setup**  
   The `Dockerfile` and environment configuration file `environment.yml` are included in the project. You can build your own Docker image and container with the following steps:

   - Open a terminal in the `RGNN_RL` directory and build the Docker image (ensure the `.` at the end is included):
     ```bash
     docker build -t analoggym_rgcn .
     ```

   - After the image is built, create a Docker container with the following command. The `-v` flag links the local path (`/xxx/rgcn_rl`) to the container's internal path (`/app`), allowing real-time file modification. The `-it` flag opens an interactive terminal:
     ```bash
     docker run -v /xxx/rgnn_rl:/app -it analoggym_rgcn /bin/bash
     ```

   - Inside the container, check if the `analoggym-env` is activated. If not, activate it with:
     ```bash
     conda activate analoggym-env
     ```

2. **Using Pre-Built Docker Images from Docker Hub**  
   A quicker method is to pull the pre-built Docker image from Docker Hub and run the simulation directly. Use the following command to pull the image:
   ```bash
   docker pull chenzhenxin/analoggym_rgcn:latest
   ```
   After pulling, refer to the steps in Method 1 to create a container and run the simulation.

3. **Running Locally for Precise Simulation Results**  
   Docker might not resolve ngspice version issues, so for precise results, you can run the simulation locally. First, create a conda virtual environment and install the required packages:
   ```bash
   conda install -c conda-forge ngspice
   conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cpuonly -c pytorch
   pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.13.1+cpu.html
   pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.13.1+cpu.html
   pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.13.1+cpu.html
   pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.13.1+cpu.html
   pip install torch-geometric
   pip install gymnasium
   pip install tabulate
   pip install numpy
   pip install matplotlib
   pip install lPython
   ```

   For details on resolving ngspice version issues, refer to the Supplementary Information. Once the environment is set up, you can run:
   ```bash
   python main_AMP.py
   ```
   or
   ```bash
   python main_LDO.py
   ```
   to start the simulation.
```

This version is more concise, polished, and suitable for a README file. It maintains a professional tone while clearly explaining the steps.
