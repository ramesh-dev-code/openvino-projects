## Running Domain-Specific Chatbot with RAG on Intel AI PC    
### Device Under Test   
Processor: Intel® Core™ Ultra 7 165H   
iGPU: Intel® Arc™ Graphics (GPU Driver: 24.22.29735.27)   
OS: Ubuntu 22.04 LTS, Kernel: 6.5.0-45-generic   
Python: 3.10.12   
RAM: 64GB
OpenVINO: 2024.2.0   

### Prerequisites   
1. Install GPU driver using the steps 1-3 in this [link](https://dgpu-docs.intel.com/driver/client/overview.html#installing-gpu-packages)
2. Create an account on [Hugging Face](https://huggingface.co/). Navigate to Profile->Settings->Access Tokens and click ‘Create New Token’ to create an access token of type Fine-grained/Read/Write
3. Request access to [Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) model

### Installation   
1. Install Python, Git and dependencies
  ```
  sudo apt-get update
  sudo apt-get upgrade
  sudo apt-get install python3-venv build-essential python3-dev git-all libgl1-mesa-dev ffmpeg    
  ```

2. Create a virtual environment and activate it   
  ```
  python3 -m venv ov_nb_venv
  source ov_nb_venv/bin/activate
  ```
3. Download the OpenVINO Notebooks (2024.2) repository and extract the folder
  ```  
  cd openvino_notebooks-2024.2
  ```
4. Download the requirements (rag_chatbot_requirements.txt) and install the dependencies
  ```
  python -m pip install --upgrade pip
  pip install wheel setuptools
  pip install -r rag_chatbot_requirements.txt --timeout 1000
  ```
