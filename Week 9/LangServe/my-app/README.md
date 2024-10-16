# rag_on_prem

## Installation

Install the LangChain CLI if you haven't yet

```bash
pip install -U langchain-cli
```

## Adding packages

```bash
# adds a predefined template or package from the official LangChain templates repository to your application. (https://github.com/langchain-ai/langchain/tree/master/templates) 
langchain app add rag_on_prem

# adds a package from a custom GitHub repository to your LangChain application. (git+url or $OWNER/$REPO)
langchain app add --repo git+https://github.com/Taekyo-Lee/Cohort_AIE4/tree/main/Week%209/LangServe/my-app

# (Optional) adds a package to your LangChain app but allows you to define a custom API path where this package will be accessible. (defaults to `/{package_name}`)
langchain app add rag_on_prem --api_path=/my/custom/path/rag
```

Note: you remove packages by their api path

```bash
langchain app remove my/custom/path/rag
```

## Run vLLM server with Docker
```bash
docker run --runtime nvidia --gpus all -d --rm  -v "{$YOUR_DIRECTORY}/.cache/huggingface:/root/.cache/huggingface" -p 8000:8000 --ipc=host --env "HUGGING_FACE_HUB_TOKEN={$HF_KEY}" vllm/vllm-openai:latest --model deepseek-ai/deepseek-llm-7b-chat  --max_model_len 640
```

## Setup LangSmith (Optional)
LangSmith will help us trace, monitor and debug LangChain applications. 
You can sign up for LangSmith [here](https://smith.langchain.com/). 
If you don't have access, you can skip this section


```shell
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=<your-api-key>
export LANGCHAIN_PROJECT=<your-project>  # if not specified, defaults to "default"
```

## Launch LangServe

```bash
langchain serve
```

## Running in Docker

This project folder includes a Dockerfile that allows you to easily build and host your LangServe app.

### Building the Image

To build the image, you simply:

```shell
docker build . -t my-langserve-app
```

If you tag your image with something other than `my-langserve-app`,
note it for use in the next step.

### Running the Image Locally

To run the image, you'll need to include any environment variables
necessary for your application.

We also expose port 8080 with the `-p 8080:8080` option.

```shell
docker run -d -e export LANGCHAIN_TRACING_V2="true" -e LANGCHAIN_API_KEY=$LANGCHAIN_API_KEY -e LANGCHAIN_PROJECT=$LANGCHAIN_PROJECT -p 8080:8080 my-langserve-app
```
