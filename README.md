# Quantized Models Prompt Playgrounds
Using Gradio and CTransformers library<br>
It runs on CPU only 

### Tested Models
- claude2-alpaca-7b.Q4_K_M.gguf
- openchat_3.5-16k.Q4_K_M.gguf
- vicuna-7b-v1.5-16k.Q4_K_M.gguf

<img src="https://github.com/fabiomatricardi/Playgrounds/raw/main/GGUF-Playground.png" width=800>

### Installation
Create a Virtual Environment and activate it

```
pip install transformers
pip install gradio
```

- included in the Repo a requirements.txt

### Instructions
- create a subfloder called `model`
- download in the `model` folder  the mentioned above models from HuggingFace Hub TheBloke repo
  - https://huggingface.co/TheBloke/claude2-alpaca-7B-GGUF
  - https://huggingface.co/TheBloke/vicuna-7B-v1.5-16K-GGUF/tree/main
  - https://huggingface.co/TheBloke/openchat_3.5-16k-GGUF
- run the python files according to the model you like

### Context window details
Every model comes with its own context window
- claude2-alpaca-7b has 4K tokens
- Vicuna-7b has 16K tokens
- Openchat 3.5 has 16K tokens
