import requests
from llamaapi import LlamaAPI
import os
import json
import qianfan
from zhipuai import ZhipuAI
from openai import OpenAI
from ollama import chat
from ollama import ChatResponse
import http.client

# Load API keys from environment variables or config file
# Example: export QIANFAN_ACCESS_KEY="your-access-key"
os.environ["QIANFAN_ACCESS_KEY"] = os.getenv("QIANFAN_ACCESS_KEY", "")
os.environ["QIANFAN_SECRET_KEY"] = os.getenv("QIANFAN_SECRET_KEY", "")

def moonshot_api(input):
    client = OpenAI(
        api_key = os.getenv("MOONSHOT_API_KEY", ""),
        base_url = "https://api.moonshot.cn/v1",
    )

    completion = client.chat.completions.create(
        model = "moonshot-v1-8k",
        messages=[{"role": "user", "content": f"{input}"}],
        temperature = 0.3,
    )

    return completion.choices[0].message.content

# ZhipuAI API
def zhipuai_api(input, model):
    client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY", ""))
    response = client.chat.completions.create(
        model=model, 
        messages=[{"role": "user", "content": f"{input}"}],
    )
    return response.choices[0].message.content

# DashScope (Qwen) API
def qwen_api(input, model):
    url = 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {os.getenv("DASHSCOPE_API_KEY", "")}'
    }
    body = {
        'model': model,
        "input": {
            "messages": [{"role": "user", "content": f"{input}"}]
        }
    }
    response = requests.post(url, headers=headers, json=body)
    return response.json()['output']['text']

# Qianfan API (Baidu)
def qianfan_api(input, model):
    chat_comp = qianfan.ChatCompletion()
    response = chat_comp.do(model=model, messages=[{
        "role": "user",
        "content": f"{input}"
    }])
    return response["body"]["result"]

# LlamaAPI (Llama or Gemma models)
def llama_api(input, model):
    llama = LlamaAPI(os.getenv("LLAMA_API_KEY", ""))
    api_request_json = {
        "model": model,
        "messages": [{"role": "user", "content": f"{input}"}],
        "stream": False
    }
    response = llama.run(api_request_json)
    return response.json()['choices'][0]['message']['content']

def deepseek_api(input):
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY", ""), 
        base_url="https://api.deepseek.com"
    )

    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": f"{input}"}],
        stream=False
    )

    return response.choices[0].message.content

def ollama_api(input, model):
    response: ChatResponse = chat(model=model, messages=[
        {"role": "user", "content": f"{input}"},
    ])

    return response['message']['content']

def minimax_api(input):
    api_key = os.getenv("MINIMAX_API_KEY", "")

    url = f"https://api.minimax.chat/v1/text/chatcompletion_v2"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "MiniMax-Text-01",
        "messages": [
            {
                "role": "user",
                "content": f"{input}"
            }
        ]
    }

    response = requests.post(url, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]

def smallai_api(input, model):
    conn = http.client.HTTPSConnection("ai98.vip")
    payload = json.dumps({
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": f"{input}"
            }
        ]
    })
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {os.getenv("SMALLAI_API_KEY", "")}',
        'Content-Type': 'application/json'
    }
    conn.request("POST", "/v1/chat/completions", payload, headers)
    res = conn.getresponse()
    data = res.read()
    json_data = json.loads(data.decode("utf-8"))
    return json_data['choices'][0]['message']['content']

# Unified API Interface
def LLMAPI(input, model):
    """
    Unified interface for calling various LLM APIs based on model name.

    :param input: User input content
    :param model: Model name
    :return: Model response
    
    Supported models:
    - ZhipuAI: glm-4-plus, glm-4-air
    - Qwen: qwen-max, qwen-turbo
    - Baidu: ERNIE-4.0-Turbo-8K-Latest, ERNIE-Speed-128K
    - LlamaAPI: llama3.1-405b, llama3.1-70b, llama3.1-8b
    - DeepSeek: deepseek-reasoner
    - Moonshot: moonshot-v1-8k
    - Ollama (local): llama3.2, phi4, gemma3, mistral, qwen2.5, deepseek-r1
    - MiniMax: MiniMax-Text-01
    - Others: claude-3-5-sonnet, gemini-2.0-flash-exp, claude-3-opus, gemini-1.5-flash-8b
    """
    if model in ['glm-4-plus', 'glm-4-air']:
        return zhipuai_api(input, model)
    elif model in ['qwen-max', 'qwen-turbo']:
        return qwen_api(input, model)
    elif model in ['ERNIE-4.0-Turbo-8K-Latest', 'ERNIE-Speed-128K']:
        return qianfan_api(input, model)
    elif model in ['llama3.1-405b','llama3.1-70b', 'llama3.1-8b']:
        return llama_api(input, model)
    elif model in ['deepseek-reasoner']: 
        return deepseek_api(input)
    elif model in ['moonshot-v1-8k']:
        return moonshot_api(input)
    elif model in ['llama3.2','phi4','gemma3','mistral','qwen2.5','deepseek-r1']:
        return ollama_api(input, model)
    elif model in ['MiniMax-Text-01']:
        return minimax_api(input)
    elif model in ['claude-3-5-sonnet-20241022','gemini-2.0-flash-exp','claude-3-opus-20240229','gemini-1.5-flash-8b']:
        return smallai_api(input, model)
    else:
        raise ValueError(f"Unsupported model: {model}")

# Configuration helper
def setup_api_keys(config_file="config.json"):
    """
    Load API keys from a configuration file.
    
    Example config.json:
    {
        "MOONSHOT_API_KEY": "your-key",
        "ZHIPUAI_API_KEY": "your-key",
        ...
    }
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            for key, value in config.items():
                os.environ[key] = value
    except FileNotFoundError:
        print(f"Warning: {config_file} not found. Please set environment variables manually.")