from app.config.settings import settings
import httpx
from app.models.common import ModelInfo
import asyncio
from typing import Optional

# 懒加载实例
_local_hf_pipeline = None
_loaded_hf_path = None

# HTTP连接池（全局复用）
_http_client: Optional[httpx.AsyncClient] = None

def get_http_client() -> httpx.AsyncClient:
    """获取全局HTTP客户端连接池"""
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(180.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
            follow_redirects=True
        )
    return _http_client

async def close_http_client():
    """关闭HTTP客户端连接池"""
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None

def get_local_hf_pipeline(model_dir: str):
    """加载本地 Hugging Face 模型目录并创建 pipeline"""
    global _local_hf_pipeline, _loaded_hf_path
    # 如果模型路径已变或模型未加载，则重新加载
    if _loaded_hf_path != model_dir or _local_hf_pipeline is None:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        print(f"Loading Hugging Face model from: {model_dir}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map=settings.llm_device,
            torch_dtype="auto"
        )
        
        _local_hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )
        _loaded_hf_path = model_dir
    return _local_hf_pipeline

async def generate_text(system_prompt: str, user_prompt: str, model_info: ModelInfo) -> str:
    """
    统一的文本生成函数，根据传入的 model_info 动态选择调用方式
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    if model_info.model_call_type == 'ollama_api':
        # 方式一：调用 Ollama API
        if not model_info.api_address or not model_info.model_name:
            raise ValueError("LLM_API_ENDPOINT and LLM_MODEL_NAME must be configured for 'ollama_api' mode.")
        
        client = get_http_client()
        # 对于Ollama的 /api/generate, 通常将system和user prompt合并
        full_prompt = f"System: {system_prompt}\nUser: {user_prompt}"
        
        payload = {
            "model": model_info.model_name,
            "prompt": full_prompt,
            "system": system_prompt, # 也可以单独传递system prompt
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.95,
                "num_predict": 512
            }
        }
        
        # 添加重试机制
        max_retries = 2
        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                response = await asyncio.wait_for(
                    client.post(model_info.api_address, json=payload),
                    timeout=settings.ENTITY_EXTRACTION_TIMEOUT
                )
                response.raise_for_status()
                # Ollama /api/generate 返回的响应在 'response' 字段
                return response.json().get("response", "")
            except asyncio.TimeoutError:
                last_exception = TimeoutError(f"LLM API调用超时（超过{settings.ENTITY_EXTRACTION_TIMEOUT}秒）")
                if attempt < max_retries:
                    await asyncio.sleep(1)  # 等待1秒后重试
                    continue
                raise last_exception
            except httpx.HTTPStatusError as e:
                last_exception = e
                if e.response.status_code >= 500 and attempt < max_retries:  # 服务器错误才重试
                    await asyncio.sleep(1)
                    continue
                raise
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    await asyncio.sleep(1)
                    continue
                raise
        
        if last_exception:
            raise last_exception

    elif model_info.model_call_type == 'vllm':
        # 方式三：调用 vLLM API
        if not model_info.api_address or not model_info.model_name:
            raise ValueError("vLLM_API_ENDPOINT and vLLM_MODEL_NAME must be configured for 'vllm' mode.")
        
        client = get_http_client()
        
        payload = {
            "model": model_info.model_name,
            "messages": messages,
            "temperature": 0.1,
            "top_p": 0.95,
            "max_tokens": 512,
            "stream": False
        }
        
        # 添加重试机制
        max_retries = 2
        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                response = await asyncio.wait_for(
                    client.post(model_info.api_address, json=payload),
                    timeout=settings.ENTITY_EXTRACTION_TIMEOUT
                )
                response.raise_for_status()
                # vLLM OpenAI兼容API返回的响应在 'choices[0].message.content'
                result = response.json()
                return result["choices"][0]["message"]["content"]
            except asyncio.TimeoutError:
                last_exception = TimeoutError(f"vLLM API调用超时（超过{settings.ENTITY_EXTRACTION_TIMEOUT}秒）")
                if attempt < max_retries:
                    await asyncio.sleep(1)  # 等待1秒后重试
                    continue
                raise last_exception
            except httpx.HTTPStatusError as e:
                last_exception = e
                if e.response.status_code >= 500 and attempt < max_retries:  # 服务器错误才重试
                    await asyncio.sleep(1)
                    continue
                raise
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    await asyncio.sleep(1)
                    continue
                raise
        
        if last_exception:
            raise last_exception

    elif model_info.model_call_type == 'local_hf':
        # 方式二：调用本地 Hugging Face 模型
        if not model_info.model_dir:
            raise ValueError("model_dir is not configured for 'local_hf' mode.")
            
        hf_pipeline = get_local_hf_pipeline(model_info.model_dir)
        
        prompt = hf_pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        outputs = hf_pipeline(
            prompt,
            max_new_tokens=1024, # 为JSON输出设置一个合理的长度
            do_sample=False, # 对于抽取任务，关闭采样以获得更稳定的结果
            temperature=0.1, # 低温确保结果的确定性
            top_p=0.95
        )
        
        full_text = outputs[0]['generated_text']
        response_text = full_text[len(prompt):]
        return response_text
    
    else:
        raise ValueError(f"Unsupported LLM_MODE: {model_info.model_call_type}")