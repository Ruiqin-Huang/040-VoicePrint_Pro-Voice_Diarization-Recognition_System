from app.config.settings import settings
import httpx

# 懒加载实例
_local_hf_pipeline = None

def get_local_hf_pipeline():
    """加载本地 Hugging Face 模型目录并创建 pipeline"""
    global _local_hf_pipeline
    if _local_hf_pipeline is None:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

        print(f"Loading Hugging Face model from: {settings.llm_hf_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(settings.llm_hf_path)
        model = AutoModelForCausalLM.from_pretrained(
            settings.llm_hf_path,
            device_map=settings.llm_device,
            torch_dtype="auto"
        )
        
        _local_hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )
    return _local_hf_pipeline

async def generate_text(system_prompt: str, user_prompt: str) -> str:
    """
    统一的文本生成函数，根据配置自动选择调用方式
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    if settings.llm_mode == 'ollama_api':
        # 方式一：调用 Ollama API
        if not settings.llm_api_endpoint or not settings.llm_model_name:
            raise ValueError("LLM_API_ENDPOINT and LLM_MODEL_NAME must be configured for 'ollama_api' mode.")
        
        async with httpx.AsyncClient() as client:
            # 对于Ollama的 /api/generate, 通常将system和user prompt合并
            full_prompt = f"System: {system_prompt}\nUser: {user_prompt}"
            
            payload = {
                "model": settings.llm_model_name,
                "prompt": full_prompt,
                "system": system_prompt, # 也可以单独传递system prompt
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.95,
                    "num_predict": 1024
                }
            }
            response = await client.post(settings.llm_api_endpoint, json=payload, timeout=120.0)
            response.raise_for_status()
            # Ollama /api/generate 返回的响应在 'response' 字段
            return response.json().get("response", "")

    elif settings.llm_mode == 'local_hf':
        # 方式二：调用本地 Hugging Face 模型
        if not settings.llm_hf_path:
            raise ValueError("LLM_HF_PATH is not configured for 'local_hf' mode.")
            
        hf_pipeline = get_local_hf_pipeline()
        
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
        raise ValueError(f"Unsupported LLM_MODE: {settings.llm_mode}")