from app.config.settings import settings

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
    使用本地 Hugging Face 模型生成文本
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    if settings.llm_mode == 'local_hf':
        if not settings.llm_hf_path:
            raise ValueError("LLM_HF_PATH is not configured in .env file.")
            
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
        # 理论上不会执行到这里，因为 config 已限定模式
        raise ValueError(f"Unsupported LLM_MODE: {settings.llm_mode}")