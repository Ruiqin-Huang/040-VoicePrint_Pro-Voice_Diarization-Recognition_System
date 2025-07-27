from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch

model_name_or_path = '../pretrained_models/iic/nlp_seqgpt-560m'
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
tokenizer.padding_side = 'left'
tokenizer.truncation_side = 'left'

if torch.cuda.is_available():
    model = model.half().cuda()
model.eval()
GEN_TOK = '[GEN]'

while True:
    sent = input('输入/Input: ').strip()
    task = input('分类/classify press 1, 抽取/extract press 2: ').strip()
    labels = input('标签集/Label-Set (e.g, labelA,LabelB,LabelC): ').strip().replace(',', '，')
    task = '分类' if task == '1' else '抽取'

    # Changing the instruction can harm the performance
    p = '输入: {}\n{}: {}\n输出: {}'.format(sent, task, labels, GEN_TOK)
    input_ids = tokenizer(p, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    input_ids = input_ids.to(model.device)
    outputs = model.generate(**input_ids, num_beams=4, do_sample=False, max_new_tokens=256)
    input_ids = input_ids.get('input_ids', input_ids)
    outputs = outputs[0][len(input_ids[0]):]
    response = tokenizer.decode(outputs, skip_special_tokens=True)
    print('BOT: ========== \n{}'.format(response))
    
# from modelscope.utils.constant import Tasks
# from modelscope.pipelines import pipeline
   
# # task可选值为 抽取、分类。text为需要分析的文本。labels为类型列表，中文逗号分隔。
# inputs = {'task': '抽取', 'text': '杭州欢迎你。', 'labels': '地名'}
# # PROMPT_TEMPLATE保持不变
# PROMPT_TEMPLATE = '输入: {text}\n{task}: {labels}\n输出: '
# prompt = PROMPT_TEMPLATE.format(**inputs)
# pipeline_ins = pipeline(task=Tasks.text_generation, model='iic/nlp_seqgpt-560m', model_revision='master', run_kwargs={'gen_token': '[GEN]'})
# print(pipeline_ins(prompt))