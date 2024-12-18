# dialoggpt_service.py

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import config 
from baidu_translate_service import BaiduTranslateService  # 导入 DialoGPT 服务
translator = BaiduTranslateService()
class DialoGPTService:
    def __init__(self):
        
        # 加载 DialoGPT 模型和 tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(config.DIALOGGPT_MODEL_NAME)
        self.chat_model = GPT2LMHeadModel.from_pretrained(config.DIALOGGPT_MODEL_NAME)

    def generate_response(self, prompt):
        """根据用户输入生成回复"""
        # 将输入文本编码为 token
        new_user_input_ids = self.tokenizer.encode(prompt + self.tokenizer.eos_token, return_tensors="pt")

        # 生成模型输出
        bot_output = self.chat_model.generate(
            new_user_input_ids,
            max_length=1000,
            pad_token_id=self.tokenizer.eos_token_id,
            top_k=10,
            top_p=0.7,
            temperature=0.9,
            no_repeat_ngram_size=2,
            do_sample=True,
            num_return_sequences=1
        )

        # 解码生成的回复
        bot_response = self.tokenizer.decode(bot_output[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)
        return translator.translate(bot_response) 
