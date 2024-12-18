import requests
import random
import config 
from hashlib import md5
class BaiduTranslateService:
    def __init__(self):
        """
        初始化百度翻译服务
        :param appid: 百度翻译 API 的 AppID
        :param appkey: 百度翻译 API 的 AppKey
        """
        self.appid =  config.BAIDU_APP_ID
        self.appkey = config.BAIDU_APP_KEY
        self.endpoint = 'http://api.fanyi.baidu.com'
        self.path = '/api/trans/vip/translate'
        self.url = self.endpoint + self.path

    def make_md5(self, s, encoding='utf-8'):
        """
        生成 MD5 签名
        
        :param s: 待签名字符串
        :param encoding: 编码方式，默认为 'utf-8'
        :return: MD5 签名
        """
        return md5(s.encode(encoding)).hexdigest()

    def translate(self, query, from_lang='en', to_lang='zh'):
        """
        翻译文本

        :param query: 要翻译的文本
        :param from_lang: 源语言（默认为英文）
        :param to_lang: 目标语言（默认为中文）
        :return: 翻译后的文本
        """
        # 生成 salt 和 sign
        salt = random.randint(32768, 65536)
        sign = self.make_md5(self.appid + query + str(salt) + self.appkey)

        # 构建请求参数
        payload = {
            'appid': self.appid,
            'q': query,
            'from': from_lang,
            'to': to_lang,
            'salt': salt,
            'sign': sign
        }

        # 发送请求
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        r = requests.post(self.url, params=payload, headers=headers)

        # 解析返回的 JSON 数据
        result = r.json()

        # 判断是否请求成功
        if 'trans_result' in result:
            # 拼接所有翻译结果的 'dst' 字段
            translated_text = " ".join([item['dst'] for item in result['trans_result']])
            return translated_text
        else:
            return "翻译失败: " + str(result.get('error_msg', '未知错误'))

# 示例：使用百度翻译 API 进行翻译
if __name__ == "__main__":
    translator = BaiduTranslateService()
    text_to_translate = "Hello World! This is 1st paragraph.\nThis is 2nd paragraph."
    
    # 翻译文本
    translated_text = translator.translate(text_to_translate)
    print(f"原文: {text_to_translate}")
    print(f"翻译后: {translated_text}")
