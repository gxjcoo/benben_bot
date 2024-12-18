import unittest
from dialoggpt_service import DialoGPTService

class TestDialoGPTService(unittest.TestCase):

    def setUp(self):
        """在每个测试用例前创建 DialoGPTService 实例"""
        self.dialoggpt_service = DialoGPTService()

    def test_generate_response(self):
        """测试生成的回应是否非空"""
        prompt = "你好"
        response = self.dialoggpt_service.generate_response(prompt)
        print(f"Input: {prompt}")
        print(f"Response: {response}")
        
        # 验证返回类型是否为字符串
        self.assertIsInstance(response, str)  
        # 确保回应非空
        self.assertGreater(len(response), 0)

    def test_generate_response_with_long_input(self):
        """测试长输入的回应生成"""
        prompt = "我今天过得很开心，感谢你的关心。你今天怎么样？"
        response = self.dialoggpt_service.generate_response(prompt)
        print(f"Input: {prompt}")
        print(f"Response: {response}")
        
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

if __name__ == '__main__':
    unittest.main()
