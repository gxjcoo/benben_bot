import json
from sentence_transformers import SentenceTransformer
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# 加载历史问答数据
def load_faq_data():
    with open('data/resource_data.json', 'r', encoding='utf-8') as f:
        resource_data = json.load(f)
    return resource_data

# 训练模型并保存
def train_model(resource_data):
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    # 提取所有问题并进行编码
    questions = [item['label'] for item in resource_data]
    question_embeddings = model.encode(questions)

    # 保存模型
    with open('models/model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'embeddings': question_embeddings, 'resource_data': resource_data}, f)

    print("模型已保存！")

if __name__ == "__main__":
    resource_data = load_faq_data()
    train_model(resource_data)
