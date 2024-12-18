# app.py

from flask import Flask, request, jsonify, render_template
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from script.dialoggpt_service import DialoGPTService  # 导入 DialoGPT 服务
app = Flask(__name__)

# 实例化 DialoGPT 服务
dialoggpt_service = DialoGPTService()

# 加载训练好的模型和数据
with open('models/model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    question_embeddings = model_data['embeddings']
    faq_data = model_data['faq_data']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['GET'])
def ask_question():
    user_question = request.args.get('question')
    
    if not user_question:
        return jsonify({"error": "没有提供问题"}), 400
    
    # 用户问题编码
    user_question_embedding = model.encode([user_question])
    
    # 计算余弦相似度
    similarities = cosine_similarity(user_question_embedding, question_embeddings)
    
    # 找到最相似的问题
    most_similar_idx = similarities.argmax()
    # 相似度
    similarity_score = similarities[0][most_similar_idx]
    print(similarity_score, 'similarity_score')

    # 如果相似度较高，返回 FAQ 答案
    if similarity_score >= 0.7:  # 相似度阈值可以根据需求调整
        answer = faq_data[most_similar_idx]["answer"]
    else:
        # 否则，调用 DialoGPT 服务生成回答
        answer = dialoggpt_service.generate_response(user_question)

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)  # 运行在5000端口
