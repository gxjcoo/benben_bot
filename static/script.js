// 监听按键事件，当按下回车时触发提交
function handleKeyDown(event) {
    if (event.key === "Enter") {
        // 如果按下回车键，触发提问
        askQuestion();
    }
}

// 提交问题的函数
function askQuestion() {
    const question = document.getElementById('question').value;
    if (question.trim() === "") {
        alert("请输入您的问题！");
        return;
    }

    // 显示正在加载的状态
    document.getElementById('loading').style.display = 'block';
    document.getElementById('answer').style.display = 'none';

    // 发送请求到后端获取答案
    fetch(`/ask?question=${encodeURIComponent(question)}`)
        .then(response => response.json())
        .then(data => {
            // 隐藏加载状态，显示答案
            document.getElementById('loading').style.display = 'none';
            document.getElementById('answer').style.display = 'block';
            document.getElementById('answer').textContent = data.answer;
        })
        .catch(error => {
            // 处理错误
            document.getElementById('loading').style.display = 'none';
            alert("发生了错误，请稍后再试！");
        });
}
