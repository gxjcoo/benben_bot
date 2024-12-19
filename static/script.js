// 监听按键事件，当按下回车时触发提交
function handleKeyDown(event) {
  if (event.key === 'Enter') {
    // 如果按下回车键，触发提问
    askQuestion();
  }
}

// 提交问题的函数
function askQuestion() {
  const label = document.getElementById('label').value;
  if (label.trim() === '') {
    alert('请输入您的问题！');
    return;
  }

  // 显示正在加载的状态
  document.getElementById('loading').style.display = 'block';
  document.getElementById('content').style.display = 'none';

  // 发送请求到后端获取答案
  fetch(`/ask?label=${encodeURIComponent(label)}`)
    .then((response) => response.json())
    .then((data) => {
      // 隐藏加载状态，显示答案
      document.getElementById('loading').style.display = 'none';
      document.getElementById('content').style.display = 'block';
      document.getElementById('content').textContent = data.content;
    })
    .catch((error) => {
      // 处理错误
      document.getElementById('loading').style.display = 'none';
      alert('发生了错误，请稍后再试！');
    });
}
