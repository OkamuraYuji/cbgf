import json
import os
from flask import Flask, request
from g4f.client import Client

class AIAssistant:
    CONFIG_FILE = "config.json"
    MAX_HISTORY = 10  
    MODELS = ["gemini-1.5-pro", "gemini-2.0-flash"]

    def __init__(self):
        self.client = Client()
        self.messages = []
        self.instruction = self._load_config()

    @staticmethod
    def _load_config() -> str:
        """Load config từ file, nếu lỗi thì trả về chuỗi rỗng."""
        if os.path.isfile(AIAssistant.CONFIG_FILE):
            try:
                with open(AIAssistant.CONFIG_FILE, "r", encoding="utf-8") as file:
                    return json.load(file).get("instruction", "")
            except (json.JSONDecodeError, IOError):
                pass
        return ""  # Mặc định không có hướng dẫn nếu file lỗi hoặc không tồn tại

    def _trim_history(self):
        """Giới hạn lịch sử tin nhắn để tối ưu bộ nhớ."""
        self.messages = self.messages[-self.MAX_HISTORY:]  
        if not self.messages or self.messages[0]["role"] != "system":
            self.messages.insert(0, {"role": "system", "content": self.instruction})

    def _call_model(self, model: str) -> str:
        """Gửi tin nhắn đến AI model và xử lý phản hồi."""
        try:
            response = self.client.chat.completions.create(
                model=model, messages=self.messages, web_search=False
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"error: {e}"

    def chat(self, user_message: str) -> str:
        """Gửi tin nhắn đến AI và lấy phản hồi với fallback model."""
        self.messages.append({"role": "user", "content": user_message})
        self._trim_history()

        for model in self.MODELS:
            bot_reply = self._call_model(model)
            if not bot_reply.startswith("error:"):
                self.messages.append({"role": "assistant", "content": bot_reply})
                self._trim_history()
                return bot_reply

        return "Xin lỗi, hệ thống đang gặp lỗi. Vui lòng thử lại sau!"

# Khởi tạo Flask app
app = Flask(__name__)

@app.route('/chat', methods=['GET', 'POST'])
def chat_endpoint():
    """API chat hỗ trợ cả GET và POST."""
    try:
        message = request.args.get("message", "").strip() if request.method == "GET" else request.json.get("message", "").strip()

        if not message:
            response = {"error": "Vui lòng cung cấp tin nhắn"}
            return json.dumps(response, ensure_ascii=False), 400, {'Content-Type': 'application/json; charset=utf-8'}
        
        assistant = AIAssistant()
        response = {"message": assistant.chat(message)}
        return json.dumps(response, ensure_ascii=False), 200, {'Content-Type': 'application/json; charset=utf-8'}

    except Exception as e:
        response = {"error": f"Lỗi hệ thống: {e}"}
        return json.dumps(response, ensure_ascii=False), 500, {'Content-Type': 'application/json; charset=utf-8'}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
