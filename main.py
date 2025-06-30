import json
import os
from typing import Dict, Optional
from flask import Flask, request, jsonify
from g4f.client import Client

class AIAssistant:
    CONFIG_FILE: str = "config.json"
    MAX_HISTORY: int = 10
    MODELS: list = ["gemini-1.5-pro", "gemini-2.0-flash"]
    DEFAULT_INSTRUCTION: str = "Bạn là một trợ lý AI hữu ích, trả lời chính xác và ngắn gọn."

    def __init__(self):
        self.client = Client()
        self.messages: list = []
        self.instruction: str = self._load_config()

    def _load_config(self) -> str:
        """Load cấu hình từ file JSON, trả về instruction mặc định nếu lỗi."""
        if not os.path.isfile(self.CONFIG_FILE):
            return self.DEFAULT_INSTRUCTION
        
        try:
            with open(self.CONFIG_FILE, "r", encoding="utf-8") as file:
                config = json.load(file)
                return config.get("instruction", self.DEFAULT_INSTRUCTION)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Lỗi khi đọc config: {e}")
            return self.DEFAULT_INSTRUCTION

    def _trim_history(self) -> None:
        """Giới hạn lịch sử tin nhắn và đảm bảo system instruction ở đầu."""
        if len(self.messages) > self.MAX_HISTORY:
            self.messages = self.messages[-self.MAX_HISTORY:]
        
        if not self.messages or self.messages[0]["role"] != "system":
            self.messages.insert(0, {"role": "system", "content": self.instruction})

    def _call_model(self, model: str) -> Optional[str]:
        """Gửi yêu cầu đến AI model và trả về phản hồi hoặc None nếu lỗi."""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=self.messages,
                web_search=False
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Lỗi khi gọi model {model}: {e}")
            return None

    def chat(self, user_message: str) -> str:
        """Xử lý tin nhắn người dùng với fallback giữa các model."""
        self.messages.append({"role": "user", "content": user_message})
        self._trim_history()

        for model in self.MODELS:
            response = self._call_model(model)
            if response:
                self.messages.append({"role": "assistant", "content": response})
                self._trim_history()
                return response

        return "Hệ thống tạm thời không phản hồi. Vui lòng thử lại sau."

app = Flask(__name__)

@app.route('/chat', methods=['GET', 'POST'])
def chat_endpoint():
    """API endpoint xử lý yêu cầu chat qua GET hoặc POST."""
    try:
        # Lấy message từ GET hoặc POST
        message = (request.args.get("message", "").strip() if request.method == "GET"
                  else request.json.get("message", "").strip())

        if not message:
            return jsonify({"error": "Vui lòng cung cấp tin nhắn"}), 400

        assistant = AIAssistant()
        response = assistant.chat(message)
        return jsonify({"message": response}), 200

    except Exception as e:
        print(f"Lỗi hệ thống: {e}")
        return jsonify({"error": f"Lỗi hệ thống: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
