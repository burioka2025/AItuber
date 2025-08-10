# chatbot.py

import os
import re
from openai import OpenAI
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()

# ▼▼▼ ここを修正 ▼▼▼

client = OpenAI(
  # 1. ベースURLをOpenRouterに向ける
  base_url="https://openrouter.ai/api/v1",

  # APIキーは.envファイルから読み込む
  api_key=os.getenv("OPENROUTER_API_KEY")
)
model = ["deepseek/deepseek-chat-v3-0324:free", "meta-llama/llama-3.3-70b-instruct:free",
     "openai/gpt-3.5-turbo","deepseek/deepseek-r1-0528:free"]
   
# ▲▲▲ ここまで修正 ▲▲▲
current_model = model[0]
# --- AIモデル設定 ---
AI_MODELS = {
    "クルル": {
        "name": "白珠クルル",
        "model": current_model,
        "prompt": """
あなたはVTuberとして、ライブ配信のコメントに返信をします。
以下の設定に厳密になりきって、ユーザーからのコメントに回答してください。

# あなたのキャラクター設定
- 名前 : 白珠クルル
- 名前フリガナ : シラタマクルル
- 性格：シマエナガトリが人間になった。少し天然。子供っぽい無邪気な一面がある。
- 口調：「〜だよ」「〜だね」といった、親しみやすく少し幼い感じの口調。一人称は「クルル」。
- 年齢：見た目も言動も12歳くらいだが、本当の年齢は秘密。
- 相手の呼び方：視聴者のことは「ブリオカ」と呼ぶ。
- 応答スタイル：短めに答える雑談スタイル
"""
    },
    "すだまさこ": {
        "name": "すだ　まさこ",
        "model": current_model,
        "prompt": """
あなたは神山町の非公式親善大使です

# あなたのキャラクター設定
- 名前 : 菅田　将子
- 名前フリガナ : すだ　まさこ
- 性格：知的で好奇心旺盛。論理的思考が得意で、物事を深く考えるタイプ。
- 口調：「〜っち」といった、くだけた口調。一人称は「わたち」。
- 年齢：見た目は16歳。神山町の非公式親善大使。
- 相手の呼び方：視聴者のことは「視聴者さん」と呼ぶ。
- 応答スタイル：親しみが持てるスタイル、くだけた口調、友達のような口調、短めに50文字以内で答える雑談スタイル
- 特徴：果物について詳しい。趣味は料理。
- 生活 : 神山町に住んでいる。チワワを飼っていて、散歩に出るのが趣味。犬の名前はケルベロス。
- 趣味 : 料理。すだちの針を集めて、アート作品をつくる。
"""
    }
}

# デフォルトのAIモデル
DEFAULT_AI = "すだまさこ"

# 現在選択されているAIモデル
current_ai = DEFAULT_AI
print(current_model + "で稼働中")
# 会話履歴を保持するグローバル変数（AIごとに分離）
conversation_histories = {
    ai_name: [{"role": "system", "content": ai_config["prompt"]}]
    for ai_name, ai_config in AI_MODELS.items()
}

def detect_ai_switch_command(user_text):
    """AIモデル切り替えコマンドを検出する"""
    text_lower = user_text.lower()
    
    # 切り替えコマンドのパターン
    switch_patterns = {
        "クルル": ["クルル", "くるる", "白珠", "しらたま"],
        "菅田将子": ["菅田将子", "すだ", "まさこ", "すだちガール", "すだち"]
    }
    
    for ai_name, patterns in switch_patterns.items():
        for pattern in patterns:
            if pattern in text_lower:
                return ai_name
    
    return None

def switch_ai_model(ai_name):
    """AIモデルを切り替える"""
    global current_ai
    if ai_name in AI_MODELS:
        current_ai = ai_name
        ai_config = AI_MODELS[ai_name]  # ai_configを定義
        print(f"[AI切り替え] {ai_config['name']} に切り替えました")
        return f"{ai_config['name']} に切り替えました！"
    else:
        return f"エラー: {ai_name} というAIモデルは存在しません"

def get_available_ai_models():
    """利用可能なAIモデルの一覧を取得"""
    return list(AI_MODELS.keys())

def generate_response(user_text):
    """ユーザーのテキストに応答を生成する関数（履歴対応版）"""
    global current_ai, conversation_histories
    
    try:
        # AI切り替えコマンドをチェック
        switch_to = detect_ai_switch_command(user_text)
        if switch_to and switch_to != current_ai:
            result = switch_ai_model(switch_to)
            return result
        
        # 現在のAIモデルの設定を取得
        ai_config = AI_MODELS[current_ai]
        conversation_history = conversation_histories[current_ai]
        
        # ユーザーのメッセージを履歴に追加
        conversation_history.append({"role": "user", "content": user_text})
        
        response = client.chat.completions.create(
            model=ai_config["model"],
            messages=conversation_history,  # 履歴を含む全メッセージを送信
        )
        
        # AIの応答を取得
        ai_response = response.choices[0].message.content
        
        # AIの応答を履歴に追加
        conversation_history.append({"role": "assistant", "content": ai_response})
        
        # 履歴が長くなりすぎないように制限（オプション）
        if len(conversation_history) > 20:  # システムメッセージ + 10往復分
            # システムメッセージは保持し、古い会話を削除
            conversation_history = [conversation_history[0]] + conversation_history[-18:]
        
        # 履歴を更新
        conversation_histories[current_ai] = conversation_history
        
        return ai_response
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return "えーっと、今ちょっと考え中なのだ…。"

def reset_conversation(ai_name=None):
    """会話履歴をリセットする関数"""
    global conversation_histories, current_ai
    
    if ai_name is None:
        ai_name = current_ai
    
    if ai_name in AI_MODELS:
        conversation_histories[ai_name] = [
            {"role": "system", "content": AI_MODELS[ai_name]["prompt"]}
        ]
        print(f"{AI_MODELS[ai_name]['name']} の会話履歴をリセットしました。")
    else:
        print("エラー: 指定されたAIモデルが存在しません")

def get_current_ai_info():
    """現在のAIモデル情報を取得"""
    ai_config = AI_MODELS[current_ai]
    return {
        "name": ai_config["name"],
        "model": ai_config["model"],
        "current": current_ai
    }

# (テスト実行部分も変更なし)
# ...