# chatbot.py

import os
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

# ▲▲▲ ここまで修正 ▲▲▲

# (キャラクター設定の character_prompt は変更なし)
# --- VTuberキャラクター設定 ---
# このプロンプトを調整して、キャラクターの個性を出してください
CHARACTER_PROMPT = """
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

def generate_response(user_text):
  """ユーザーのテキストに応答を生成する関数"""
  try:
    response = client.chat.completions.create(
      # 2. 使いたいモデル名を指定（例: GPT-3.5 Turbo）
      # OpenRouterのサイトで使えるモデル名を確認してください
      model="openai/gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": CHARACTER_PROMPT},
        {"role": "user", "content": user_text},
      ]
    )
    return response.choices[0].message.content
  except Exception as e:
    print(f"エラーが発生しました: {e}")
    return "えーっと、今ちょっと考え中なのだ…。"

# (テスト実行部分も変更なし)
# ...