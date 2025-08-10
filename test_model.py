import os
import requests
import json
from dotenv import load_dotenv
# --- 初期設定 ---
# .envファイルから環境変数を読み込む
load_dotenv()
# 環境変数 'OPENROUTER_API_KEY' からAPIキーを読み込む
api_key = os.getenv("OPENROUTER_API_KEY")

# APIキーが設定されているか確認
if not api_key:
    print("エラー: 環境変数 'OPENROUTER_API_KEY' が設定されていません。")
else:
    # ヘッダー情報
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    model = ["deepseek/deepseek-chat-v3-0324:free", "meta-llama/llama-3.3-70b-instruct:free",
     "openai/gpt-3.5-turbo","deepseek/deepseek-r1-0528:free"]
    # 送信するデータ
    data = {
        "model": model[3], # または他の試したいモデル
        "messages": [
            {"role": "user", "content": "こんにちは、自己紹介をしてください。"}
        ]
    }

    try:
        print("APIにリクエストを送信します...(最大60秒待機)")
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            data=json.dumps(data),
            timeout=180  # タイムアウトを60秒に設定
        )
        response.raise_for_status() # 4xx, 5xxエラーの場合は例外を発生させる
        
        print("\n--- 成功しました！ ---")
        print(response.json())

    except requests.exceptions.Timeout:
        print("\n--- エラー: タイムアウト ---")
        print("リクエストがタイムアウトしました。モデルの応答が遅いか、ネットワークに問題がある可能性があります。")

    except requests.exceptions.HTTPError as err:
        print("\n--- エラー: HTTPエラー ---")
        print(f"HTTPステータスコード: {err.response.status_code}")
        print("サーバーからのレスポンス内容:")
        print(err.response.text) # 最も重要なエラー情報

    except Exception as e:
        print(f"\n--- 予期せぬエラーが発生しました ---")
        print(e)