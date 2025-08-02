import subprocess
import time
from chat_downloader import ChatDownloader
from chatbot import generate_response # 先ほど作ったファイルをインポート
import os, sys

# --- 設定項目 ---
YOUTUBE_URL = "https://www.youtube.com/watch?v=YOUR_LIVE_VIDEO_ID" # あなたのYouTube LiveのURL
AIVIS_SPEECH_PATH = "/Applications/vtuber/AivisSpeech.app/Contents/MacOS/AivisSpeech" # Aivis speechの実行ファイルのパス
RESPONSE_TEXT_PATH = "./response.txt" # OBSで読み込むテキストファイルのパス

if not os.path.isfile(AIVIS_SPEECH_PATH):
    sys.exit(f"AivisSpeech 実行ファイルが見つかりません: {AIVIS_SPEECH_PATH}")

def text_to_speech(text):
    """Aivis speechでテキストを読み上げる関数"""
    print(f"音声合成中...: {text}")
    try:
        # Aivis speechのコマンドライン実行（-t の後にテキストを渡す）
        # ※Aivis speechのコマンドライン仕様に応じて変更が必要な場合があります
        subprocess.run([AIVIS_SPEECH_PATH, "-t", text])
    except Exception as e:
        print(f"音声合成でエラーが発生しました: {e}")

def save_text_for_obs(text):
    """OBS表示用にテキストをファイルに保存する関数"""
    try:
        with open(RESPONSE_TEXT_PATH, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        print(f"テキストファイルの書き込みエラー: {e}")

def main():
    print("コメントの待機を開始します...")
    #downloader = ChatDownloader()
    try:
        print("標準入力からコメントを受け付けます。終了するには Ctrl+C を押してください。")
        def chat_generator():
            while True:
                try:
                    text = input("コメント: ")
                    if text.strip() == "":
                        continue
                    yield {"author": {"name": "User"}, "message": text}
                except EOFError:
                    break
        chat = chat_generator()
        for message in chat:
            author = message['author']['name']
            text = message['message']
            print(f"[{author}]: {text}")

            # --- ここからが自動応答の核 ---
            # 1. LLMで応答を生成
            response_text = generate_response(text)
            print(f"[AI]: {response_text}")

            # 2. OBS用にテキストをファイルに保存
            save_text_for_obs(response_text)

            # 3. Aivis speechで音声を再生
            text_to_speech(response_text)

            time.sleep(1) # 少し待機

    except Exception as e:
        print(f"チャットの取得でエラーが発生しました: {e}")
    finally:
        # downloader.close() # This line was removed as per the new_code, as the downloader object is no longer defined.
        pass # Added pass to avoid syntax error if the line was removed.

if __name__ == "__main__":
    main()