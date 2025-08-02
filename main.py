# main.py (最終実装版)

import os
import sys
import io
import requests
import json
import time
from openai import OpenAI
from dotenv import load_dotenv
import sounddevice as sd
import soundfile as sf
import numpy as np
from chatbot import generate_response # 先ほど作ったファイルをインポート
# ==============================================================================
# ユーザー設定項目
# ==============================================================================

# --- 初期設定 ---
# .envファイルから環境変数を読み込む
load_dotenv()

# --- LLM関連 ---
# OpenRouterで使用するモデル名を指定
OPENROUTER_MODEL = "openai/gpt-3.5-turbo" 


# AivisSpeechの場合: "http://localhost:10101"
# VOICEVOXの場合: "http://localhost:50021"
SPEECH_API_URL = os.getenv("SPEECH_API_URL", "http://localhost:10101")

# 使用したい話者のID (事前にcheck_speakers.pyなどで調べておく)
# 例: 1, 2, 3 など
SPEAKER_ID = int(os.getenv("SPEAKER_ID", "888753760"))

# OBSで読み込むためのテキストファイルのパス
RESPONSE_TEXT_PATH = "./response.txt"
# ==============================================================================
# プログラム本体 (通常はここから下を編集する必要はありません)
# ==============================================================================


def save_text_for_obs(text: str):
    """OBS表示用にテキストをファイルに保存する"""
    try:
        with open(RESPONSE_TEXT_PATH, "w", encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        print(f"[エラー] OBS用テキストファイルの書き込みに失敗しました: {e}")


def text_to_speech(text: str):
    """音声合成APIを呼び出して音声を再生する"""
    if not text:
        print("[音声合成] テキストが空のためスキップしました。")
        return
        
    try:
        # 1. audio_query (音声合成用のクエリを作成)
        query_params = {"text": text, "speaker": SPEAKER_ID}
        res_query = requests.post(f"{SPEECH_API_URL}/audio_query", params=query_params, timeout=10)
        res_query.raise_for_status()
        audio_query = res_query.json()

        # 2. synthesis (クエリを元にWAV音声データを生成)
        synth_params = {"speaker": SPEAKER_ID}
        res_synth = requests.post(f"{SPEECH_API_URL}/synthesis", params=synth_params, json=audio_query, timeout=20)
        res_synth.raise_for_status()
        wav_data = res_synth.content

        # 3. 取得したWAVデータを再生
        data, samplerate = sf.read(io.BytesIO(wav_data))
        sd.play(data, samplerate,device=5)
        sd.wait()
        print("[音声合成] 再生完了。")

    except requests.exceptions.RequestException as e:
        print(f"[エラー] 音声合成APIとの通信に失敗しました: {e}")
        print("-> AivisSpeech/VOICEVOXが起動しているか、URLやポートが正しいか確認してください。")
    except Exception as e:
        print(f"[エラー] 音声の生成または再生で予期せぬエラーが発生しました: {e}")


def main():
    """メイン処理（対話形式）"""
    print("="*50)
    print("自動応答AIとの対話を開始します。")
    print("メッセージを入力し、Enterキーで送信してください。")
    print("終了するには Ctrl + C を押してください。")
    print("="*50)
# --- 初期設定 ---
# .envファイルから環境変数を読み込む
    load_dotenv()
    print("SPEAKER_ID")
    print(os.getenv("SPEAKER_ID", "888753760"))
    try:
        # 対話を続けるための無限ループ
        while True:
            # 標準入力からコメントを受け付ける
            text = input("あなた: ")

            # ユーザーが何も入力せずにEnterを押した場合は、ループの先頭に戻る
            if not text.strip():
                continue

            # 応答を生成
            response_text = generate_response(text)
            print(f"AI: {response_text}")

            # OBS用テキストを保存
            save_text_for_obs(response_text)

            # 音声で応答
            text_to_speech(response_text)

    except KeyboardInterrupt:
        # Ctrl+Cが押されたらループを抜ける
        print("\n\n" + "="*50)
        print("対話を終了します。お疲れ様でした！")
    
    except Exception as e:
        print(f"\n[致命的なエラー] プログラムの実行中に問題が発生しました: {e}")


if __name__ == "__main__":
    main()