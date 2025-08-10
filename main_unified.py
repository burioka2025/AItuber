# main_unified.py (テキスト・音声入力切り替え対応版)

import os
import sys
import io
import requests
import json
import time
import whisper
from openai import OpenAI
from dotenv import load_dotenv
import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import torch
import torchaudio.functional as AF
import shutil
from chatbot import generate_response, get_current_ai_info, get_available_ai_models

# ==============================================================================
# ユーザー設定項目
# ==============================================================================

# AivisSpeechの場合: "http://localhost:10101"
# VOICEVOXの場合: "http://localhost:50021"
SPEECH_API_URL = os.getenv("SPEECH_API_URL", "http://localhost:10101")

# 使用したい話者のID (事前にcheck_speakers.pyなどで調べておく)
# 例: 1, 2, 3 など
SPEAKER_ID = int(os.getenv("SPEAKER_ID", "888753760"))

# OBSで読み込むためのテキストファイルのパス
RESPONSE_TEXT_PATH = "./response.txt"

# 音声認識設定
LANGUAGE = "ja"  # 日本語（Whisper用）
ENERGY_THRESHOLD = 1000  # 音声検出の感度（4000→1000に下げて感度向上）
PAUSE_THRESHOLD = 0.8  # 無音検出の秒数
SAMPLE_RATE = 16000  # Whisper用のサンプリングレート

# 男性声向け 前処理フィルタ設定
HPF_CUTOFF_HZ = 80.0   # ハイパス 低域ノイズ抑制（呼吸/ハム）
LPF_CUTOFF_HZ = 5000.0 # ローパス 高域ノイズ抑制
PRE_EMPHASIS = 0.97    # プリエンファシス係数
VAD_WINDOW_SEC = 0.03  # VADフレーム長
VAD_HOP_SEC = 0.01     # VADホップ長
VAD_MARGIN_SEC = 0.2   # 前後マージン

# Whisperモデル設定
WHISPER_MODEL = "small"  # "tiny", "base", "small", "medium", "large" から選択

# Whisperデコード設定（精度重視）
WHISPER_DECODE_OPTS = {
    "language": LANGUAGE,
    "temperature": 0.0,          # 貪欲+ビームサーチ
    "beam_size": 5,              # ビーム幅
    "patience": 1.0,
    "condition_on_previous_text": False,
    "fp16": False,               # CPU想定
}

# ==============================================================================
# プログラム本体
# ==============================================================================

# --- 初期設定 ---
# .envファイルから環境変数を読み込む
load_dotenv()

# ffmpeg の存在確認とPATH補正
def _ensure_ffmpeg_in_path():
    if shutil.which("ffmpeg") is not None:
        return
    for p in ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"]:
        ff = os.path.join(p, "ffmpeg")
        if os.path.exists(ff):
            os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
            break

_ensure_ffmpeg_in_path()

# 再生デバイスの自動選択（環境変数 SD_OUTPUT_DEVICE_INDEX があれば優先）
def _setup_output_device():
    try:
        env_idx = os.getenv("SD_OUTPUT_DEVICE_INDEX")
        env_name = os.getenv("SD_OUTPUT_DEVICE_NAME")  # 例: "VB-Cable"
        in_idx, out_idx = None, None
        current = sd.default.device
        if isinstance(current, (list, tuple)) and len(current) == 2:
            in_idx = current[0]
            out_idx = current[1]

        devices = sd.query_devices()
        def find_out_by_name_substr(substrs):
            if not substrs:
                return None
            substrs_norm = [s.lower() for s in substrs if isinstance(s, str) and s]
            for i, d in enumerate(devices):
                if d.get("max_output_channels", 0) <= 0:
                    continue
                name = str(d.get("name", ""))
                name_norm = name.lower()
                if any(s in name_norm for s in substrs_norm):
                    return i
            return None

        # 1) 環境変数でインデックス指定があれば最優先
        if env_idx is not None:
            out_idx = int(env_idx)
        else:
            # 2) 環境変数で名前を指定できる
            if env_name:
                idx = find_out_by_name_substr([env_name])
                if idx is not None:
                    out_idx = idx
            # 3) VB-Cable を最優先で自動選択
            if out_idx is None:
                vb_idx = find_out_by_name_substr(["vb-cable", "vb cable", "vb", "vb_cable"]) 
                if vb_idx is not None:
                    out_idx = vb_idx
            # 4) 次点: スピーカー/ Speaker を選択
            if out_idx is None:
                spk_idx = find_out_by_name_substr(["スピーカー", "speaker"]) 
                if spk_idx is not None:
                    out_idx = spk_idx
            # 5) 最後の手段: 最初の出力可能デバイス
            if out_idx is None:
                for i, d in enumerate(devices):
                    if d.get("max_output_channels", 0) > 0:
                        out_idx = i
                        break

        if out_idx is not None:
            sd.default.device = (in_idx, out_idx)
            try:
                dev = sd.query_devices(out_idx)
                print(f"[音声出力] デフォルト出力デバイス: #{out_idx} {dev.get('name')}")
            except Exception:
                pass
    except Exception:
        pass

_setup_output_device()

# Whisperモデルを初期化
print("Whisperモデルを読み込み中...")
whisper_model = whisper.load_model(WHISPER_MODEL)
print(f"Whisperモデル '{WHISPER_MODEL}' の読み込み完了")

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
        sd.play(data, samplerate)  # デフォルト出力を使用
        sd.wait()
        print("[音声合成] 再生完了。")

    except requests.exceptions.RequestException as e:
        print(f"[エラー] 音声合成APIとの通信に失敗しました: {e}")
        print("-> AivisSpeech/VOICEVOXが起動しているか、URLやポートが正しいか確認してください。")
    except Exception as e:
        print(f"[エラー] 音声の生成または再生で予期せぬエラーが発生しました: {e}")

def _pre_emphasis(signal: np.ndarray, coeff: float) -> np.ndarray:
    if signal.size == 0:
        return signal
    emphasized = np.empty_like(signal)
    emphasized[0] = signal[0]
    emphasized[1:] = signal[1:] - coeff * signal[:-1]
    return emphasized

def _apply_male_band_filters(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """torchaudioのバイカッドで男性声帯域を強調しノイズを抑制"""
    try:
        tensor = torch.from_numpy(signal).float().unsqueeze(0)  # [1, T]
        # DCオフセット除去
        tensor = tensor - tensor.mean(dim=-1, keepdim=True)
        # ハイパス/ローパス
        tensor = AF.highpass_biquad(tensor, sample_rate, HPF_CUTOFF_HZ, Q=0.707)
        tensor = AF.lowpass_biquad(tensor, sample_rate, LPF_CUTOFF_HZ, Q=0.707)
        return tensor.squeeze(0).cpu().numpy()
    except Exception as e:
        print(f"フィルタ処理でエラー: {e}")
        return signal  # エラー時は元の信号を返す

def _normalize_peak(signal: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    peak = float(np.max(np.abs(signal))) if signal.size > 0 else 0.0
    if peak > 0:
        return signal * (target_peak / peak)
    return signal

def _trim_silence_energy(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """簡易VAD: 短時間エネルギーで前後の無音をトリム"""
    if signal.size == 0:
        return signal
    frame_len = int(VAD_WINDOW_SEC * sample_rate)
    hop_len = int(VAD_HOP_SEC * sample_rate)
    if frame_len <= 0:
        return signal

    # フレームRMS
    num_frames = 1 + max(0, (len(signal) - frame_len) // hop_len)
    if num_frames <= 0:
        return signal
    rms = np.empty(num_frames, dtype=np.float32)
    for i in range(num_frames):
        start = i * hop_len
        end = start + frame_len
        frame = signal[start:end]
        if frame.size == 0:
            rms[i] = 0.0
        else:
            rms[i] = np.sqrt(np.mean(frame.astype(np.float32) ** 2))

    # しきい値: 環境に合わせ自動調整（中央値ベース）
    thresh = max(0.01, float(np.median(rms)) * 0.7)
    voiced = np.where(rms > thresh)[0]
    if voiced.size == 0:
        return signal

    start_frame = int(voiced[0])
    end_frame = int(voiced[-1])
    margin = int(VAD_MARGIN_SEC / VAD_HOP_SEC)
    start_frame = max(0, start_frame - margin)
    end_frame = min(num_frames - 1, end_frame + margin)

    start_sample = start_frame * hop_len
    end_sample = min(len(signal), end_frame * hop_len + frame_len)
    return signal[start_sample:end_sample]

def preprocess_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """男性声向けに音声を前処理"""
    try:
        if audio is None or audio.size == 0:
            return audio
        # モノラル化・1次元化
        audio = audio.astype(np.float32).flatten()
        # DC除去
        audio = audio - float(np.mean(audio))
        # 正規化（ピーク）
        audio = _normalize_peak(audio, target_peak=0.95)
        # プリエンファシス
        audio = _pre_emphasis(audio, PRE_EMPHASIS)
        # 帯域フィルタ
        audio = _apply_male_band_filters(audio, sample_rate)
        # 再度正規化（過度な振幅を抑制）
        audio = _normalize_peak(audio, target_peak=0.9)
        # 無音トリム
        audio = _trim_silence_energy(audio, sample_rate)
        return audio
    except Exception as e:
        print(f"音声前処理でエラー: {e}")
        return audio  # エラー時は元の音声を返す

def record_audio():
    """マイクから音声を録音する"""
    print("話してください...")
    
    try:
        # 音声を録音
        audio_data = sd.rec(int(SAMPLE_RATE * 10), samplerate=SAMPLE_RATE, channels=1, dtype=np.float32)
        sd.wait()
        
        # 無音部分を除去
        audio_data = audio_data.flatten()
        
        # 音声レベルをチェック（閾値を下げて感度向上）
        if np.max(np.abs(audio_data)) < 0.005:  # 0.01 → 0.005に下げて感度向上
            print("音声が検出されませんでした。")
            return None
            
        return audio_data
        
    except Exception as e:
        print(f"音声録音でエラーが発生しました: {e}")
        return None

def listen_for_speech():
    """マイクから音声を聞き取ってテキストに変換する（ローカルWhisper使用）"""
    
    # 音声を録音
    audio_data = record_audio()
    if audio_data is None:
        return None

    # 男性声向け 前処理
    audio_data = preprocess_audio(audio_data, SAMPLE_RATE)
    if audio_data is None or audio_data.size == 0:
        print("音声を認識できませんでした。")
        return None
    
    try:
        print("音声を認識中...")
        
        # 一時ファイルに音声を保存
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            
        # 音声データをWAVファイルとして保存
        sf.write(temp_path, audio_data, SAMPLE_RATE)
        
        # Whisperで音声をテキストに変換（精度重視オプション）
        result = whisper_model.transcribe(temp_path, **WHISPER_DECODE_OPTS)
        text = result.get("text", "").strip()
        
        # 一時ファイルを削除
        os.unlink(temp_path)
        
        if text:
            print(f"認識結果: {text}")
            return text
        else:
            print("音声を認識できませんでした。")
            return None
            
    except Exception as e:
        print(f"音声認識で予期せぬエラーが発生しました: {e}")
        # 一時ファイルの削除を試行
        try:
            if 'temp_path' in locals():
                os.unlink(temp_path)
        except:
            pass
        return None

def show_menu():
    """メインメニューを表示"""
    current_ai_info = get_current_ai_info()
    available_models = get_available_ai_models()
    
    print("="*50)
    print("AI VTuber チャットボット")
    print("="*50)
    print(f"現在のAI: {current_ai_info['name']} ({current_ai_info['model']})")
    print("利用可能なAI:", ", ".join(available_models))
    print("="*50)
    print("1. テキスト入力モード")
    print("2. 音声入力モード")
    print("3. AIモデル切り替え")
    print("4. 終了")
    print("="*50)
    
    while True:
        try:
            choice = input("選択してください (1-4): ").strip()
            if choice == '1':
                return 'text'
            elif choice == '2':
                return 'voice'
            elif choice == '3':
                return 'switch_ai'
            elif choice == '4':
                return 'exit'
            else:
                print("1, 2, 3, または 4 を入力してください。")
        except KeyboardInterrupt:
            return 'exit'

def switch_ai_menu():
    """AIモデル切り替えメニュー"""
    from chatbot import switch_ai_model, get_available_ai_models, get_current_ai_info
    
    current_ai_info = get_current_ai_info()
    available_models = get_available_ai_models()
    
    print("="*50)
    print("AIモデル切り替え")
    print("="*50)
    print(f"現在のAI: {current_ai_info['name']}")
    print("="*50)
    
    for i, ai_name in enumerate(available_models, 1):
        print(f"{i}. {ai_name}")
    
    print(f"{len(available_models) + 1}. 戻る")
    print("="*50)
    
    while True:
        try:
            choice = input(f"選択してください (1-{len(available_models) + 1}): ").strip()
            choice_num = int(choice)
            
            if 1 <= choice_num <= len(available_models):
                selected_ai = available_models[choice_num - 1]
                result = switch_ai_model(selected_ai)
                print(result)
                input("Enterキーを押して続行...")
                return 'menu'
            elif choice_num == len(available_models) + 1:
                return 'menu'
            else:
                print(f"1 から {len(available_models) + 1} の数字を入力してください。")
        except ValueError:
            print("有効な数字を入力してください。")
        except KeyboardInterrupt:
            return 'menu'

def text_input_mode():
    """テキスト入力モード"""
    current_ai_info = get_current_ai_info()
    print("="*50)
    print("テキスト入力モード")
    print(f"現在のAI: {current_ai_info['name']}")
    print("メッセージを入力し、Enterキーで送信してください。")
    print("AI切り替え: メッセージに「クルル」「アリス」を含める")
    print("モード切り替え: 'voice' と入力")
    print("終了: Ctrl + C")
    print("="*50)

    try:
        while True:
            text = input("あなた: ")
            
            if not text.strip():
                continue
                
            # モード切り替えコマンド
            if text.lower() == 'voice':
                return 'voice'
            
            # 応答を生成
            response_text = generate_response(text)
            print(f"AI: {response_text}")

            # OBS用テキストを保存
            save_text_for_obs(response_text)

            # 音声で応答
            text_to_speech(response_text)

    except KeyboardInterrupt:
        return 'exit'

def voice_input_mode():
    """音声入力モード"""
    current_ai_info = get_current_ai_info()
    print("="*50)
    print("音声入力モード")
    print(f"現在のAI: {current_ai_info['name']}")
    print("マイクに向かって話してください。")
    print("モード切り替え: 'text' と話す")
    print("終了: Ctrl + C")
    print("="*50)

    try:
        while True:
            # 音声入力を待機
            user_text = listen_for_speech()
            
            # 音声認識に失敗した場合は次のループへ
            if not user_text:
                continue

            # モード切り替えコマンド
            if user_text.lower() in ['テキスト', 'text', 'テキストモード']:
                return 'text'
            
            # 応答を生成
            response_text = generate_response(user_text)
            print(f"AI: {response_text}")

            # OBS用テキストを保存
            save_text_for_obs(response_text)

            # 音声で応答
            text_to_speech(response_text)

    except KeyboardInterrupt:
        return 'exit'

def main():
    """メイン処理（モード切り替え対応）"""
    print("AI VTuber チャットボットを起動中...")
    
    # 初期モード選択
    current_mode = show_menu()
    
    while current_mode != 'exit':
        if current_mode == 'text':
            current_mode = text_input_mode()
        elif current_mode == 'voice':
            current_mode = voice_input_mode()
        elif current_mode == 'switch_ai':
            current_mode = switch_ai_menu()
        elif current_mode == 'menu':
            current_mode = show_menu()
        else:
            break
    
    print("\n" + "="*50)
    print("プログラムを終了します。お疲れ様でした！")
    print("="*50)

if __name__ == "__main__":
    main() 