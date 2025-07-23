# 音声ストリームの設定
SAMPLE_RATE = 16000  # Whisperモデルは通常16kHzで学習されているため、リサンプリング後のレート
SAMPLE_WIDTH = 2     # 16bit = 2bytes (s16)
CHANNELS = 1         # モノラル

# 音声認識のバッファリング設定
BUFFER_DURATION_SECONDS = 3  # 何秒分の音声が溜まったら認識処理を行うか
TARGET_BUFFER_SIZE_BYTES = SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS * BUFFER_DURATION_SECONDS

# Whisperモデルの設定
WHISPER_MODEL = "openai/whisper-tiny" # tiny, base, small, medium, largeなど