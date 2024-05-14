import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))

import common.voice_processor as vp


def testis_audio_file():
    assert vp.VoiceProcessor.is_audio_file(".wav")
    assert not vp.VoiceProcessor.is_audio_file(".wai")
    assert vp.VoiceProcessor.is_audio_file("hoyo.wav")
    assert vp.VoiceProcessor.is_audio_file("hoyo/hoyo.hoyo/hoyo.wav")
    assert not vp.VoiceProcessor.is_audio_file("mp3.mp4")
    assert not vp.VoiceProcessor.is_audio_file(".mp31")
