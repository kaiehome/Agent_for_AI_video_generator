import os
import json
import random
from typing import List, Dict, Optional
import librosa
import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize

class MusicMatcher:
    """
    BGM音乐匹配和音量调节模块
    """
    def __init__(self, music_library_path: str = 'data/music_library'):
        """
        初始化音乐库
        Args:
            music_library_path: 音乐库目录路径
        """
        self.music_library_path = music_library_path
        self.music_features = self._load_music_features()
    
    def _load_music_features(self) -> Dict[str, Dict]:
        """加载音乐特征库"""
        features_file = os.path.join(self.music_library_path, 'music_features.json')
        if os.path.exists(features_file):
            with open(features_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _extract_music_features(self, audio_path: str) -> Dict:
        """提取音乐特征"""
        y, sr = librosa.load(audio_path)
        
        # 提取音乐特征
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        return {
            'tempo': float(tempo),
            'chroma_mean': np.mean(chroma, axis=1).tolist(),
            'mfcc_mean': np.mean(mfcc, axis=1).tolist(),
            'spectral_contrast_mean': np.mean(spectral_contrast, axis=1).tolist(),
            'duration': librosa.get_duration(y=y, sr=sr)
        }
    
    def match_music(self, mood: str, duration: float) -> Optional[str]:
        """
        根据情绪和时长匹配BGM
        Args:
            mood: 情绪类型 (happy, sad, exciting, calm, etc.)
            duration: 所需音乐时长(秒)
        Returns:
            匹配的音乐文件路径
        """
        # 简单实现：随机选择符合时长的音乐
        # 实际应用中应根据音乐特征进行智能匹配
        matched_music = []
        for music_file, features in self.music_features.items():
            if abs(features['duration'] - duration) < 10:  # 时长相差不超过10秒
                matched_music.append(music_file)
        
        if matched_music:
            return os.path.join(self.music_library_path, random.choice(matched_music))
        return None
    
    def adjust_volume(self, audio_path: str, target_db: float = -20.0) -> str:
        """
        自动调节音频音量
        Args:
            audio_path: 音频文件路径
            target_db: 目标音量(dB)
        Returns:
            处理后的音频文件路径
        """
        sound = AudioSegment.from_file(audio_path)
        normalized_sound = normalize(sound)
        
        # 计算当前音量与目标音量的差异
        current_db = normalized_sound.dBFS
        change_in_db = target_db - current_db
        
        # 应用音量调整
        adjusted_sound = normalized_sound.apply_gain(change_in_db)
        
        # 保存处理后的文件
        output_path = os.path.splitext(audio_path)[0] + '_adjusted.mp3'
        adjusted_sound.export(output_path, format='mp3')
        
        return output_path

    def analyze_script_mood(self, script: str) -> str:
        """
        分析脚本情绪
        Args:
            script: 视频脚本文本
        Returns:
            情绪类型 (happy, sad, exciting, calm, etc.)
        """
        # 简单实现：根据关键词判断情绪
        # 实际应用中应使用NLP模型进行更精确的分析
        positive_words = ['快乐', '开心', '高兴', '兴奋']
        negative_words = ['悲伤', '难过', '伤心', '忧郁']
        
        if any(word in script for word in positive_words):
            return 'happy'
        elif any(word in script for word in negative_words):
            return 'sad'
        return 'neutral'