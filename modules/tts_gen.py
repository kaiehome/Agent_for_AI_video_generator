import edge_tts
import asyncio
import os
import logging
import json
import base64
import hashlib
import hmac
import time
import requests
from urllib.parse import urlencode
import uuid
from typing import Tuple, List, Dict, Optional
from loguru import logger
import wave

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

class AliyunTTSClient:
    """阿里云语音合成客户端"""
    def __init__(self, access_key_id: str, access_key_secret: str):
        self.access_key_id = access_key_id
        self.access_key_secret = access_key_secret
        self.endpoint = "https://nls-meta.cn-shanghai.aliyuncs.com"
        self.api_version = "2019-02-28"
    
    def _sign_request(self, params: dict) -> str:
        """生成请求签名"""
        sorted_params = sorted(params.items())
        canonicalized_query_string = urlencode(sorted_params)
        string_to_sign = "GET&%2F&" + requests.utils.quote(canonicalized_query_string, safe="")
        
        h = hmac.new(
            key=(safe_str(self.access_key_secret, 'access_key_secret') + "&").encode("utf-8"),
            msg=string_to_sign.encode("utf-8"),
            digestmod=hashlib.sha1
        )
        signature = base64.b64encode(h.digest()).strip()
        return signature.decode("utf-8")
    
    async def synthesize(self, text: str, voice: str = "xiaoyun", format: str = "mp3") -> bytes:
        """语音合成"""
        params = {
            "Action": "CreateSynthesizer",
            "Version": self.api_version,
            "AccessKeyId": self.access_key_id,
            "Format": "JSON",
            "Timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "SignatureMethod": "HMAC-SHA1",
            "SignatureVersion": "1.0",
            "SignatureNonce": str(uuid.uuid4()),
            "Text": text,
            "Voice": voice,
            "Format": format
        }
        
        params["Signature"] = self._sign_request(params)
        
        try:
            response = requests.get(self.endpoint, params=params)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logging.error(f"阿里云TTS请求失败: {e}")
            raise

async def generate_audio(text: str, index: int, output_dir: str, use_aliyun: bool = False):
    try:
        path = f"{output_dir}/audio_{index}.mp3"
        if use_aliyun:
            aliyun_client = AliyunTTSClient(
                access_key_id=os.getenv("ALIYUN_ACCESS_KEY_ID"),
                access_key_secret=os.getenv("ALIYUN_ACCESS_KEY_SECRET")
            )
            audio_data = await aliyun_client.synthesize(text, voice="xiaoyun")
            with open(path, "wb") as f:
                f.write(audio_data)
        else:
            communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
            await communicate.save(path)
        
        logging.info(f"音频已保存到 {path}")
        return path
    except Exception as e:
        logging.error(f"音频生成异常: {e}", exc_info=True)
        raise

async def generate_audio_batch_async(texts, output_dir: str, voice: str = "zh-CN-XiaoxiaoNeural", use_aliyun: bool = False):
    os.makedirs(output_dir, exist_ok=True)
    tasks = []
    paths = []
    
    for idx, text in enumerate(texts):
        path = f"{output_dir}/audio_{idx+1}.mp3"
        paths.append(path)
        if use_aliyun:
            aliyun_client = AliyunTTSClient(
                access_key_id=os.getenv("ALIYUN_ACCESS_KEY_ID"),
                access_key_secret=os.getenv("ALIYUN_ACCESS_KEY_SECRET")
            )
            tasks.append(aliyun_client.synthesize(text, voice="xiaoyun"))
        else:
            tasks.append(edge_tts.Communicate(text, voice).save(path))
    
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        audio_paths = []
        errors = []
        
        for idx, res in enumerate(results):
            if isinstance(res, Exception):
                logging.error(f"音频{idx+1}生成失败: {res}")
                errors.append(f"audio_{idx+1}: {str(res)}")
            else:
                if use_aliyun:
                    with open(paths[idx], "wb") as f:
                        f.write(res)
                audio_paths.append(paths[idx])
        
        if errors:
            logging.warning(f"部分音频生成失败: {errors}")
        
        return audio_paths, errors
    except Exception as e:
        logging.error(f"批量音频生成异常: {e}", exc_info=True)
        raise

def safe_str(val, name):
    if val is None:
        raise ValueError(f"音频生成关键参数{name}为None，请检查环境变量或调用参数！")
    return val

AUDIO_DIR = os.path.join('output', 'audios')
os.makedirs(AUDIO_DIR, exist_ok=True)

DEFAULT_VOICE = 'zh-CN-XiaoxiaoNeural'
ALLOWED_VOICES = {'zh-CN-XiaoxiaoNeural', 'zh-CN-YunxiNeural', 'zh-CN-YunyangNeural', 'en-US-JennyNeural'}

async def _async_tts(text: str, audio_path: str, voice: str) -> Optional[float]:
    try:
        import edge_tts
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(audio_path)
        try:
            import mutagen
            from mutagen.mp3 import MP3
            audio = MP3(audio_path)
            duration = audio.info.length
        except Exception as e:
            logger.warning("TTS音频时长解析失败", extra={"audio_path": audio_path, "error": str(e)})
            duration = None
        logger.info("TTS生成成功", extra={"audio_path": audio_path, "voice": voice, "duration": duration})
        return duration
    except Exception as e:
        logger.error("TTS生成失败", extra={"audio_path": audio_path, "voice": voice, "error": str(e)})
        return None

def text_to_speech(text: str, idx: int, voice: str = DEFAULT_VOICE) -> Dict:
    audio_path = os.path.join(AUDIO_DIR, f'segment_{idx+1}.mp3')
    if voice not in ALLOWED_VOICES:
        logger.warning("非法voice参数，使用默认值", extra={"voice": voice})
        voice = DEFAULT_VOICE
    if not text or not text.strip():
        logger.warning('TTS文本为空，跳过', extra={"idx": idx})
        return {'audio_path': '', 'subtitle': text, 'duration': 0, 'voice': voice}
    if len(text) > 500:
        logger.warning('TTS文本过长，截断前500字', extra={"idx": idx})
        text = text[:500]
    try:
        duration = asyncio.run(_async_tts(text, audio_path, voice))
        return {'audio_path': audio_path, 'subtitle': text, 'duration': duration, 'voice': voice}
    except Exception as e:
        logger.error('TTS生成失败', extra={"audio_path": audio_path, "voice": voice, "error": str(e)})
        return {'audio_path': '', 'subtitle': text, 'duration': 0, 'voice': voice}

async def batch_tts(tasks: List[Dict]) -> List[Dict]:
    results = []
    coros = []
    for task in tasks:
        text = task.get('text', '')
        idx = task.get('idx', 0)
        voice = task.get('voice', DEFAULT_VOICE)
        if voice not in ALLOWED_VOICES:
            logger.warning("非法voice参数，使用默认值", extra={"voice": voice})
            voice = DEFAULT_VOICE
        audio_path = os.path.join(AUDIO_DIR, f'segment_{idx+1}.mp3')
        if not text or not text.strip():
            logger.warning(f'[{idx}] TTS文本为空，跳过', extra={"idx": idx})
            results.append({'audio_path': '', 'subtitle': text, 'duration': 0, 'voice': voice})
            continue
        if len(text) > 500:
            logger.warning(f'[{idx}] TTS文本过长，截断前500字', extra={"idx": idx})
            text = text[:500]
        coros.append(_async_tts(text, audio_path, voice))
    durations = await asyncio.gather(*coros)
    coro_idx = 0
    for task in tasks:
        text = task.get('text', '')
        idx = task.get('idx', 0)
        voice = task.get('voice', DEFAULT_VOICE)
        if voice not in ALLOWED_VOICES:
            voice = DEFAULT_VOICE
        audio_path = os.path.join(AUDIO_DIR, f'segment_{idx+1}.mp3')
        if not text or not text.strip():
            continue
        if len(text) > 500:
            text = text[:500]
        duration = durations[coro_idx]
        results.append({'audio_path': audio_path, 'subtitle': text, 'duration': duration, 'voice': voice})
        coro_idx += 1
    return results
