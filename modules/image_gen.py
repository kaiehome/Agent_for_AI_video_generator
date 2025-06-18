"""
文生图/视频模块，集成通义万相API
"""
import asyncio
import json
from typing import List, Dict, Optional
import os
import logging
import aiohttp
import requests
import re
from dotenv import load_dotenv
import time

load_dotenv()

DASHSCOPE_API_KEY = os.getenv('DASHSCOPE_API_KEY') or os.getenv('WANXIANG_API_KEY')
IMG_DIR = os.path.join('output', 'images')
os.makedirs(IMG_DIR, exist_ok=True)

WANXIANG_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"
WANXIANG_TASK_URL = "https://dashscope.aliyuncs.com/api/v1/tasks/{}"

DEFAULT_SIZE = "1024*1024"
DEFAULT_STYLE = "<auto>"

STYLE_ENUM = ['<auto>', '<photography>', '<portrait>', '<3d cartoon>', '<anime>', '<oil painting>', '<watercolor>', '<sketch>', '<chinese painting>', '<flat illustration>']

def _get_default_image(idx: int) -> Optional[str]:
    default_path = os.path.join('assets', 'default.jpg')
    if not os.path.exists(default_path) or os.path.getsize(default_path) == 0:
        print(f'[通义万相] default.jpg 占位图不存在或为空，请检查！ idx={idx}')
        return None
    return default_path

def generate_image(text: str, idx: int, style: Optional[str] = None, size: Optional[str] = None, negative_prompt: Optional[str] = None, output_dir: Optional[str] = None) -> str:
    if style == 'general':
        style = '<auto>'
    if not DASHSCOPE_API_KEY:
        print(f'缺少 DASHSCOPE_API_KEY idx={idx}')
        return _get_default_image(idx)
    if not text or not text.strip():
        print(f'[通义万相] content 为空，跳过生图 idx={idx}')
        return _get_default_image(idx)
    if size and not re.match(r'^\d+\*\d+$', size):
        print(f'[通义万相] 非法尺寸参数 size={size} idx={idx}')
        return _get_default_image(idx)
    if style and style not in STYLE_ENUM:
        print(f'[通义万相] 非法风格参数 style={style} idx={idx}')
        return _get_default_image(idx)
    prompt = text.strip()[:800]
    style = style or DEFAULT_STYLE
    size = size or DEFAULT_SIZE
    headers = {
        'Authorization': f'Bearer {DASHSCOPE_API_KEY}',
        'Content-Type': 'application/json',
        'X-DashScope-Async': 'enable'
    }
    payload = {
        'model': 'wanx2.1-t2i-turbo',
        'input': {
            'prompt': prompt
        },
        'parameters': {
            'size': size,
            'n': 1
        }
    }
    if negative_prompt:
        payload['input']['negative_prompt'] = negative_prompt[:500]
    if style:
        payload['parameters']['style'] = style
    try:
        resp = requests.post(WANXIANG_API_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        task_id = result['output']['task_id']
    except Exception as e:
        print(f'[通义万相] 创建任务失败 idx={idx} error={e}')
        return _get_default_image(idx)
    for _ in range(60):
        try:
            time.sleep(3)
            task_url = WANXIANG_TASK_URL.format(task_id)
            task_resp = requests.get(task_url, headers={'Authorization': f'Bearer {DASHSCOPE_API_KEY}'}, timeout=30)
            task_resp.raise_for_status()
            task_result = task_resp.json()
            status = task_result['output']['task_status']
            if status == 'SUCCEEDED':
                results = task_result['output'].get('results', [])
                if results and 'url' in results[0]:
                    image_url = results[0]['url']
                    break
                else:
                    print(f'[通义万相] 结果无图片URL idx={idx}')
                    return _get_default_image(idx)
            elif status == 'FAILED':
                print(f'[通义万相] 任务失败 idx={idx}')
                return _get_default_image(idx)
        except Exception as e:
            print(f'[通义万相] 轮询任务异常 idx={idx} error={e}')
    else:
        print(f'[通义万相] 任务超时未完成 idx={idx}')
        return _get_default_image(idx)
    img_dir = output_dir or IMG_DIR
    os.makedirs(img_dir, exist_ok=True)
    img_path = os.path.join(img_dir, f'segment_{idx}.jpg')
    try:
        img_resp = requests.get(image_url, timeout=60)
        img_resp.raise_for_status()
        with open(img_path, 'wb') as f:
            f.write(img_resp.content)
    except Exception as e:
        print(f'[通义万相] 图片下载失败 idx={idx} error={e}')
        return _get_default_image(idx)
    print(f'[通义万相] 图片生成成功 idx={idx} img_path={img_path}')
    return img_path

class ImageGenerator:
    """
    使用通义万相API生成图片/视频片段
    """
    
    def __init__(self, api_key: str = None):
        # 优先用传入参数，否则用环境变量
        self.api_key = api_key or DASHSCOPE_API_KEY
        if not self.api_key:
            logging.error("[安全] DASHSCOPE_API_KEY 环境变量未设置，且未传入api_key参数，API请求将失败！")
            raise ValueError("DASHSCOPE_API_KEY 环境变量未设置，且未传入api_key参数！")
        self.base_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/image-generation/generation"
    
    async def generate_images(self, script_segments: List[Dict], output_dir: Optional[str] = None) -> List[Dict]:
        """
        根据脚本分镜生成配图
        Args:
            script_segments: 分镜列表，包含description、mood等字段
        Returns:
            包含生成图片URL的分镜列表
        """
        loop = asyncio.get_event_loop()
        results = []
        for idx, seg in enumerate(script_segments):
            img_path = await loop.run_in_executor(None, generate_image, seg.get('description', ''), idx, seg.get('mood', None), None, None, output_dir)
            seg['image_path'] = img_path
            results.append(seg)
        return results
    
    async def generate_video_clips(self, script_segments: List[Dict]) -> List[Dict]:
        """
        根据脚本分镜生成视频片段
        Args:
            script_segments: 分镜列表，包含description、duration等字段
        Returns:
            包含生成视频URL的分镜列表
        """
        tasks = [self._generate_single_video(seg) for seg in script_segments]
        return await asyncio.gather(*tasks)
    
    async def _generate_single_video(self, segment: Dict) -> Dict:
        """
        生成单个分镜的视频片段
        """
        prompt = (
            f"根据以下描述生成视频片段:\n"
            f"主题: {segment.get('description', '')}\n"
            f"时长: {segment.get('duration', 5)}秒\n"
            f"风格: {segment.get('mood', '')}\n"
            f"关键词: {', '.join(segment.get('keywords', []))}"
        )
        
        payload = {
            "prompt": prompt,
            "duration": segment.get('duration', 5),
            "style": segment.get('mood', 'general'),
            "resolution": "1024x576"
        }
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.base_url,
                json=payload,
                headers=headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    segment['video_url'] = result.get('data', {}).get('url')
                else:
                    segment['video_error'] = await response.text()
                
        return segment