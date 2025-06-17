import os
# os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# os.environ['https_proxy'] = 'http://127.0.0.1:7890'

from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, Optional, List
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Depends, Header
from uuid import uuid4
import os
import shutil
import requests
import json
from diffusers import StableDiffusionPipeline
import torch
import asyncio
import edge_tts
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips, CompositeAudioClip
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from fastapi import status
from modules.image_gen import ImageGenerator
from modules.tts_gen import generate_audio_batch_async, AliyunTTSClient
from modules.video_effects import add_subtitle_ffmpeg, add_cover_ffmpeg, add_overlay_ffmpeg
from modules.music_match import MusicMatcher
from modules.parser import parse_script

from dotenv import load_dotenv
load_dotenv()

from fastapi.responses import JSONResponse
from fastapi.requests import Request
import logging.handlers
from functools import wraps
import inspect
import glob
import re

# 日志配置：写入logs/app.log，按天切分，保留7天
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "app.log")
handler = logging.handlers.TimedRotatingFileHandler(log_file, when="midnight", backupCount=7, encoding="utf-8")
formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
handler.setFormatter(formatter)
logging.basicConfig(level=logging.INFO, handlers=[handler])

# 操作日志装饰器
def op_log(action):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            user = kwargs.get('x_api_token', 'unknown')
            logging.info(f"[操作日志] {action} by {user} at {datetime.now().isoformat()}")
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# 统一返回结构装饰器
def api_response(func):
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            result = await func(*args, **kwargs)
            return {"code": 0, "msg": "success", "data": result}
        except HTTPException as e:
            logging.warning(f"[API异常] {e.detail}")
            return {"code": e.status_code, "msg": e.detail, "data": None}
        except Exception as e:
            logging.error(f"[API异常] {str(e)}", exc_info=True)
            return {"code": 500, "msg": "服务器内部错误", "data": None}
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return {"code": 0, "msg": "success", "data": result}
        except HTTPException as e:
            logging.warning(f"[API异常] {e.detail}")
            return {"code": e.status_code, "msg": e.detail, "data": None}
        except Exception as e:
            logging.error(f"[API异常] {str(e)}", exc_info=True)
            return {"code": 500, "msg": "服务器内部错误", "data": None}
    if inspect.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

# 通义万相API配置
WANXIANG_API_KEY = os.getenv("WANXIANG_API_KEY")
# 阿里云TTS配置
ALIYUN_ACCESS_KEY_ID = os.getenv("ALIYUN_ACCESS_KEY_ID")
ALIYUN_ACCESS_KEY_SECRET = os.getenv("ALIYUN_ACCESS_KEY_SECRET")
import traceback
import logging
import re
import psutil

class ResourceMonitor:
    """系统资源监控器"""
    def __init__(self, cpu_threshold: float = 0.8, mem_threshold: float = 0.8):
        self.cpu_threshold = cpu_threshold
        self.mem_threshold = mem_threshold
        self.history = {
            'cpu': [],
            'memory': [],
            'disk': []
        }
    
    async def check_resources(self) -> bool:
        """检查系统资源是否可用"""
        cpu_percent = await self._get_cpu_usage()
        mem_percent = await self._get_memory_usage()
        disk_percent = await self._get_disk_usage()
        
        # 记录历史数据
        await self._update_history('cpu', cpu_percent)
        await self._update_history('memory', mem_percent)
        await self._update_history('disk', disk_percent)
        
        # 检查资源使用情况
        overload = False
        if cpu_percent > self.cpu_threshold:
            logging.warning(f"CPU overload: {cpu_percent:.1%} > {self.cpu_threshold:.0%}")
            overload = True
        if mem_percent > self.mem_threshold:
            logging.warning(f"Memory overload: {mem_percent:.1%} > {self.mem_threshold:.0%}")
            overload = True
        if disk_percent > 0.9:  # 磁盘空间不足
            logging.error(f"Disk space critical: {disk_percent:.1%}")
            overload = True
            
        if overload:
            await self._log_resource_trends()
            return False
            
        return True
    
    async def _update_history(self, metric: str, value: float):
        """更新资源使用历史记录"""
        async with self.lock:
            self.history[metric].append(value)
            if len(self.history[metric]) > 60:  # 保留最近60次记录
                self.history[metric] = self.history[metric][-60:]
    
    async def _log_resource_trends(self):
        """记录资源使用趋势"""
        async with self.lock:
            for metric, values in self.history.items():
                if values:
                    avg = sum(values) / len(values)
                    max_val = max(values)
                    min_val = min(values)
                    logging.info(
                        f"{metric.capitalize()} usage trends - "
                        f"Avg: {avg:.1%}, Max: {max_val:.1%}, Min: {min_val:.1%}"
                    )
    
    async def _get_cpu_usage(self) -> float:
        """获取CPU使用率"""
        return psutil.cpu_percent(interval=1) / 100
    
    async def _get_memory_usage(self) -> float:
        """获取内存使用率"""
        return psutil.virtual_memory().percent / 100
        
    async def _get_disk_usage(self) -> float:
        """获取磁盘使用率"""
        return psutil.disk_usage('/').percent / 100

app = FastAPI()

class TaskStatus(Enum):
    CREATED = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()

@dataclass
class Task:
    id: str
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    progress: float = 0.0
    result: Optional[Dict] = None

class AIServiceClient:
    """AI服务客户端基类"""
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.retry_count = 3
        self.timeout = 30
    
    async def call_api(self, payload: Dict) -> Dict:
        """调用API的通用方法"""
        # 检查系统资源
        if not await self.resource_monitor.check_resources():
            raise HTTPException(
                status_code=429,
                detail="System resources are under heavy load, please try again later"
            )
            
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        last_error = None
        for attempt in range(self.retry_count):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.base_url,
                        headers=headers,
                        json=payload,
                        timeout=self.timeout
                    ) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            error_msg = await response.text()
                            last_error = f"API request failed with status {response.status}: {error_msg}"
            except Exception as e:
                last_error = str(e)
            
            if attempt < self.retry_count - 1:
                await asyncio.sleep(2 ** attempt)  # 指数退避
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed after {self.retry_count} attempts. Last error: {last_error}"
        )

class QwenClient(AIServiceClient):
    """通义千问API客户端"""
    def __init__(self, api_key: str = None):
        super().__init__(
            api_key=api_key or QWEN_API_KEY,
            base_url=QWEN_API_URL,
            model=QWEN_MODEL
        )
    
    async def generate_storyboards(self, script: str) -> List[Dict]:
        """生成分镜"""
        prompt = f"请将以下脚本拆分为分镜，每个分镜包含画面描述和台词，输出JSON数组，每个元素包含description和dialogue字段：\n{script}"
        payload = {
            "model": self.model,
            "input": {"prompt": prompt}
        }
        
        response = await self.call_api(payload)
        try:
            output = response.get('output', '{}')
            if isinstance(output, str):
                return json.loads(output)
            return output
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Qwen API parse error: {str(e)}")

class DeepSeekClient(AIServiceClient):
    """DeepSeek API客户端"""
    def __init__(self, api_key: str = None):
        super().__init__(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_API_URL,
            model=DEEPSEEK_MODEL
        )
    
    async def generate_storyboards(self, script: str, polish: bool = True) -> List[Dict]:
        """
        生成分镜，包含详细语气和时间节奏分析
        Args:
            script: 输入脚本
            polish: 是否使用通义千问进行文本润色
        Returns:
            包含以下字段的分镜列表:
            - description: 画面描述
            - dialogue: 台词
            - mood: 语气分析(gentle, exciting, etc.)
            - pace: 节奏(slow, medium, fast)
            - duration: 建议时长(秒)
            - keywords: 关键词列表(用于音乐匹配)
            - polished_text: 润色后的文本(当polish=True时)
        """
        prompt = (
            "请深入分析以下脚本，将其拆分为分镜，并详细分析每个分镜的语气特征和时间节奏。"
            "具体要求:\n"
            "1. 画面描述(description): 详细描述视觉元素和构图\n"
            "2. 台词(dialogue): 精确提取台词文本\n"
            "3. 语气(mood): 从以下选项中选择: gentle, exciting, sad, happy, mysterious, romantic, serious, humorous\n"
            "4. 节奏(pace): 根据内容密度和情绪强度判断: slow, medium, fast\n"
            "5. 时长(duration): 根据台词长度和节奏建议时长(秒)\n"
            "6. 关键词(keywords): 提取3-5个描述场景/情绪的关键词\n"
            "输出格式要求:\n"
            "- 严格使用JSON数组格式\n"
            "- 每个分镜必须包含上述所有字段\n"
            "- 时长必须合理(台词1秒约4-6字)\n"
            "脚本内容:\n"
            f"{script}"
        )
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        response = await self.call_api(payload)
        try:
            content = response.get('choices', [{}])[0].get('message', {}).get('content', '{}')
            if not content.strip():
                raise ValueError("Empty response from API")
                
            # 验证JSON格式和必填字段
            storyboards = json.loads(content)
            required_fields = {'description', 'dialogue', 'mood', 'pace', 'duration', 'keywords'}
            for sb in storyboards:
                if not all(field in sb for field in required_fields):
                    raise ValueError(f"Missing required fields in storyboard: {sb}")
                
                # 验证时长合理性
                word_count = len(sb['dialogue'].split())
                min_duration = max(2, word_count / 6)  # 最少2秒，最多1秒6字
                max_duration = word_count / 4
                if not (min_duration <= sb['duration'] <= max_duration):
                    sb['duration'] = min(max_duration, max(min_duration, sb['duration']))
                    
            if polish:
                # 调用通义千问进行文本润色
                polish_prompt = (
                    "请对以下分镜文本进行专业润色，提升文案质量和传播力:\n"
                    f"{json.dumps(storyboards, ensure_ascii=False)}\n"
                    "润色要求:\n"
                    "1. 保持原意的同时优化表达\n"
                    "2. 增强文案的感染力和传播性\n"
                    "3. 保持专业性和准确性\n"
                    "4. 输出格式必须与原JSON结构一致"
                )
                polish_payload = {
                    "model": "qwen-turbo",
                    "messages": [{"role": "user", "content": polish_prompt}],
                    "temperature": 0.5
                }
                polish_response = await self.call_api(polish_payload)
                polished_content = polish_response.get('choices', [{}])[0].get('message', {}).get('content', '{}')
                polished_storyboards = json.loads(polished_content)
                
                # 合并润色结果
                for orig, polished in zip(storyboards, polished_storyboards):
                    orig['polished_text'] = polished
                
            return storyboards
            content_clean = re.sub(r'^```json\s*|```$', '', content.strip(), flags=re.MULTILINE)
            return json.loads(content_clean)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"DeepSeek API parse error: {str(e)}")

class TaskScheduler:
    def __init__(self, max_concurrent_tasks: int = 3, persistence_file: str = 'task_state.json'):
        self.tasks: Dict[str, Task] = {}
        self.pending_queue: List[str] = []
        self.max_concurrent_tasks = max_concurrent_tasks
        self.current_tasks = 0
        self.lock = asyncio.Lock()  # 使用异步锁
        self.persistence_file = persistence_file
        self.error_types = {
            'api_error': 'API调用失败',
            'timeout': '请求超时',
            'validation': '数据验证失败',
            'io_error': '文件IO错误',
            'resource': '资源不足',
            'unknown': '未知错误'
        }
        self.resource_monitor = ResourceMonitor()
        self._load_tasks()
        
    async def _periodic_cleanup(self):
        """定期清理过期任务和检查系统资源"""
        while True:
            try:
                # 清理超过24小时未完成的失败任务
                now = datetime.now()
                async with self.lock:
                    for task_id, task in list(self.tasks.items()):
                        if task.status == TaskStatus.FAILED and \
                           (now - task.completed_at).total_seconds() > 86400:
                            del self.tasks[task_id]
                            if task_id in self.pending_queue:
                                self.pending_queue.remove(task_id)
                    await self._save_tasks()
                
                # 检查系统资源并调整并发数
                cpu_usage = await self.resource_monitor._get_cpu_usage()
                mem_usage = await self.resource_monitor._get_memory_usage()
                
                if cpu_usage > 0.8 or mem_usage > 0.8:
                    self.max_concurrent_tasks = max(1, self.max_concurrent_tasks - 1)
                elif cpu_usage < 0.5 and mem_usage < 0.5:
                    self.max_concurrent_tasks = min(5, self.max_concurrent_tasks + 1)
                
                await asyncio.sleep(60)  # 每分钟检查一次
            except Exception as e:
                logging.error(f"Periodic cleanup error: {e}")
                await asyncio.sleep(60)
    
    def _load_tasks(self):
        """从持久化文件加载任务状态"""
        if os.path.exists(self.persistence_file):
            try:
                with open(self.persistence_file, 'r') as f:
                    data = json.load(f)
                    for task_id, task_data in data.get('tasks', {}).items():
                        self.tasks[task_id] = Task(
                            id=task_id,
                            status=TaskStatus[task_data['status']],
                            created_at=datetime.fromisoformat(task_data['created_at']),
                            started_at=datetime.fromisoformat(task_data['started_at']) if task_data['started_at'] else None,
                            completed_at=datetime.fromisoformat(task_data['completed_at']) if task_data['completed_at'] else None,
                            error=task_data.get('error'),
                            progress=task_data.get('progress', 0.0),
                            result=task_data.get('result')
                        )
                    self.pending_queue = data.get('pending_queue', [])
                    self.current_tasks = data.get('current_tasks', 0)
            except Exception as e:
                logging.error(f"Failed to load tasks from {self.persistence_file}: {str(e)}")
    
    async def _save_tasks(self):
        """保存任务状态到持久化文件"""
        async with self.lock:
            try:
                tasks_data = {}
                for task_id, task in self.tasks.items():
                    tasks_data[task_id] = {
                        'status': task.status.name,
                        'created_at': task.created_at.isoformat(),
                        'started_at': task.started_at.isoformat() if task.started_at else None,
                        'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                        'error': task.error,
                        'progress': task.progress,
                        'result': task.result
                    }
                data = {
                    'tasks': tasks_data,
                    'pending_queue': self.pending_queue,
                    'current_tasks': self.current_tasks
                }
                with open(self.persistence_file, 'w') as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                logging.error(f"Failed to save tasks to {self.persistence_file}: {str(e)}")
                
    def classify_error(self, error: Exception) -> str:
        """根据异常类型分类错误"""
        error_str = str(error).lower()
        if 'timeout' in error_str or 'timed out' in error_str:
            return 'timeout'
        elif 'api' in error_str or 'http' in error_str:
            return 'api_error'
        elif 'validation' in error_str or 'invalid' in error_str:
            return 'validation'
        elif 'file' in error_str or 'io' in error_str or 'os' in error_str:
            return 'io_error'
        elif 'memory' in error_str or 'resource' in error_str:
            return 'resource'
        return 'unknown'
    
    def should_retry(self, error_type: str) -> bool:
        """根据错误类型判断是否应该重试"""
        return error_type in ['api_error', 'timeout', 'resource']
    
    async def recover_failed_tasks(self, max_retries: int = 3) -> List[str]:
        """恢复可重试的失败任务"""
        recovered_ids = []
        async with self.lock:
            for task_id, task in self.tasks.items():
                if task.status == TaskStatus.FAILED and task.error:
                    error_type = self.classify_error(Exception(task.error))
                    if self.should_retry(error_type) and \
                       task.error.count('Retry') < max_retries:
                        task.status = TaskStatus.CREATED
                        task.error = f"{task.error} (Retry {task.error.count('Retry') + 1})"
                        self.pending_queue.append(task_id)
                        recovered_ids.append(task_id)
            await self._save_tasks()
        return recovered_ids
    
    async def create_task(self, priority: int = 1, timeout: int = 300) -> str:
        """创建新任务
        Args:
            priority: 任务优先级 (1=low, 2=normal, 3=high)
            timeout: 任务超时时间(秒)
        """
        task_id = str(uuid4())
        async with self.lock:
            self.tasks[task_id] = Task(
                id=task_id,
                status=TaskStatus.CREATED,
                created_at=datetime.now(),
                priority=min(max(1, priority), 3),  # 限制在1-3范围内
                timeout=max(30, timeout)  # 最小30秒超时
            )
            # 根据优先级插入队列
            if priority >= 3:
                self.pending_queue.insert(0, task_id)  # 高优先级插队
            else:
                self.pending_queue.append(task_id)
            await self._save_tasks()
        return task_id
    
    async def start_task(self, task_id: str) -> bool:
        async with self.lock:
            if self.current_tasks >= self.max_concurrent_tasks:
                return False
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            
            # 检查任务是否已超时
            if task.status == TaskStatus.PROCESSING and task.started_at:
                elapsed = (datetime.now() - task.started_at).total_seconds()
                if elapsed > task.timeout:
                    task.status = TaskStatus.FAILED
                    task.error = f"Task timeout after {elapsed:.1f}s (limit: {task.timeout}s)"
                    self.current_tasks -= 1
                    await self._save_tasks()
                    return False
            
            if task.status != TaskStatus.CREATED:
                return False
                
            task.status = TaskStatus.PROCESSING
            task.started_at = datetime.now()
            self.current_tasks += 1
            if task_id in self.pending_queue:
                self.pending_queue.remove(task_id)
            await self._save_tasks()
            return True
    
    async def complete_task(self, task_id: str, result: Optional[Dict] = None):
        async with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].status = TaskStatus.COMPLETED
                self.tasks[task_id].completed_at = datetime.now()
                self.tasks[task_id].result = result
                self.current_tasks -= 1
                await self._save_tasks()
    
    async def fail_task(self, task_id: str, error: str):
        async with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].status = TaskStatus.FAILED
                self.tasks[task_id].completed_at = datetime.now()
                self.tasks[task_id].error = error
                self.current_tasks -= 1
                await self._save_tasks()
    
    async def update_progress(self, task_id: str, progress: float):
        async with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].progress = progress
                await self._save_tasks()
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        return self.tasks.get(task_id)
    
    async def get_next_pending_task(self) -> Optional[str]:
        """获取下一个待处理任务ID"""
        async with self.lock:
            if self.pending_queue and self.current_tasks < self.max_concurrent_tasks:
                return self.pending_queue[0]
            return None
    
    async def recover_failed_tasks_simple(self) -> List[str]:
        """恢复失败的任务，返回恢复的任务ID列表"""
        recovered = []
        async with self.lock:
            for task_id, task in self.tasks.items():
                if task.status == TaskStatus.FAILED:
                    task.status = TaskStatus.CREATED
                    task.started_at = None
                    task.completed_at = None
                    task.error = None
                    task.progress = 0.0
                    if task_id not in self.pending_queue:
                        self.pending_queue.append(task_id)
                    recovered.append(task_id)
            if recovered:
                await self._save_tasks()
        return recovered

# 全局任务调度器实例
task_scheduler = TaskScheduler()
TASKS = {}

QWEN_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_MODEL = "qwen-plus"

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = "deepseek-chat"

# 加载 Stable Diffusion pipeline（全局只加载一次，节省显存）
sd_pipe = None
def get_sd_pipe():
    global sd_pipe
    if sd_pipe is None:
        sd_pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        if torch.cuda.is_available():
            sd_pipe = sd_pipe.to("cuda")
    return sd_pipe

def generate_storyboards_with_qwen(script: str):
    """
    调用通义千问API，将脚本拆分为分镜（画面描述+台词）。
    """
    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = f"请将以下脚本拆分为分镜，每个分镜包含画面描述和台词，输出JSON数组，每个元素包含description和dialogue字段：\n{script}"
    payload = {
        "model": QWEN_MODEL,
        "input": {
            "prompt": prompt
        }
    }
    response = requests.post(QWEN_API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Qwen API error: {response.text}")
    try:
        result = response.json()
        # 假设返回体中有 'output' 字段，且为JSON字符串
        output = result.get('output', '{}')
        # 兼容直接返回JSON对象
        if isinstance(output, str):
            storyboards = json.loads(output)
        else:
            storyboards = output
        return storyboards
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qwen API parse error: {str(e)}")

def generate_storyboards_with_deepseek(script: str):
    """
    调用 DeepSeek API，将脚本拆分为分镜（画面描述+台词）。
    """
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = f"请将以下脚本拆分为分镜，每个分镜包含画面描述和台词，输出JSON数组，每个元素包含description和dialogue字段：\n{script}"
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        print(f"[DeepSeek API Error] status={response.status_code}, text={response.text}")
        raise HTTPException(status_code=500, detail=f"DeepSeek API error: {response.text}")
    try:
        result = response.json()
        print(f"[DeepSeek API Raw Result]: {result}")
        content = result.get('choices', [{}])[0].get('message', {}).get('content', '{}')
        print(f"[DeepSeek API Content]: {content}")
        if not content or content.strip() == '':
            raise Exception("DeepSeek API 返回内容为空，请检查API Key、额度、请求参数或服务状态。")
        # 去除 markdown 代码块包裹
        content_clean = re.sub(r'^```json\s*|```$', '', content.strip(), flags=re.MULTILINE)
        print(f"[DeepSeek API Content Cleaned]: {content_clean}")
        storyboards = json.loads(content_clean)
        print(f"[DEBUG] storyboards type after first loads: {type(storyboards)}")
        print(f"[DEBUG] storyboards value after first loads: {storyboards}")
        if isinstance(storyboards, str):
            print("[DEBUG] storyboards is str, try json.loads again")
            storyboards = json.loads(storyboards)
            print(f"[DEBUG] storyboards type after second loads: {type(storyboards)}")
            print(f"[DEBUG] storyboards value after second loads: {storyboards}")
        return storyboards
    except Exception as e:
        print(f"[DeepSeek API Parse Error] content={content if 'content' in locals() else 'N/A'}")
        raise HTTPException(status_code=500, detail=f"DeepSeek API parse error: {str(e)}")

def extract_json_from_text(text):
    """
    尝试从文本中提取 JSON 数组，支持多种常见格式。
    """
    # 1. 优先提取 ```json ... ``` 代码块
    match = re.search(r"```json[\s\S]*?(\[.*?\])\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    # 2. 其次提取 ``` ... ``` 代码块
    match = re.search(r"```[\s\S]*?(\[.*?\])\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    # 3. 直接查找第一个 [ 到最后一个 ]
    match = re.search(r"(\[.*\])", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    raise Exception("Qwen API 返回内容未找到 JSON 代码块")

async def generate_storyboards_mixed(script: str):
    """
    先用 DeepSeek 生成分镜，再用 Qwen 润色/补充。
    """
    # Step 1: DeepSeek 生成初稿
    ds_storyboards = generate_storyboards_with_deepseek(script)
    # Step 2: Qwen 润色/补充
    # 直接将分镜JSON作为prompt传给Qwen
    qwen_prompt = f"请对以下分镜（每个分镜包含画面描述和台词）进行润色和优化，输出优化后的JSON数组：\n{json.dumps(ds_storyboards, ensure_ascii=False)}"
    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": QWEN_MODEL,
        "input": {
            "prompt": qwen_prompt
        }
    }
    response = requests.post(QWEN_API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Qwen API error: {response.text}")
    try:
        result = response.json()
        output = result.get('output', '{}')
        # 新增：如果 output 是 dict 且有 text 字段，提取 text
        if isinstance(output, dict) and 'text' in output:
            text = output['text']
            print(f"[Qwen API Output text]: {text}")
            # 使用更健壮的 JSON 提取逻辑
            storyboards = extract_json_from_text(text)
        else:
            # 兼容原有直接 JSON 字符串
            if isinstance(output, str):
                storyboards = json.loads(output)
            else:
                storyboards = output
        return storyboards
    except Exception as e:
        print(f"[Qwen API Parse Error] output={output if 'output' in locals() else 'N/A'}")
        raise HTTPException(status_code=500, detail=f"Qwen API parse error: {str(e)}")

# 统一风格列表，和前端一致
AVAILABLE_STYLES = [
    {"value": "realistic", "label": "写实风格", "desc": "真实写实，适合生活、风景等场景"},
    {"value": "anime", "label": "动漫风格", "desc": "二次元、日漫、卡通等风格"},
    {"value": "illustration", "label": "插画风格", "desc": "艺术插画、绘本、创意风格"},
    {"value": "oil-painting", "label": "油画风格", "desc": "油画质感，艺术氛围浓厚"},
    {"value": "cyberpunk", "label": "赛博朋克", "desc": "未来科技、霓虹、朋克风格"},
    {"value": "chinese-style", "label": "中国风", "desc": "国潮、水墨、古风等中国元素"}
]

def validate_style(style: str):
    valid_values = [s["value"] for s in AVAILABLE_STYLES]
    if not style:
        return valid_values[0]
    if style not in valid_values:
        raise HTTPException(status_code=400, detail=f"不支持的图片风格: {style}. 可选: {valid_values}")
    return style

def get_job_id():
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def get_job_dir(job_id):
    path = os.path.join('outputs', job_id)
    os.makedirs(path, exist_ok=True)
    return path

def get_image_dir(job_id):
    path = os.path.join(get_job_dir(job_id), 'images')
    os.makedirs(path, exist_ok=True)
    return path

def get_audio_dir(job_id):
    path = os.path.join(get_job_dir(job_id), 'audios')
    os.makedirs(path, exist_ok=True)
    return path

def get_bgm_dir(job_id):
    path = os.path.join(get_job_dir(job_id), 'bgm')
    os.makedirs(path, exist_ok=True)
    return path

def get_script_path(job_id):
    return os.path.join(get_job_dir(job_id), 'script.txt')

def get_video_path(job_id):
    return os.path.join(get_job_dir(job_id), 'final_video.mp4')

async def generate_images_from_storyboards_async(storyboards: list, job_id: str, style: str = "realistic", size: str = "1024x1024", progress_callback=None):
    """
    根据分镜异步生成图片(使用通义万相API)
    Args:
        storyboards: 分镜列表，包含description、mood、keywords等字段
        job_id: 任务ID
        style: 图片风格
        size: 图片尺寸
        progress_callback: 进度回调函数
    Returns:
        (image_paths, errors): 图片路径列表和错误列表
    """
    output_dir = get_image_dir(job_id)
    
    # 初始化通义万相图片生成器
    image_generator = ImageGenerator(WANXIANG_API_KEY)
    
    # 构建分镜数据
    segments = []
    for scene in storyboards:
        segments.append({
            "description": scene.get("description", ""),
            "mood": scene.get("mood", "general"),
            "keywords": scene.get("keywords", []),
            "duration": scene.get("duration", 5)
        })
    
    # 生成图片
    image_paths = []
    errors = []
    try:
        results = await image_generator.generate_images(segments)
        
        # 下载并保存图片
        for idx, result in enumerate(results):
            if "image_url" in result:
                image_url = result["image_url"]
                image_path = os.path.join(output_dir, f"scene_{idx+1}.png")
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(image_url) as resp:
                        if resp.status == 200:
                            with open(image_path, 'wb') as f:
                                f.write(await resp.read())
                            image_paths.append(image_path)
                        else:
                            errors.append(f"scene_{idx+1}: 图片下载失败 HTTP {resp.status}")
            elif "image_error" in result:
                errors.append(f"scene_{idx+1}: {result['image_error']}")
            
            if progress_callback:
                progress_callback(idx+1, len(segments))
    except Exception as e:
        errors.append(f"图片生成失败: {str(e)}")
    
    return image_paths, errors

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

async def generate_audio_async(text, output_path, voice="zh-CN-XiaoxiaoNeural"):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

async def generate_audios_from_storyboards_async(storyboards: List[dict], job_id: str, voice="zh-CN-XiaoxiaoNeural", use_aliyun: bool = False, progress_callback=None, image_count=None):
    output_dir = get_audio_dir(job_id)
    ensure_dir(output_dir)
    # 只取前 image_count 个分镜生成音频
    if image_count is not None:
        storyboards = storyboards[:image_count]
    texts = [scene.get("dialogue", "") for scene in storyboards]
    audio_paths, errors = await generate_audio_batch_async(texts, output_dir, voice, use_aliyun)
    if progress_callback:
        for idx in range(len(texts)):
            progress_callback(idx+1, len(texts))
    return audio_paths, errors

def natural_key(s):
    """用于自然排序的key，支持文件名中的数字排序。"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\\d+)', s)]

def generate_video_from_images_audios(image_paths: list, audio_paths: list, job_id: str, output_size=(720, 1280), fps=25, max_duration=120):
    """
    根据图片和音频路径合成竖屏短视频，自动均分时长，总时长不超过max_duration秒。
    """
    # 修正排序，确保自然顺序
    image_paths = sorted(image_paths, key=natural_key)
    audio_paths = sorted(audio_paths, key=natural_key)
    n = min(len(image_paths), len(audio_paths))
    if n == 0:
        raise Exception("无可用图片或音频，无法合成视频")
    clips = []
    audio_clips = []
    for img_path, audio_path in zip(image_paths[:n], audio_paths[:n]):
        audio_clip = AudioFileClip(audio_path)
        duration = audio_clip.duration
        img_clip = ImageClip(img_path).with_duration(duration).with_audio(audio_clip)
        img_clip = img_clip.resized(new_size=output_size)
        clips.append(img_clip)
        audio_clips.append(audio_clip)
    final_clip = concatenate_videoclips(clips, method="compose")
    # 拼接所有音频，强制赋值给 final_clip.audio，确保音轨完整
    if audio_clips:
        from moviepy.audio.AudioClip import CompositeAudioClip as CAudioClip
        composite_audio = CAudioClip(audio_clips)
        final_clip = final_clip.with_audio(composite_audio)
    video_path = get_video_path(job_id)
    final_clip.write_videofile(video_path, fps=fps, audio_codec="aac", threads=2, logger='bar')
    return video_path

MAX_SCRIPT_LENGTH = 2000  # 最大脚本长度
MAX_STORYBOARDS = 20     # 最大分镜数

# 校验脚本内容
def validate_script(content: bytes):
    if not content:
        raise HTTPException(status_code=400, detail="脚本内容不能为空")
    try:
        script = content.decode("utf-8")
    except Exception:
        raise HTTPException(status_code=400, detail="脚本文件必须为UTF-8编码文本")
    if len(script) > MAX_SCRIPT_LENGTH:
        raise HTTPException(status_code=400, detail=f"脚本内容过长，限制{MAX_SCRIPT_LENGTH}字内")
    return script

# 校验分镜结构
def validate_storyboards(storyboards):
    """
    校验分镜列表，自动过滤掉 description 或 dialogue 为空的分镜。
    """
    if not isinstance(storyboards, list):
        raise HTTPException(status_code=400, detail="分镜格式错误，应为列表")
    filtered = []
    for idx, scene in enumerate(storyboards):
        desc = scene.get("description", "").strip()
        dial = scene.get("dialogue", "").strip()
        if not desc or not dial:
            # 跳过 description 或 dialogue 为空的分镜
            continue
        filtered.append({"description": desc, "dialogue": dial})
    if not filtered:
        raise HTTPException(status_code=400, detail="所有分镜均无效，请检查脚本或分镜生成API输出")
    return filtered

# 校验 job_id 合法性
JOB_ID_PATTERN = re.compile(r"^[a-f0-9\-]{36}$")
def validate_job_id(job_id):
    if not JOB_ID_PATTERN.match(job_id):
        raise HTTPException(status_code=400, detail="job_id 非法")
    return job_id

API_TOKEN = os.getenv("API_TOKEN")
def verify_token(x_api_token: str = Header(...)):
    if not API_TOKEN or x_api_token != API_TOKEN:
        raise HTTPException(status_code=401, detail="无效的API Token")

@app.post("/generate")
@api_response
@op_log("生成视频")
async def generate(file: UploadFile = File(...), dependencies=[Depends(verify_token)]):
    content = await file.read()
    script = validate_script(content)
    job_id = get_job_id()
    job_dir = get_job_dir(job_id)
    script_path = get_script_path(job_id)
    try:
        logging.info(f"[TRACE] 保存脚本到 {script_path}")
        with open(script_path, "wb") as f_out:
            f_out.write(content)
        logging.info("[TRACE] 脚本保存成功")
    except Exception as e:
        logging.error(f"[ERROR] 脚本保存失败: {e}")
        return {"status": "failed", "error": f"脚本保存失败: {str(e)}", "trace": traceback.format_exc()}
    TASKS[job_id] = {"status": "processing", "progress": 0, "video_path": None, "error": None}
    return {"job_id": job_id, "status": "processing", "progress": 0, "video_path": None, "error": None}

@app.get("/result/{job_id}")
@api_response
def result(job_id: str, dependencies=[Depends(verify_token)]):
    if job_id not in TASKS:
        return {"error": "Invalid job_id"}
    return {"job_id": job_id, "status": TASKS[job_id]["status"]}

@app.post("/generate_storyboards")
@api_response
async def generate_storyboards(file: UploadFile = File(...), dependencies=[Depends(verify_token)]):
    script = (await file.read()).decode("utf-8")
    storyboards = generate_storyboards_with_qwen(script)
    return {"storyboards": storyboards}

@app.post("/generate_storyboards_mixed")
@api_response
async def generate_storyboards_mixed_api(file: UploadFile = File(...), dependencies=[Depends(verify_token)]):
    script = (await file.read()).decode("utf-8")
    storyboards = await generate_storyboards_mixed(script)
    return {"storyboards": storyboards}

@app.post("/generate_images")
@api_response
async def generate_images(
    storyboards: List[dict] = Body(..., example=[{"description": "阳光明媚的公园，孩子们在草地上奔跑。", "dialogue": "今天的天气真好，我们一起去玩吧！"}]),
    job_id: str = Body(..., example="your_job_id"),
    style: str = Body("realistic", example="realistic"),
    size: str = Body("1024x1024", example="1024x1024"),
    dependencies=[Depends(verify_token)]
):
    validate_job_id(job_id)
    validate_storyboards(storyboards)
    style = validate_style(style)
    image_paths, errors = await generate_images_from_storyboards_async(storyboards, job_id, style, size)
    return {"image_paths": image_paths, "errors": errors, "style": style, "size": size}

@app.post("/generate_audios")
@api_response
async def generate_audios(
    storyboards: List[dict] = Body(..., example=[{"description": "阳光明媚的公园，孩子们在草地上奔跑。", "dialogue": "今天的天气真好，我们一起去玩吧！"}]),
    job_id: str = Body(..., example="your_job_id"),
    voice: str = Body("zh-CN-XiaoxiaoNeural", example="zh-CN-XiaoxiaoNeural"),
    use_aliyun: bool = Body(False, example=False),
    dependencies=[Depends(verify_token)]
):
    validate_job_id(job_id)
    validate_storyboards(storyboards)
    audio_paths, errors = await generate_audios_from_storyboards_async(storyboards, job_id, voice, use_aliyun)
    return {"audio_paths": audio_paths, "errors": errors}

@app.post("/generate_video")
@api_response
async def generate_video(
    image_paths: list = Body(..., example=["outputs/your_job_id/images/scene_1.png"]),
    audio_paths: list = Body(..., example=["outputs/your_job_id/audios/scene_1.mp3"]),
    job_id: str = Body(..., example="your_job_id"),
    subtitles: Optional[List[Dict]] = Body(None, example=[{"text": "第一句台词", "start_time": 0.0, "end_time": 3.5}]),
    bg_music_path: Optional[str] = Body(None, example="data/music_library/happy_bgm.mp3"),
    dependencies=[Depends(verify_token)]
):
    """
    输入图片、音频路径列表及job_id，合成竖屏短视频(支持字幕)，返回视频路径。
    Args:
        image_paths: 图片路径列表
        audio_paths: 音频路径列表
        job_id: 任务ID
        subtitles: 字幕列表(可选)
    Returns:
        合成后的视频路径
    """
    try:
        output_dir = get_job_dir(job_id)
        os.makedirs(output_dir, exist_ok=True)
        video_path = get_video_path(job_id)
        
        # 使用FFmpeg合成视频
        from modules.video_merge import merge_video_with_ffmpeg
        video_path = merge_video_with_ffmpeg(
            image_paths=image_paths,
            audio_paths=audio_paths,
            output_path=video_path,
            subtitles=subtitles,
            bg_music_path=bg_music_path,
            resolution="720x1280",
            fps=30
        )
        
        return {"video_path": video_path}
    except Exception as e:
        logging.error(f"视频合成失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"视频合成失败: {str(e)}")

@app.post("/generate_video_from_script")
@api_response
async def generate_video_from_script(
    file: UploadFile = File(...),
    style: str = "realistic",
    voice: str = "zh-CN-XiaoxiaoNeural",
    subtitle_file: UploadFile = File(None),
    cover_image: UploadFile = File(None),
    overlay: dict = Body(None, example={"image": "logo.png", "start": 1.0, "end": 3.0, "position": "10:10"}),
    dependencies=[Depends(verify_token)]
):
    content = await file.read()
    script = validate_script(content)
    script_str = script.decode("utf-8") if isinstance(script, bytes) else script
    style = validate_style(style)
    job_id = get_job_id()
    job_dir = get_job_dir(job_id)
    script_path = get_script_path(job_id)
    try:
        logging.info(f"[TRACE] 保存脚本到 {script_path}")
        with open(script_path, "wb") as f_out:
            f_out.write(content)
        logging.info("[TRACE] 脚本保存成功")
    except Exception as e:
        logging.error(f"[ERROR] 脚本保存失败: {e}")
        return {"status": "failed", "error": f"脚本保存失败: {str(e)}", "trace": traceback.format_exc()}
    # 分镜生成
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    storyboards = loop.run_until_complete(generate_storyboards_mixed(script_str))
    storyboards = validate_storyboards(storyboards)
    # 图片生成
    image_paths, image_errors = loop.run_until_complete(generate_images_from_storyboards_async(storyboards, job_id, style))
    image_count = len(image_paths)
    # 音频生成，严格按图片数量生成
    audio_paths, audio_errors = loop.run_until_complete(generate_audios_from_storyboards_async(storyboards, job_id, voice, use_aliyun=False, image_count=image_count))
    # 视频合成及后处理
    try:
        logging.info(f"[TRACE] 开始视频合成 job_id={job_id}")
        video_path = generate_video_from_images_audios(image_paths, audio_paths, job_id)
        final_path = video_path
        logging.info(f"[TRACE] 视频合成完成 job_id={job_id}，路径: {final_path}")
        with TASKS_LOCK:
            TASKS[job_id]["status"] = "finished"
            TASKS[job_id]["video_path"] = final_path
            TASKS[job_id]["progress"] = 100
            TASKS[job_id]["stage"] = "完成"
        logging.info(f"[TRACE] 任务完成 job_id={job_id}")
    except Exception as e:
        logging.error(f"[ERROR] 视频合成失败: {e}\n{traceback.format_exc()}")
        with TASKS_LOCK:
            TASKS[job_id]["status"] = "failed"
            TASKS[job_id]["error"] = f"视频合成失败: {str(e)}\n{traceback.format_exc()}"
            TASKS[job_id]["progress"] = 0
            TASKS[job_id]["stage"] = "视频合成"
    return {"job_id": job_id, "status": TASKS[job_id]["status"], "progress": TASKS[job_id]["progress"], "video_path": TASKS[job_id]["video_path"], "error": TASKS[job_id]["error"], "style": style, "voice": voice}

EXECUTOR = ThreadPoolExecutor(max_workers=2)
TASKS_LOCK = threading.Lock()

@app.post("/generate_video_from_script_async")
@api_response
async def generate_video_from_script_async(
    file: UploadFile = File(...),
    style: str = "realistic",
    voice: str = "zh-CN-XiaoxiaoNeural",
    subtitle_file: UploadFile = File(None),
    cover_image: UploadFile = File(None),
    overlay: list = Body(None, example=[{"image": "logo.png", "start": 1.0, "end": 3.0, "position": "10:10"}]),
    use_ai_storyboard: bool = Body(True, example=True),  # 新增参数，控制是否用AI分镜
    dependencies=[Depends(verify_token)]
):
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
    content = await file.read()
    try:
        logging.info("[TRACE] 开始校验脚本内容")
        script = validate_script(content)
        logging.info("[TRACE] 脚本校验通过")
    except Exception as e:
        logging.error(f"[ERROR] 脚本校验失败: {e}")
        return {"status": "failed", "error": f"脚本校验失败: {str(e)}", "trace": traceback.format_exc()}
    try:
        logging.info(f"[TRACE] 校验风格: {style}")
        style = validate_style(style)
        logging.info(f"[TRACE] 风格校验通过: {style}")
    except Exception as e:
        logging.error(f"[ERROR] 风格校验失败: {e}")
        return {"status": "failed", "error": f"风格校验失败: {str(e)}", "trace": traceback.format_exc()}
    job_id = get_job_id()
    job_dir = get_job_dir(job_id)
    script_path = get_script_path(job_id)
    try:
        logging.info(f"[TRACE] 保存脚本到 {script_path}")
        with open(script_path, "wb") as f_out:
            f_out.write(content)
        logging.info("[TRACE] 脚本保存成功")
    except Exception as e:
        logging.error(f"[ERROR] 脚本保存失败: {e}")
        return {"status": "failed", "error": f"脚本保存失败: {str(e)}", "trace": traceback.format_exc()}
    with TASKS_LOCK:
        TASKS[job_id] = {"status": "pending", "video_path": None, "error": None, "progress": 0, "style": style, "voice": voice, "stage": "等待中"}

    def segments_to_storyboards(segments: list) -> list:
        # 可扩展字段，如mood/keywords等
        return [{"description": seg, "dialogue": seg} for seg in segments]

    def task():
        try:
            with TASKS_LOCK:
                TASKS[job_id]["status"] = "processing"
                TASKS[job_id]["progress"] = 5
                TASKS[job_id]["stage"] = "分镜生成"
            # 分镜生成
            try:
                logging.info(f"[TRACE] 开始分镜生成 job_id={job_id}")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                script_str = script.decode("utf-8") if isinstance(script, bytes) else script
                # 集成parse_script，分段并打印日志
                segments = parse_script(script_str)
                logging.info(f"[DEBUG] 脚本分段结果: {segments}")
                if use_ai_storyboard:
                    storyboards = loop.run_until_complete(generate_storyboards_mixed(script_str))
                    logging.info(f"[TRACE] 分镜生成完成 job_id={job_id} (AI模式)")
                else:
                    storyboards = segments_to_storyboards(segments)
                    logging.info(f"[TRACE] 分镜生成完成 job_id={job_id} (直接分段模式)")
                storyboards = validate_storyboards(storyboards)
                logging.info(f"[TRACE] 分镜校验通过 job_id={job_id}")
            except Exception as e:
                logging.error(f"[ERROR] 分镜生成失败: {e}\n{traceback.format_exc()}")
                with TASKS_LOCK:
                    TASKS[job_id]["status"] = "failed"
                    TASKS[job_id]["error"] = f"分镜生成失败: {str(e)}\n{traceback.format_exc()}"
                    TASKS[job_id]["progress"] = 0
                    TASKS[job_id]["stage"] = "分镜生成"
                return
            with TASKS_LOCK:
                TASKS[job_id]["progress"] = 20
                TASKS[job_id]["stage"] = "音频生成"
            # 音频生成
            # try:
            #     logging.info(f"[TRACE] 开始音频生成 job_id={job_id}")
            #     def progress_callback_audio(done, total):
            #         with TASKS_LOCK:
            #             TASKS[job_id]["progress"] = 40 + int(30 * done / total)
            #             TASKS[job_id]["stage"] = "音频生成"
            #     texts = [scene.get("dialogue", "") for scene in storyboards]
            #     for idx, text in enumerate(texts):
            #         logging.info(f"[AUDIO] 生成第{idx+1}段音频，文本: {text}")
            #     audio_paths, audio_errors = loop.run_until_complete(generate_audios_from_storyboards_async(storyboards, job_id, voice, use_aliyun=False, progress_callback=progress_callback_audio))
            #     for idx, path in enumerate(audio_paths):
            #         logging.info(f"[AUDIO] 第{idx+1}段音频已保存: {path}")
            #     if audio_errors:
            #         for err in audio_errors:
            #             logging.error(f"[AUDIO] 音频生成错误: {err}")
            #     logging.info(f"[TRACE] 音频生成完成 job_id={job_id}，音频数: {len(audio_paths)}，错误: {audio_errors}")
            #     with TASKS_LOCK:
            #         TASKS[job_id]["status"] = "finished"
            #         TASKS[job_id]["progress"] = 100
            #         TASKS[job_id]["stage"] = "音频生成完成"
            #         if audio_errors:
            #             TASKS[job_id]["error"] = f"音频生成失败: {audio_errors}"
            #     logging.info(f"[TRACE] 任务完成（仅音频验证） job_id={job_id}")
            #     return
            # except Exception as e:
            #     logging.error(f"[ERROR] 音频生成失败: {e}\n{traceback.format_exc()}")
            #     with TASKS_LOCK:
            #         TASKS[job_id]["status"] = "failed"
            #         TASKS[job_id]["error"] = f"音频生成失败: {str(e)}\n{traceback.format_exc()}"
            #         TASKS[job_id]["progress"] = 0
            #         TASKS[job_id]["stage"] = "音频生成"
            #     return

            with TASKS_LOCK:
                TASKS[job_id]["progress"] = 80
                TASKS[job_id]["stage"] = "视频合成"
            # 视频合成及后处理
            try:
                logging.info(f"[TRACE] 开始视频合成 job_id={job_id}")
                video_path = generate_video_from_images_audios(image_paths, audio_paths, job_id)
                final_path = video_path
                logging.info(f"[TRACE] 视频合成完成 job_id={job_id}，路径: {final_path}")
                with TASKS_LOCK:
                    TASKS[job_id]["status"] = "finished"
                    TASKS[job_id]["video_path"] = final_path
                    TASKS[job_id]["progress"] = 100
                    TASKS[job_id]["stage"] = "完成"
                logging.info(f"[TRACE] 任务完成 job_id={job_id}")
            except Exception as e:
                logging.error(f"[ERROR] 视频合成失败: {e}\n{traceback.format_exc()}")
                with TASKS_LOCK:
                    TASKS[job_id]["status"] = "failed"
                    TASKS[job_id]["error"] = f"视频合成失败: {str(e)}\n{traceback.format_exc()}"
                    TASKS[job_id]["progress"] = 0
                    TASKS[job_id]["stage"] = "视频合成"
        except Exception as e:
            logging.error(f"[ERROR] 任务异常: {e}\n{traceback.format_exc()}")
            with TASKS_LOCK:
                TASKS[job_id]["status"] = "failed"
                TASKS[job_id]["error"] = f"任务异常: {str(e)}\n{traceback.format_exc()}"
                TASKS[job_id]["progress"] = 0
                TASKS[job_id]["stage"] = "未知异常"
    EXECUTOR.submit(task)
    return {"job_id": job_id, "status": "pending", "progress": 0, "video_path": None, "error": None, "style": style, "voice": voice, "stage": "等待中"}

@app.get("/result/{job_id}")
@api_response
def get_result(job_id: str, dependencies=[Depends(verify_token)]):
    with TASKS_LOCK:
        task = TASKS.get(job_id)
        if not task:
            return {"error": "Invalid job_id"}
        # 补全所有字段
        return {
            "job_id": job_id,
            "status": task.get("status"),
            "progress": task.get("progress", 0),
            "video_path": task.get("video_path"),
            "error": task.get("error"),
            "style": task.get("style"),
            "voice": task.get("voice"),
            "stage": task.get("stage", "")
        }

def clean_outputs_dir(outputs_dir="outputs", expire_seconds=86400):
    """
    定期清理outputs目录下，超过expire_seconds未修改的任务子目录。
    """
    while True:
        now = time.time()
        for job_id in os.listdir(outputs_dir):
            job_path = os.path.join(outputs_dir, job_id)
            if os.path.isdir(job_path):
                try:
                    mtime = os.path.getmtime(job_path)
                    if now - mtime > expire_seconds:
                        shutil.rmtree(job_path)
                except Exception as e:
                    print(f"[CLEAN] Failed to remove {job_path}: {e}")
        time.sleep(3600)  # 每小时检查一次

# 启动后台清理线程
cleaner_thread = threading.Thread(target=clean_outputs_dir, args=("outputs", 86400), daemon=True)
cleaner_thread.start()

@app.get("/available_image_styles")
@api_response
def available_image_styles():
    return {"available_styles": AVAILABLE_STYLES}

@app.get("/available_tts_voices")
@api_response
def available_tts_voices():
    # 推荐返回所有 edge-tts 支持的声音，最简单可写死
    return {"voices": ["zh-CN-XiaoxiaoNeural"]}

@app.on_event("startup")
async def startup_event():
    # 启动定时清理任务
    task_scheduler.cleanup_task = asyncio.create_task(task_scheduler._periodic_cleanup())

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error(f"[全局异常] {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "服务器内部错误，请联系管理员。"})

def print_audio_image_durations(image_paths, audio_paths):
    """调试用：输出每段音频和图片片段的时长及差异。"""
    import re
    from moviepy import AudioFileClip, ImageClip
    def natural_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\\d+)', s)]
    image_paths = sorted(image_paths, key=natural_key)
    audio_paths = sorted(audio_paths, key=natural_key)
    n = min(len(image_paths), len(audio_paths))
    audio_durations = [AudioFileClip(a).duration for a in audio_paths[:n]]
    img_durations = []
    for img_path, audio_path in zip(image_paths[:n], audio_paths[:n]):
        audio_clip = AudioFileClip(audio_path)
        img_clip = ImageClip(img_path).with_duration(audio_clip.duration).with_audio(audio_clip)
        img_durations.append(img_clip.duration)
    print('idx | audio_duration | img_duration | diff')
    for i, (ad, idur) in enumerate(zip(audio_durations, img_durations)):
        print(f'{i+1:>3} | {ad:>13.2f} | {idur:>11.2f} | {abs(ad-idur):>6.2f}')
    print('音频总时长:', sum(audio_durations))
    print('图片片段总时长:', sum(img_durations))
