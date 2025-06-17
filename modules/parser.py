import re
from loguru import logger

def parse_script(script: str):
    """
    按换行或标点分段，简单情绪/节奏识别（可扩展），返回分段文本列表。
    """
    try:
        # 按换行或句号、问号、感叹号分段
        segments = re.split(r'[\n。！？!?]+', script)
        segments = [seg.strip() for seg in segments if seg.strip()]
        logger.debug(f'分段结果: {segments}')
        return segments
    except Exception as e:
        logger.exception(f'脚本解析失败: {e}')
        return [] 