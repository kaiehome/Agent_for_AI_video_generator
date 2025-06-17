import subprocess
import os
import logging
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips
import whisper
from whisper.utils import get_writer
from enum import Enum
import json
from typing import List, Dict

class SubtitleStyle(Enum):
    """字幕样式枚举"""
    DEFAULT = 1
    BOLD = 2
    SHADOW = 3
    OUTLINE = 4

class VideoError(Exception):
    """视频处理错误基类"""
    def __init__(self, error_type: str, message: str, recoverable: bool = False):
        self.error_type = error_type
        self.message = message
        self.recoverable = recoverable
        super().__init__(message)

class AudioProcessingError(VideoError):
    """音频处理错误"""
    pass

class SubtitleError(VideoError):
    """字幕处理错误"""
    pass

class VideoMergeError(VideoError):
    """视频合并错误"""
    pass

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

async def add_subtitle_ffmpeg(video_path: str, output_path: str, storyboards: List[Dict] = None, 
                          audio_path: str = None, style: SubtitleStyle = SubtitleStyle.DEFAULT) -> str:
    """
    使用FFmpeg添加字幕
    Args:
        video_path: 输入视频路径
        output_path: 输出视频路径
        storyboards: 分镜列表(可选)
        audio_path: 音频路径(用于语音转文字，可选)
        style: 字幕样式
    Returns:
        处理后的视频路径
    """
    try:
        subtitle_path = os.path.splitext(video_path)[0] + '.srt'
        
        # 如果有音频路径但没有分镜，使用Whisper生成字幕
        if audio_path and not storyboards:
            model = whisper.load_model("small")
            result = model.transcribe(audio_path)
            
            # 保存为SRT格式
            writer = get_writer("srt", os.path.dirname(subtitle_path))
            writer(result, subtitle_path)
            
            # 读取生成的字幕用于样式处理
            with open(subtitle_path, 'r') as f:
                subtitles = f.read()
        elif storyboards:
            # 从分镜生成字幕
            subtitles = ''
            for i, sb in enumerate(storyboards):
                start_time = sum(s['duration'] for s in storyboards[:i])
                end_time = start_time + sb['duration']
                subtitles += f"{i+1}\n"
                subtitles += f"{_format_time(start_time)} --> {_format_time(end_time)}\n"
                subtitles += f"{sb['dialogue']}\n\n"
            
            with open(subtitle_path, 'w') as f:
                f.write(subtitles)
        else:
            raise SubtitleError("invalid_input", "Either storyboards or audio_path must be provided")
        
        # 根据样式配置FFmpeg滤镜
        style_filters = {
            SubtitleStyle.DEFAULT: "subtitles='{subtitle_path}'",
            SubtitleStyle.BOLD: "subtitles='{subtitle_path}':force_style='Fontsize=24,PrimaryColour=&HFFFFFF&,Bold=1'",
            SubtitleStyle.SHADOW: "subtitles='{subtitle_path}':force_style='Fontsize=24,PrimaryColour=&HFFFFFF&,Shadow=2'",
            SubtitleStyle.OUTLINE: "subtitles='{subtitle_path}':force_style='Fontsize=24,PrimaryColour=&HFFFFFF&,OutlineColour=&H000000&,BorderStyle=1'"
        }
        
        # 使用FFmpeg添加字幕
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vf', style_filters[style].format(subtitle_path=subtitle_path),
            '-c:a', 'copy',
            '-y',
            output_path
        ]
        
        proc = await asyncio.create_subprocess_exec(*cmd)
        await proc.wait()
        
        return output_path
    except SubtitleError:
        raise
    except Exception as e:
        logging.error(f"Failed to add subtitles: {str(e)}")
        raise SubtitleError("processing_error", f"Subtitle processing failed: {str(e)}", True)


def add_cover_ffmpeg(input_video, cover_image, output_video):
    """
    为视频添加封面（mp4元数据/首帧）。
    """
    try:
        cmd = [
            "ffmpeg", "-y", "-i", input_video, "-i", cover_image,
            "-map", "0", "-map", "1", "-c", "copy", "-disposition:1", "attached_pic", output_video
        ]
        subprocess.run(cmd, check=True)
        logging.info(f"封面添加成功: {output_video}")
    except Exception as e:
        logging.error(f"封面添加失败: {e}", exc_info=True)
        raise


def add_overlay_ffmpeg(input_video, overlay_image, start_time, end_time, position, output_video):
    """
    为视频指定时间段叠加动态元素（如logo/贴纸/动画）。
    position: (x, y) 坐标字符串，如 '10:10'。
    """
    try:
        duration = float(end_time) - float(start_time)
        filter_str = (
            f"[1:v]format=rgba,fade=t=in:st=0:d=0.3:alpha=1,fade=t=out:st={duration-0.3}:d=0.3:alpha=1[ov];"
            f"[0:v][ov]overlay={position}:enable='between(t,{start_time},{end_time})'"
        )
        cmd = [
            "ffmpeg", "-y", "-i", input_video, "-i", overlay_image,
            "-filter_complex", filter_str,
            "-c:a", "copy", output_video
        ]
        subprocess.run(cmd, check=True)
        logging.info(f"动态元素添加成功: {output_video}")
    except Exception as e:
        logging.error(f"动态元素添加失败: {e}", exc_info=True)
        raise