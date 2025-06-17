import os
import logging
import subprocess
import tempfile
from typing import List, Dict, Optional
from moviepy import ImageClip, AudioFileClip

# 设置日志
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

def merge_video_with_ffmpeg(
    image_paths: List[str], 
    audio_paths: List[str], 
    output_path: str,
    subtitles: Optional[List[Dict]] = None,
    bg_music_path: Optional[str] = None,
    resolution: str = '720x1280',
    fps: int = 30
) -> str:
    """
    使用FFmpeg合成视频(图像+音频+字幕)
    Args:
        image_paths: 图片路径列表
        audio_paths: 音频路径列表
        output_path: 输出视频路径
        subtitles: 字幕列表(可选)
        resolution: 视频分辨率(默认720x1280竖屏)
        fps: 视频帧率(默认30)
    Returns:
        合成后的视频路径
    """
    try:
        if not image_paths or not audio_paths:
            logging.error('输入的图片或音频路径列表为空')
            raise ValueError('图片或音频路径列表不能为空')
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. 准备输入文件列表
            concat_file = os.path.join(tmp_dir, 'concat.txt')
            with open(concat_file, 'w') as f:
                for img_path, audio_path in zip(image_paths, audio_paths):
                    if not os.path.isfile(img_path):
                        raise FileNotFoundError(f'图片文件不存在: {img_path}')
                    if not os.path.isfile(audio_path):
                        raise FileNotFoundError(f'音频文件不存在: {audio_path}')
                    
                    # 获取音频时长
                    audio_clip = AudioFileClip(audio_path)
                    duration = audio_clip.duration
                    audio_clip.close()
                    
                    # 写入concat文件
                    f.write(f"file '{img_path}'\n")
                    f.write(f"duration {duration}\n")
            
            # 2. 合成视频(图像+音频)
            temp_video = os.path.join(tmp_dir, 'temp.mp4')
            cmd = [
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                '-i', concat_file,
                '-i', audio_paths[0],  # 使用第一个音频作为主音轨
                '-filter_complex', f'[0:v]scale={resolution},setsar=1,fps={fps}[v]',
                '-map', '[v]', '-map', '1:a',
                '-c:v', 'h264_videotoolbox', '-preset', 'fast',  # 使用硬件加速
                '-threads', '4',  # 启用多线程处理
                '-movflags', '+faststart',  # 优化网络播放
                '-crf', '23', '-pix_fmt', 'yuv420p',
                temp_video
            ]
            try:
                logging.info(f'执行FFmpeg命令: {" ".join(cmd)}')
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                logging.debug(f'FFmpeg输出: {result.stdout}')
                if result.stderr:
                    logging.debug(f'FFmpeg错误输出: {result.stderr}')
            except subprocess.CalledProcessError as e:
                logging.error(f'FFmpeg命令执行失败: {e.stderr}')
                raise RuntimeError(f'视频合成失败: {e.stderr}')
            
            # 3. 添加字幕(如果提供)
            if subtitles:
                subtitle_file = os.path.join(tmp_dir, 'subtitles.srt')
                with open(subtitle_file, 'w') as f:
                    for i, sub in enumerate(subtitles):
                        f.write(f"{i+1}\n")
                        f.write(f"{_format_time(sub['start_time'])} --> {_format_time(sub['end_time'])}\n")
                        f.write(f"{sub['text']}\n\n")
                
                cmd = [
                    'ffmpeg', '-y', '-i', temp_video,
                    '-vf', f"subtitles='{subtitle_file}':force_style='Fontsize=24,PrimaryColour=&HFFFFFF&'"
                    '-c:v', 'h264_videotoolbox',  # 使用硬件加速
                    '-threads', '4',  # 启用多线程处理
                    '-c:a', 'copy', output_path
                ]
                try:
                    logging.info(f'执行FFmpeg字幕命令: {" ".join(cmd)}')
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                    logging.debug(f'FFmpeg字幕输出: {result.stdout}')
                    if result.stderr:
                        logging.debug(f'FFmpeg字幕错误输出: {result.stderr}')
                except subprocess.CalledProcessError as e:
                    logging.error(f'FFmpeg字幕命令执行失败: {e.stderr}')
                    raise RuntimeError(f'字幕添加失败: {e.stderr}')
            else:
                os.rename(temp_video, output_path)
                
            # 4. 添加背景音乐(如果提供)
            if bg_music_path:
                if not os.path.exists(bg_music_path):
                    logging.warning(f'背景音乐文件不存在: {bg_music_path}')
                else:
                    final_output = os.path.join(tmp_dir, 'final_with_bgm.mp4')
                    cmd = [
                        'ffmpeg', '-y', '-i', output_path, '-i', bg_music_path,
                        '-filter_complex', '[0:a]volume=1.0[voice];[1:a]volume=0.3[bgm];[voice][bgm]amix=inputs=2:duration=first',
                        '-c:v', 'h264_videotoolbox',  # 使用硬件加速
                        '-threads', '4',  # 启用多线程处理
                        '-movflags', '+faststart',  # 优化网络播放
                        final_output
                    ]
                    try:
                        logging.info(f'执行FFmpeg背景音乐命令: {" ".join(cmd)}')
                        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                        logging.debug(f'FFmpeg背景音乐输出: {result.stdout}')
                        if result.stderr:
                            logging.debug(f'FFmpeg背景音乐错误输出: {result.stderr}')
                    except subprocess.CalledProcessError as e:
                        logging.error(f'FFmpeg背景音乐命令执行失败: {e.stderr}')
                        raise RuntimeError(f'背景音乐添加失败: {e.stderr}')
                    os.rename(final_output, output_path)
            
            logging.info(f'视频合成成功: {output_path}')
            return output_path
            
    except subprocess.CalledProcessError as e:
        logging.error(f'FFmpeg处理失败: {e}', exc_info=True)
        raise RuntimeError(f'视频合成失败: {e}')
    except Exception as e:
        logging.error(f'视频合成失败: {e}', exc_info=True)
        raise

def _format_time(seconds: float) -> str:
    """将秒数格式化为SRT时间格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
