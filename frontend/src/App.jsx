import React, { useState, useRef, useEffect } from "react";
import { submitScript, getJobStatus, getImageStyles, getTTSVoices } from "./api";

const gradientBg = "linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)";
const cardBg = "rgba(255,255,255,0.98)";
const sectionBg = "linear-gradient(90deg, #e0e7ff 0%, #f7f8fa 100%)";
const accent = "#7f53ac";
const accent2 = "#43cea2";
const errorColor = "#e53e3e";

const STATIC_STYLES = [
  "realistic",
  "anime",
  "illustration",
  "oil-painting",
  "cyberpunk",
  "chinese-style"
];
const STATIC_VOICES = [
  { value: "zh-CN-XiaoxiaoNeural", label: "女声·温柔（小晓）", desc: "微软官方，温柔自然，适合解说/旁白" },
  { value: "zh-CN-XiaohanNeural", label: "女声·活泼（小涵）", desc: "微软官方，活泼有感染力，适合故事/广告" },
  { value: "zh-CN-XiaomoNeural", label: "女声·知性（小默）", desc: "微软官方，知性成熟，适合新闻/讲解" },
  { value: "zh-CN-YunxiNeural", label: "男声·温和（云希）", desc: "微软官方，温和亲切，适合解说/旁白" },
  { value: "zh-CN-YunyangNeural", label: "男声·磁性（云扬）", desc: "微软官方，磁性低沉，适合广告/纪录片" },
  { value: "zh-CN-XiaoxuanNeural", label: "男声·青年（小轩）", desc: "微软官方，青年活力，适合故事/娱乐" }
];

// 默认占位图片和音频
const DEFAULT_STYLE_IMG = require("./assets/styles/default.jpg");
const DEFAULT_VOICE_MP3 = require("./assets/voices/default.mp3");

// 统一前端风格选项，确保与后端 HuggingFace Spaces 顺序和含义一致
const SPACE_STYLES = [
  { value: "realistic", label: "写实风格", desc: "真实写实，适合生活、风景等场景" },
  { value: "anime", label: "动漫风格", desc: "二次元、日漫、卡通等风格" },
  { value: "illustration", label: "插画风格", desc: "艺术插画、绘本、创意风格" },
  { value: "oil-painting", label: "油画风格", desc: "油画质感，艺术氛围浓厚" },
  { value: "cyberpunk", label: "赛博朋克", desc: "未来科技、霓虹、朋克风格" },
  { value: "chinese-style", label: "中国风", desc: "国潮、水墨、古风等中国元素" }
];

// 颜色主题
const THEME = {
  primary: '#5B8CFF', // 柔和蓝
  accent: '#A389F4', // 浅紫
  bg: '#F7F8FA',     // 浅灰背景
  border: '#E0E7FF', // 浅蓝边框
  text: '#222',
  desc: '#888',
  selectBg: '#fff',
  selectHover: '#F0F4FF',
  selectActive: '#E6EDFF',
};

// 进度条颜色和样式
const PROGRESS_THEME = {
  bar: '#5B8CFF',
  bg: '#E0E7FF',
  text: '#444',
  stage: '#A389F4',
};

// 动态获取风格图片（本地优先，远程次之，最后兜底）
function getStyleImg(style) {
  // 1. 如果 style 对象有 imageUrl 字段（后端返回远程URL），优先用远程
  if (typeof style === 'object' && style.imageUrl) return style.imageUrl;
  // 2. 仅支持 default.jpg，其他情况返回默认
  if ((typeof style === 'string' && style === 'realistic') || (typeof style === 'object' && style.value === 'realistic')) {
    return DEFAULT_STYLE_IMG;
  }
  return DEFAULT_STYLE_IMG;
}
// 动态获取声音试听音频（本地优先，远程次之，最后兜底）
function getVoiceMp3(voice) {
  if (typeof voice === 'object' && voice.audioUrl) return voice.audioUrl;
  // 仅支持 default.mp3，其他情况返回默认
  if ((typeof voice === 'string' && voice === 'zh-CN-XiaoxiaoNeural') || (typeof voice === 'object' && voice.value === 'zh-CN-XiaoxiaoNeural')) {
    return DEFAULT_VOICE_MP3;
  }
  return DEFAULT_VOICE_MP3;
}

const IMAGE_SIZES = [
  { value: "512x512", label: "512x512 (正方形)" },
  { value: "768x768", label: "768x768 (正方形)" },
  { value: "1024x1024", label: "1024x1024 (正方形)" },
  { value: "1024x576", label: "1024x576 (横向)" },
  { value: "576x1024", label: "576x1024 (竖向)" },
  { value: "自定义", label: "自定义尺寸..." }
];

function App() {
  const [script, setScript] = useState("");
  const [jobId, setJobId] = useState("");
  const [status, setStatus] = useState("");
  const [videoUrl, setVideoUrl] = useState("");
  const [error, setError] = useState("");
  const [progress, setProgress] = useState(0);
  const [style, setStyle] = useState("");
  const [voice, setVoice] = useState("");
  const [styles, setStyles] = useState([]);
  const [voices, setVoices] = useState([]);
  const [subtitleFile, setSubtitleFile] = useState(null);
  const [coverImage, setCoverImage] = useState(null);
  const [overlays, setOverlays] = useState([
    { image: null, start: "", end: "", position: "" }
  ]);
  const [formError, setFormError] = useState("");
  const intervalRef = useRef(null);
  const [styleLoadError, setStyleLoadError] = useState(false);
  const [voiceLoadError, setVoiceLoadError] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [stage, setStage] = useState('');
  const [size, setSize] = useState("1024x1024");
  const [customSize, setCustomSize] = useState("");
  const [useAiStoryboard, setUseAiStoryboard] = useState(true);

  useEffect(() => {
    getImageStyles().then(res => {
      if (res && res.length > 0) setStyles(res);
      else setStyles(STATIC_STYLES);
    }).catch(() => {
      setStyleLoadError(true);
      setStyles(STATIC_STYLES);
    });
    getTTSVoices().then(res => {
      if (res && res.length > 0) setVoices(res);
      else setVoices(STATIC_VOICES);
    }).catch(() => {
      setVoiceLoadError(true);
      setVoices(STATIC_VOICES);
    });
  }, []);

  // 校验逻辑
  const validate = () => {
    if (!script.trim()) return "脚本不能为空";
    if (!style || styleLoadError) return "请选择图片风格（如下拉不可用请检查后端服务）";
    if (!voice || voiceLoadError) return "请选择TTS声音（如下拉不可用请检查后端服务）";
    for (let i = 0; i < overlays.length; i++) {
      const o = overlays[i];
      if (o.image || o.start || o.end || o.position) {
        if (!o.image) return `第${i+1}个动态元素图片未上传`;
        if (!o.start || isNaN(Number(o.start)) || Number(o.start) < 0) return `第${i+1}个动态元素起始时间无效`;
        if (!o.end || isNaN(Number(o.end)) || Number(o.end) <= Number(o.start)) return `第${i+1}个动态元素结束时间无效`;
      }
    }
    return "";
  };

  const handleSubmit = async () => {
    setError("");
    setFormError("");
    setVideoUrl("");
    setStatus("提交中...");
    setProgress(0);
    const err = validate();
    if (err) {
      setFormError(err);
      setStatus("");
      return;
    }
    let finalSize = size === "自定义" ? customSize : size;
    if (!/^\d+x\d+$/.test(finalSize)) {
      setFormError("图片尺寸格式应为 1024x1024 或 768x768 等");
      setStatus("");
      return;
    }
    try {
      const id = await submitScript({
        script,
        style,
        voice,
        subtitleFile,
        coverImage,
        overlays: overlays.filter(o => o.image),
        size: finalSize,
        use_ai_storyboard: useAiStoryboard
      });
      setJobId(id);
      setStatus("处理中...");
      setProgress(5);
      setStage('分镜生成中...');
      await pollStatus(id);
    } catch (e) {
      setError("提交失败：" + e.message);
      setStatus("");
    }
  };

  const pollStatus = async (id) => {
    intervalRef.current = setInterval(async () => {
      try {
        const res = await getJobStatus(id);
        setStatus(res.status);
        if (typeof res.progress === "number") {
          setProgress(res.progress);
        }
        if (res.stage) {
          setStage(res.stage);
        }
        if (res.status === "finished") {
          setProgress(100);
          setStage(res.stage || '生成完成！');
          setVideoUrl(res.video_path);
          setIsGenerating(false);
          clearInterval(intervalRef.current);
        }
        if (res.status === "failed") {
          setError("生成失败：" + (res.error || "未知错误"));
          setProgress(0);
          setIsGenerating(false);
          clearInterval(intervalRef.current);
        }
      } catch (e) {
        setError("查询失败：" + e.message);
        setProgress(0);
        setIsGenerating(false);
        clearInterval(intervalRef.current);
      }
    }, 2000);
  };

  // 多overlay操作
  const handleOverlayChange = (idx, key, value) => {
    setOverlays(prev => prev.map((o, i) => i === idx ? { ...o, [key]: value } : o));
  };
  const addOverlay = () => setOverlays([...overlays, { image: null, start: "", end: "", position: "" }]);
  const removeOverlay = idx => setOverlays(overlays.length > 1 ? overlays.filter((_, i) => i !== idx) : overlays);

  // 卡片内温和提示
  const renderServiceWarning = () => (styleLoadError || voiceLoadError) && (
    <div style={{
      background: "#fff7e6",
      color: "#b26a00",
      textAlign: "center",
      padding: 10,
      fontWeight: 600,
      letterSpacing: 1,
      fontSize: 15,
      borderRadius: 10,
      marginBottom: 16,
      border: "1px solid #ffe0b2"
    }}>
      ⚠️ 与后端服务连接异常，部分选项为默认内容，功能不受影响。
    </div>
  );

  return (
    <div style={{
      minHeight: "100vh",
      background: gradientBg,
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      padding: 0,
      flexDirection: "column"
    }}>
      <div style={{
        maxWidth: 810,
        width: "98vw",
        background: cardBg,
        borderRadius: 24,
        boxShadow: "0 8px 32px rgba(127,83,172,0.13)",
        padding: "44px 28px 36px 28px",
        color: accent,
        transition: "box-shadow 0.3s"
      }}>
        {renderServiceWarning()}
        <h2 style={{
          textAlign: "center",
          fontWeight: 900,
          letterSpacing: 2,
          color: "transparent",
          background: "linear-gradient(90deg, #7f53ac 0%, #43cea2 100%)",
          WebkitBackgroundClip: "text",
          fontSize: 28,
          marginBottom: 22
        }}>AI 智能体短视频生成</h2>
        <div style={{ marginBottom: 20, background: sectionBg, borderRadius: 12, padding: 16, boxShadow: "0 2px 8px #e0e7ff33", minWidth: 0 }}>
          <textarea
            value={script}
            onChange={e => setScript(e.target.value)}
            rows={6}
            placeholder="请输入要生成视频的脚本..."
            style={{
              width: "100%",
              minHeight: 120,
              maxHeight: 220,
              borderRadius: 12,
              border: `1.5px solid ${accent}`,
              padding: 14,
              fontSize: 16,
              boxShadow: "0 2px 8px rgba(127,83,172,0.06)",
              outline: "none",
              transition: "border 0.2s, box-shadow 0.2s",
              resize: "vertical",
              lineHeight: 1.7,
              background: "#f7f8fa",
              boxSizing: "border-box"
            }}
            onFocus={e => e.target.style.border = `1.5px solid ${accent2}`}
            onBlur={e => e.target.style.border = `1.5px solid ${accent}`}
          />
        </div>
        <div style={{
          display: 'flex',
          flexDirection: 'row',
          gap: 24,
          marginBottom: 24,
          background: THEME.bg,
          borderRadius: 14,
          boxShadow: '0 2px 12px #A389F422',
          padding: 18,
          minWidth: 0,
          alignItems: 'flex-start',
          flexWrap: 'wrap'
        }}>
          {/* 图片风格选择 */}
          <div style={{ flex: 1, minWidth: 180 }}>
            <label style={{ fontWeight: 700, color: THEME.primary, marginRight: 8, fontSize: 17 }}>图片风格：</label>
            <select value={style} onChange={e => setStyle(e.target.value)}
              style={{
                borderRadius: 10,
                padding: '10px 16px',
                fontSize: 16,
                border: `1.5px solid ${THEME.accent}`,
                background: THEME.selectBg,
                color: THEME.text,
                boxShadow: '0 1px 4px #A389F411',
                outline: 'none',
                marginRight: 12,
                transition: 'border 0.2s, box-shadow 0.2s',
                width: '100%'
              }}
              onFocus={e => e.target.style.border = `2px solid ${THEME.primary}`}
              onBlur={e => e.target.style.border = `1.5px solid ${THEME.accent}`}
            >
              <option value="" disabled>请选择风格</option>
              {SPACE_STYLES.map((styleObj, idx) => (
                <option key={styleObj.value} value={styleObj.value} style={{ background: THEME.selectBg, color: THEME.text }}>
                  {styleObj.label}
                </option>
              ))}
            </select>
            {style && (
              <div style={{ marginTop: 6, color: THEME.desc, fontSize: 14, fontStyle: 'italic', paddingLeft: 2 }}>
                {SPACE_STYLES.find(s => s.value === style)?.desc || '无描述'}
              </div>
            )}
          </div>
          {/* 图片尺寸选择 */}
          <div style={{ flex: 1, minWidth: 160 }}>
            <label style={{ fontWeight: 700, color: THEME.primary, marginRight: 8, fontSize: 16 }}>图片尺寸：</label>
            <select value={size} onChange={e => setSize(e.target.value)}
              style={{
                borderRadius: 8,
                padding: '8px 14px',
                fontSize: 15,
                border: `1.2px solid ${THEME.accent}`,
                background: THEME.selectBg,
                color: THEME.text,
                marginRight: 10,
                width: '100%'
              }}>
              {IMAGE_SIZES.map(opt => (
                <option key={opt.value} value={opt.value}>{opt.label}</option>
              ))}
            </select>
            {size === "自定义" && (
              <input
                type="text"
                placeholder="如 900x1600"
                value={customSize}
                onChange={e => setCustomSize(e.target.value)}
                style={{
                  borderRadius: 8,
                  padding: '8px 14px',
                  fontSize: 15,
                  border: `1.2px solid ${THEME.accent}`,
                  width: 120,
                  marginTop: 6
                }}
              />
            )}
          </div>
          {/* TTS声音选择 */}
          <div style={{ flex: 1, minWidth: 180 }}>
            <label style={{ fontWeight: 700, color: THEME.accent, marginRight: 8, fontSize: 17 }}>TTS声音：</label>
            <select value={voice} onChange={e => setVoice(e.target.value)}
              style={{
                borderRadius: 10,
                padding: '10px 16px',
                fontSize: 16,
                border: `1.5px solid ${THEME.primary}`,
                background: THEME.selectBg,
                color: THEME.text,
                boxShadow: '0 1px 4px #5B8CFF11',
                outline: 'none',
                marginRight: 12,
                transition: 'border 0.2s, box-shadow 0.2s',
                width: '100%'
              }}
              onFocus={e => e.target.style.border = `2px solid ${THEME.accent}`}
              onBlur={e => e.target.style.border = `1.5px solid ${THEME.primary}`}
            >
              <option value="" disabled>请选择声音</option>
              {voices.map((voiceObj, idx) => (
                <option key={voiceObj.value || voiceObj} value={voiceObj.value || voiceObj} style={{ background: THEME.selectBg, color: THEME.text }}>
                  {voiceObj.label || voiceObj}
                </option>
              ))}
            </select>
            {voice && (
              <div style={{ marginTop: 6, color: THEME.desc, fontSize: 14, fontStyle: 'italic', paddingLeft: 2 }}>
                {voices.find(v => (v.value || v) === voice)?.desc || '无描述'}
              </div>
            )}
          </div>
        </div>
        <div style={{ display: "flex", gap: 12, marginBottom: 18, minWidth: 0 }}>
          <label style={{ flex: 1 }}>
            <span style={{ fontSize: 14, color: accent }}>字幕文件</span>
            <input type="file" accept=".srt,.ass,.vtt" onChange={e => setSubtitleFile(e.target.files[0])} style={{ width: "100%", marginTop: 4 }} />
          </label>
          <label style={{ flex: 1 }}>
            <span style={{ fontSize: 14, color: accent }}>封面图片</span>
            <input type="file" accept="image/*" onChange={e => setCoverImage(e.target.files[0])} style={{ width: "100%", marginTop: 4 }} />
          </label>
        </div>
        <div style={{ background: sectionBg, borderRadius: 12, padding: 14, marginBottom: 20, boxShadow: "0 1px 4px #e0e7ff22", minWidth: 0 }}>
          <div style={{ fontWeight: 700, marginBottom: 8, color: accent2, fontSize: 16, letterSpacing: 1 }}>动态元素（Overlay，可添加多个）</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {overlays.map((o, idx) => (
              <div key={idx} style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 8, background: "#fff", borderRadius: 8, padding: 8, boxShadow: "0 1px 4px #e0e7ff22" }}>
                <input type="file" accept="image/*" onChange={e => handleOverlayChange(idx, "image", e.target.files[0])} style={{ flex: 2, minWidth: 120 }} />
                <input type="number" min="0" step="0.1" placeholder="起始时间(s)" value={o.start} onChange={e => handleOverlayChange(idx, "start", e.target.value)} style={{ flex: 1, minWidth: 90, borderRadius: 6, padding: 6, border: `1px solid ${accent}` }} />
                <input type="number" min="0" step="0.1" placeholder="结束时间(s)" value={o.end} onChange={e => handleOverlayChange(idx, "end", e.target.value)} style={{ flex: 1, minWidth: 90, borderRadius: 6, padding: 6, border: `1px solid ${accent}` }} />
                <input type="text" placeholder="位置(如10:10)" value={o.position} onChange={e => handleOverlayChange(idx, "position", e.target.value)} style={{ flex: 1, minWidth: 90, borderRadius: 6, padding: 6, border: `1px solid ${accent}` }} />
                <button onClick={() => removeOverlay(idx)} style={{ background: "#fff0f3", color: errorColor, border: `1px solid ${errorColor}33`, borderRadius: 6, padding: "4px 10px", fontWeight: 700, cursor: overlays.length > 1 ? "pointer" : "not-allowed", fontSize: 16, marginLeft: 2, transition: "background 0.2s" }} disabled={overlays.length === 1}>✕</button>
              </div>
            ))}
            <button onClick={addOverlay} style={{
              background: "linear-gradient(90deg, #a8edea 0%, #fed6e3 100%)",
              color: accent,
              border: `1.2px solid ${accent2}`,
              borderRadius: 8,
              padding: "6px 18px",
              fontWeight: 700,
              fontSize: 15,
              marginTop: 4,
              cursor: "pointer",
              boxShadow: "0 1px 4px #e0e7ff22",
              transition: "background 0.2s",
              alignSelf: "flex-start"
            }}>+ 添加动态元素</button>
          </div>
        </div>
        <div style={{ margin: "24px 0 12px 0" }}>
          <label style={{ fontWeight: 500, marginRight: 16 }}>分镜生成方式：</label>
          <label style={{ marginRight: 12 }}>
            <input
              type="radio"
              name="storyboard_mode"
              checked={useAiStoryboard}
              onChange={() => setUseAiStoryboard(true)}
            />
            <span style={{ color: '#5B8CFF', fontWeight: 600 }}>AI智能分镜（推荐）</span>
          </label>
          <label>
            <input
              type="radio"
              name="storyboard_mode"
              checked={!useAiStoryboard}
              onChange={() => setUseAiStoryboard(false)}
            />
            直接按脚本分段（更快）
          </label>
          <div style={{ margin: "8px 0 20px 0", color: "#888", fontSize: 14 }}>
            <div>
              <b>AI智能分镜：</b>自动理解脚本，生成更丰富的画面与台词。适合复杂/创意脚本。
            </div>
            <div>
              <b>直接分段：</b>按脚本分句生成，速度更快，适合结构清晰、无需AI润色的场景。
            </div>
          </div>
        </div>
        <div style={{ margin: '0 0 8px 0' }}>
          {!useAiStoryboard && (
            <div style={{ color: '#A389F4', fontSize: 13, marginTop: 4 }}>
              建议每句话独立成段，便于生成效果更佳。
            </div>
          )}
        </div>
        {formError && <div style={{ color: errorColor, fontWeight: 700, marginBottom: 10, textAlign: "center" }}>{formError}</div>}
        <div style={{ margin: '32px 0 18px 0', textAlign: 'center' }}>
          <button
            onClick={async () => {
              setIsGenerating(true);
              setProgress(5);
              setStage('分镜生成中...');
              await handleSubmit();
            }}
            disabled={isGenerating}
            style={{
              background: isGenerating ? PROGRESS_THEME.bar : THEME.primary,
              color: '#fff',
              fontWeight: 700,
              fontSize: 18,
              border: 'none',
              borderRadius: 10,
              padding: '12px 38px',
              boxShadow: '0 2px 12px #5B8CFF22',
              cursor: isGenerating ? 'not-allowed' : 'pointer',
              opacity: isGenerating ? 0.7 : 1,
              transition: 'all 0.2s',
            }}
          >
            {isGenerating ? '正在生成视频...' : '一键生成视频'}
          </button>
          {isGenerating && (
            <div style={{ marginTop: 18 }}>
              <div style={{ color: PROGRESS_THEME.stage, fontWeight: 600, fontSize: 16, marginBottom: 6 }}>{stage}</div>
              <div style={{ width: 320, height: 12, background: PROGRESS_THEME.bg, borderRadius: 8, overflow: 'hidden', margin: '0 auto' }}>
                <div style={{ width: `${progress}%`, height: '100%', background: PROGRESS_THEME.bar, borderRadius: 8, transition: 'width 0.5s' }} />
              </div>
              <div style={{ color: PROGRESS_THEME.text, fontSize: 13, marginTop: 4 }}>{progress}%</div>
            </div>
          )}
        </div>
        <div style={{ margin: "10px 0", minHeight: 24, textAlign: "center" }}>
          {status && <span>任务状态：{status}</span>}
          {error && <div style={{ color: errorColor, fontWeight: 600 }}>{error}</div>}
        </div>
        {videoUrl && (
          <div style={{ marginTop: 28, textAlign: "center" }}>
            <video src={videoUrl} controls width="100%" style={{
              borderRadius: 18,
              boxShadow: "0 2px 16px rgba(127,83,172,0.18)",
              marginBottom: 12,
              maxHeight: 340,
              background: "#000"
            }} />
            <br />
            <a href={videoUrl} download style={{
              display: "inline-block",
              marginTop: 12,
              background: "linear-gradient(90deg, #43cea2 0%, #7f53ac 100%)",
              color: "#fff",
              padding: "12px 36px",
              borderRadius: 12,
              fontWeight: 800,
              textDecoration: "none",
              fontSize: 18,
              boxShadow: "0 2px 8px #7f53ac22",
              transition: "background 0.3s"
            }}>下载视频</a>
          </div>
        )}
        {isGenerating && (
          <div style={{ textAlign: 'center', margin: '8px 0', color: '#888' }}>
            当前分镜模式：{useAiStoryboard ? 'AI智能分镜' : '直接分段'}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;