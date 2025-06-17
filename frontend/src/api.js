import axios from "axios";

export async function getImageStyles() {
  const res = await axios.get("/available_image_styles");
  return res.data.available_styles || [];
}

export async function getTTSVoices() {
  const res = await axios.get("/available_tts_voices");
  return res.data.voices || [];
}

export async function submitScript({ script, style, voice, subtitleFile, coverImage, overlays, size, use_ai_storyboard }) {
  const formData = new FormData();
  const blob = new Blob([script], { type: "text/plain" });
  formData.append("file", blob, "script.txt");
  if (style) formData.append("style", style);
  if (voice) formData.append("voice", voice);
  if (subtitleFile) formData.append("subtitle_file", subtitleFile);
  if (coverImage) formData.append("cover_image", coverImage);
  if (overlays && overlays.length > 0) {
    overlays.forEach((o, i) => {
      if (o.image) {
        formData.append(`overlay_${i}_image`, o.image);
        formData.append(`overlay_${i}_meta`, JSON.stringify({
          start: o.start,
          end: o.end,
          position: o.position
        }));
      }
    });
    formData.append("overlay_count", overlays.length);
  }
  if (size) formData.append("size", size);
  if (typeof use_ai_storyboard !== 'undefined') formData.append("use_ai_storyboard", use_ai_storyboard);
  const res = await axios.post("/generate_video_from_script_async", formData);
  return res.data.job_id;
}

export async function getJobStatus(jobId) {
  const res = await axios.get(`/result/${jobId}`);
  return res.data;
} 