"""
Gradio web app for AI personality detection (AIVIEW).

Upload an interview video and get the Big Five (OCEAN) personality scores
predicted by the BiLSTM models in ../models (audio + visual, late-fused).

Run:
    python app.py
"""

import gradio as gr

from personality_detector import OCEAN, get_predictor, warmup

TRAIT_DESC = {
    "extraversion": "Outgoing, energetic, assertive",
    "agreeableness": "Cooperative, compassionate, trusting",
    "conscientiousness": "Organized, disciplined, dependable",
    "neuroticism": "Emotional sensitivity to stress",
    "openness": "Curious, imaginative, open-minded",
}


def _level(score):
    if score >= 0.6:
        return "High", "green"
    if score <= 0.4:
        return "Low", "red"
    return "Moderate", "orange"


def _bars_perc(scores):
    html = '<div style="display:flex;flex-direction:column;gap:6px;width:100%">'
    for t in OCEAN:
        v = scores[t] * 100
        level, color = _level(scores[t])
        html += (
            f'<div><div style="display:flex;justify-content:space-between;font-size:13px">'
            f'<span><b>{t}</b> &mdash; {TRAIT_DESC[t]}</span>'
            f'<span>{v:.1f}% &middot; <span style="color:{color}">{level}</span></span></div>'
            f'<div style="background:#e5e7eb;border-radius:4px;height:12px">'
            f'<div style="width:{v:.1f}%;background:{color};height:12px;border-radius:4px"></div>'
            f'</div></div>'
        )
    return html + "</div>"


def render_table(result):
    def row(mod, title):
        cells = "".join(f"<td>{result[mod][t] * 100:.1f}%</td>" for t in OCEAN)
        return f"<tr><td><b>{title}</b></td>{cells}</tr>"

    head = "".join(f"<th>{t}</th>" for t in OCEAN)
    return (
        '<div style="max-width:900px;margin:0 auto;font-family:sans-serif">'
        '<table style="border-collapse:collapse;width:100%;text-align:center">'
        f"<tr><th>Source</th>{head}</tr>"
        + row("audio", "Audio model")
        + row("visual", "Visual model")
        + row("fused", "Fused (avg)")
        + "</table>"
        '<div style="margin-top:16px">'
        + _bars_perc(result["fused"])
        + "</div></div>"
    )


def detect(video_path):
    if video_path is None:
        raise gr.Error("Please upload a video file first.")
    result = get_predictor().predict(video_path)
    return render_table(result), result["fused"]


with gr.Blocks(title="AIVIEW - Personality Detection") as demo:
    gr.Markdown(
        "# AIView - Personality Detection (Big Five / OCEAN)\n"
        "Upload a video of an interview and the BiLSTM models (audio + visual, "
        "late-fused) will estimate the five OCEAN personality traits."
    )
    with gr.Row():
        video = gr.Video(label="Interview video", sources=["upload"], format="mp4")
        with gr.Column():
            button = gr.Button("Detect Personality", variant="primary")
            gr.HTML(
                "<small><b>Note:</b> the first run downloads the VGGish and "
                "VGG-Face weights (~620 MB total). Visual analysis samples 30 "
                "frames, so detection may take a few seconds.</small>"
            )
    result_html = gr.HTML(label="Result")
    result_label = gr.Label(label="Fused OCEAN scores (0-1)")
    button.click(detect, inputs=video, outputs=[result_html, result_label])

if __name__ == "__main__":
    warmup()
    demo.launch(server_name="0.0.0.0", server_port=7860)