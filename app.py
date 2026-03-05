import gradio as gr
from inference import predict

THRESHOLD = 0.5

def predict_gradio(image):

    preds = predict(image)

    sorted_preds = dict(sorted(preds.items(), key=lambda x: x[1], reverse=True))

    filtered = {k: v for k, v in sorted_preds.items() if v > THRESHOLD}

    if len(filtered) == 0:
        top = list(sorted_preds.items())[0]
        filtered = {top[0]: top[1]}

    return filtered


with gr.Blocks() as demo:

    gr.Markdown("# 🩺 PleuraMind — Chest X-ray Disease Detector")

    gr.Markdown(
        "Upload a chest X-ray image and the AI model will predict possible thoracic diseases."
    )

    with gr.Row():

        image_input = gr.Image(type="pil", label="Upload Chest X-ray")

        prediction_output = gr.Label(
            num_top_classes=3,
            label="Model Predictions"
        )

    submit = gr.Button("🔍 Analyze X-ray")
    clear = gr.Button("Clear")

    submit.click(
        fn=predict_gradio,
        inputs=image_input,
        outputs=prediction_output
    )

    clear.click(lambda: (None, None), outputs=[image_input, prediction_output])

    gr.Markdown(
    "⚠️ This tool is for research and educational purposes only and not for clinical diagnosis."
    )

demo.launch()