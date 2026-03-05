import gradio as gr
from inference import predict  # the function above that accepts image path or PIL

def predict_gradio(img):
    # gradio passes a PIL Image when type="pil"
    img.save("tmp_input.jpg")
    preds = predict("tmp_input.jpg")
    # format as label dict sorted by prob
    return {k: float(v) for k,v in sorted(preds.items(), key=lambda x:-x[1])}

iface = gr.Interface(
    fn=predict_gradio,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=8),
    title="Chest X-ray Disease Detector",
    description="Upload a frontal chest X-ray image."
)

if __name__ == "__main__":
    iface.launch()