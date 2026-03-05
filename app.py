import gradio as gr
from inference import predict

THRESHOLD = 0.5

# Disease explanations
disease_info = {
    "Atelectasis": "Partial collapse of lung tissue.",
    "Cardiomegaly": "Enlargement of the heart.",
    "Consolidation": "Lung tissue filled with fluid instead of air.",
    "Edema": "Fluid accumulation in the lungs.",
    "Pleural Effusion": "Fluid buildup around the lungs.",
    "Pneumonia": "Infection causing inflammation in lung air sacs.",
    "Pneumothorax": "Collapsed lung caused by air in chest cavity.",
    "No Finding": "No significant abnormality detected."
}


def risk_level(p):
    if p > 0.8:
        return "🔴 High"
    elif p > 0.5:
        return "🟡 Medium"
    else:
        return "🟢 Low"


def predict_gradio(image):

    preds = predict(image)

    # sort predictions
    sorted_preds = dict(sorted(preds.items(), key=lambda x: x[1], reverse=True))

    # keep confident predictions
    filtered = {k: v for k, v in sorted_preds.items() if v > THRESHOLD}

    # if nothing above threshold → show top prediction
    if len(filtered) == 0:
        top = list(sorted_preds.items())[0]
        filtered = {top[0]: top[1]}

    # format output text
    text_output = ""

    for disease, prob in filtered.items():

        risk = risk_level(prob)
        info = disease_info[disease]

        text_output += (
            f"### {disease}\n"
            f"Probability: **{prob:.2%}**\n"
            f"Risk Level: **{risk}**\n"
            f"{info}\n\n"
        )

    return filtered, text_output


with gr.Blocks() as demo:

    gr.Markdown("# 🩺 PleuraMind — Chest X-ray Disease Detector")

    gr.Markdown(
        "Upload a **chest X-ray image** and the AI model will predict possible thoracic diseases."
    )

    with gr.Row():

        image_input = gr.Image(
            type="pil",
            label="Upload Chest X-ray"
        )

        prediction_output = gr.Label(
            num_top_classes=3,
            label="Model Predictions"
        )

    analyze_btn = gr.Button("Analyze X-ray")

    explanation_output = gr.Markdown()

    analyze_btn.click(
        fn=predict_gradio,
        inputs=image_input,
        outputs=[prediction_output, explanation_output]
    )

    gr.Markdown(
        """
        ⚠️ **Disclaimer**

        This AI system is intended for **research and educational purposes only**.  
        It **must not be used for medical diagnosis**.  
        Always consult a qualified medical professional.
        """
    )


demo.launch()