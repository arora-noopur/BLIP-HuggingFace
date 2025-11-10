import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# --- Model and Processor Loading ---
# Load the model and processor only ONCE when the app starts.
# This is crucial for performance.
print("Loading BLIP model and processor...")
try:
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    # Check if a GPU is available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    print(f"Model loaded successfully on {device}.")
except Exception as e:
    print(f"Error loading model: {e}")
    # We can't continue if the model fails to load
    processor = None
    model = None


def generate_caption(pil_image: Image.Image) -> str:
    """
    Generates a caption for the given PIL image using the pre-loaded BLIP model.
    """
    if not model or not processor:
        return "Error: Model is not loaded. Please check server logs."

    try:
        # Ensure image is in RGB format
        image = pil_image.convert("RGB")

        print("Processing image and generating caption...")

        # Prepare the image
        inputs = processor(image, return_tensors="pt").to(device)

        # Generate captions
        outputs = model.generate(**inputs)

        # Decode the generated tokens to a string
        caption = processor.decode(outputs[0], skip_special_tokens=True)

        print(f"Generated caption: {caption}")
        return caption

    except Exception as e:
        print(f"Error during caption generation: {e}")
        return f"Error: Could not generate caption. {e}"


# --- Gradio Interface Definition ---

# Define the Gradio interface
# 'type="pil"' ensures the input function receives a PIL Image object
iface = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs="text",
    title="Image Captioning with BLIP",
    description="Upload any image and the Salesforce BLIP model will generate a caption for it.",
    examples=[
        ["https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ic.jpg"],
        ["https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beavers.png"]
    ]
)

# --- Launch the App ---
if __name__ == "__main__":
    if model and processor:
        print("Launching Gradio interface...")
        # Set share=True to get a public link (as discussed)
        iface.launch(share=True)
    else:
        print("Cannot launch Gradio app because the model failed to load.")