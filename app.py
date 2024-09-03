#importing necessary variables
import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import io
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Load the CLIP model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

# Refined feature descriptions
feature_descriptions = [
    "Screen for selecting the starting point, destination, and travel date",
    "Screen to select the bus for travel",
    "Screen for choosing your seat on the bus",
    "Screen to select the boarding point",
    "Screen for applying any available discounts",
    "Screen to sort buses by time, price, or other criteria",
    "Screen showing details about the bus such as amenities, photos, and user reviews"
]

# Initializing the text generator with API key and model settings
generator = OpenAIGenerator(
    api_key=Secret.from_env_var("GROQ_API_KEY"),
    api_base_url="https://api.groq.com/openai/v1",
    model="llama-3.1-70b-versatile",
    generation_kwargs={"max_tokens": 2048}
)

# Multi-shot Prompts
multi_shot_examples = [
    {
        "feature": "Source, Destination, and Date Selection",
        "example": """
        Feature: Source, Destination, and Date Selection
        Description: This test case is about verifying the user’s ability to select a source, destination, and travel date.
        Pre-conditions: Ensure that the app is installed and launched.
        Testing Steps:
        1. Open the app and navigate to the home screen.
        2. Tap on the "Source" field and select the source city from the list.
        3. Tap on the "Destination" field and select the destination city from the list.
        4. Tap on the "Date" field and choose the travel date from the calendar.
        5. Verify that the selected source, destination, and date are displayed correctly.
        Expected Result: The selected source, destination, and date should be correctly displayed on the home screen.
        """
    },
    {
        "feature": "Bus Selection",
        "example": """
        Feature: Bus Selection
        Description: This test case is about verifying the user’s ability to select a bus from the available options.
        Pre-conditions: Ensure that the source, destination, and date are already selected.
        Testing Steps:
        1. After selecting the source, destination, and date, tap on the "Search Buses" button.
        2. Review the list of available buses displayed.
        3. Tap on any bus to view more details.
        4. Verify that the details of the selected bus are displayed correctly.
        Expected Result: The app should display the details of the selected bus accurately.
        """
    },
     {
        "feature": "Seat Selection",
        "example": """
        Feature: Seat Selection
        Description: This test case is about verifying the user’s ability to select a seat on the bus.
        Pre-conditions: Ensure that a bus has been selected.
        Testing Steps:
        1. After selecting a bus, navigate to the seat selection screen.
        2. Tap on an available seat to select it.
        3. Verify that the selected seat is highlighted and no other seat is selected.
        4. Tap on the "Continue" button to proceed.
        Expected Result: The selected seat should be highlighted, and the user should be able to proceed to the next step.
        """
    },
    {
        "feature": "Pick-up and Drop-off Point Selection",
        "example": """
        Feature: Pick-up and Drop-off Point Selection
        Description: This test case is about verifying the user’s ability to select a pick-up and drop-off point.
        Pre-conditions: Ensure that a bus has been selected.
        Testing Steps:
        1. After selecting a bus, navigate to the pick-up and drop-off point selection screen.
        2. Tap on the "Pick-up Point" field and select the desired pick-up point from the list.
        3. Tap on the "Drop-off Point" field and select the desired drop-off point from the list.
        4. Verify that the selected points are displayed correctly.
        Expected Result: The selected pick-up and drop-off points should be correctly displayed on the screen.
        """
    },
    {
        "feature": "Offers",
        "example": """
        Feature: Offers
        Description: This test case is about verifying the user’s ability to apply available discounts or promotions.
        Pre-conditions: Ensure that a bus and seat have been selected.
        Testing Steps:
        1. After selecting a seat, navigate to the offers screen.
        2. Review the list of available offers.
        3. Tap on an offer to apply it to the booking.
        4. Verify that the discount or promotion is correctly applied to the total amount.
        Expected Result: The selected offer should be correctly applied, and the total amount should reflect the discount.
        """
    },
    {
        "feature": "Filters",
        "example": """
        Feature: Filters
        Description: This test case is about verifying the user’s ability to filter the list of available buses.
        Pre-conditions: Ensure that the source, destination, and date are already selected.
        Testing Steps:
        1. After selecting the source, destination, and date, tap on the "Search Buses" button.
        2. Tap on the "Filters" option to open the filter settings.
        3. Apply filters such as departure time, bus type, and price range.
        4. Verify that the list of available buses updates according to the applied filters.
        Expected Result: The list of available buses should be filtered correctly based on the selected criteria.
        """
    },
    {
        "feature": "Bus Information",
        "example": """
        Feature: Bus Information
        Description: This test case is about verifying the user’s ability to view details about the selected bus.
        Pre-conditions: Ensure that a bus has been selected.
        Testing Steps:
        1. After selecting a bus, navigate to the bus information screen.
        2. Review the details such as amenities, photos, and user reviews.
        3. Verify that all information is displayed correctly and matches the selected bus.
        Expected Result: The bus information should be displayed accurately, including all details like amenities, photos, and user reviews.
        """
    }
]

# Function to get image embedding
def get_image_embedding(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    return image_features

# Function to generate testing instructions using LLM
def generate_testing_instructions(images):
    results = []
    for image in images:
        image = Image.open(io.BytesIO(image.read())).convert('RGB')
        img_features = get_image_embedding(image)
        similarities = []

         # Calculating similarity between image features and text descriptions
        for desc in feature_descriptions:
            text_inputs = processor(text=desc, return_tensors="pt").to(device)
            with torch.no_grad():
                text_features = clip_model.get_text_features(**text_inputs)
            similarity = torch.nn.functional.cosine_similarity(img_features, text_features)
            similarities.append((desc, similarity.item()))
        
        # Finding the feature with highest similarity
        highest_similarity_desc = max(similarities, key=lambda x: x[1])[0]
        
        # Finding the corresponding example from Multi-shot Prompts
        example_prompt = ""
        for example in multi_shot_examples:
            if example["feature"] in highest_similarity_desc:
                example_prompt = example["example"]
                break

        # Including the optional context in the prompt
        prompt = (
            f"Based on the provided screenshot and the optional context: '{context}', generate detailed testing instructions for the following feature: {highest_similarity_desc}. "
            f"Include tests for visible functionalities and exclude unrelated features. "
            f"Here is an example of how to structure the instructions: \n{example_prompt}"
        )

        # Generating the testing instructions
        try:
            result = generator.run(prompt=prompt)
            if "replies" in result:
                instruction = result["replies"][0].strip()
            else:
                instruction = "No result found"
        except Exception as e:
            instruction = f"Error generating instructions: {e}"

        # Formatting the result for display in Streamlit
        formatted_result = (
            f"<div style='border: 1px solid #D5D8DC; padding: 10px; margin: 10px 0; background-color: #F7F9F9;'>"
            f"<h2>Instructions for Image</h2>"
            f"<h3 style='color: #1A5276;'>Feature: {highest_similarity_desc}</h3>"
            f"{instruction}"
            f"</div>"
        )
        results.append(formatted_result)

    return "<br>".join(results)


# Streamlit UI
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f7fa;
        padding: 20px;
        border-radius: 10px;
        font-family: 'Arial', sans-serif;
    }
    h1 {
        color: #2C3E50;
        text-align: center;
        font-weight: bold;
        margin-bottom: 30px;
    }
    .stTextInput > label {
        font-size: 18px;
        font-weight: 600;
        color: #34495E;
    }
    .stTextArea > label {
        font-size: 18px;
        font-weight: 600;
        color: #34495E;
    }
    .stFileUploader > label {
        font-size: 18px;
        font-weight: 600;
        color: #34495E;
    }
    .stButton button {
        background-color: #1A5276;
        color: white;
        border-radius: 5px;
        font-size: 18px;
        font-weight: 600;
        padding: 10px 20px;
        margin-top: 20px;
    }
    .stButton button:hover {
        background-color: #154360;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='main'><h1>Testing Instructions Generator for Red Bus</h1>", unsafe_allow_html=True)

# Explicit warning about screenshot relevance
st.warning(
    "This tool is designed specifically for generating testing instructions for screenshots related to the Red Bus mobile app. "
    "Uploading screenshots unrelated to Red Bus app features may lead to inaccurate or irrelevant outputs."
)

st.markdown(
    "Please upload screenshots of the Red Bus app's features, and click 'Describe Testing Instructions' to generate detailed instructions."
)

context = st.text_area("Optional Context", help="Provide any additional context that might help in generating accurate testing instructions.")

uploaded_files = st.file_uploader(
    "Upload Screenshots",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
    help="Upload one or more screenshots of the Red Bus app."
)

if st.button("Describe Testing Instructions", key="submit-button"):
    if uploaded_files:
        with st.spinner("Generating testing instructions..."):
            result_html = generate_testing_instructions(uploaded_files)
            st.markdown(result_html, unsafe_allow_html=True)
    else:
        st.error("Please upload at least one screenshot.")