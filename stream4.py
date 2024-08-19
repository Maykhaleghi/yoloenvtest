import streamlit as st
from PIL import Image
import torch
import numpy as np
import cv2
import pandas as pd

# Custom CSS for a modern, clean look
st.markdown(
    """
    <style>
    body {
        background-color: #ADD8E6;
        font-family: 'Roboto', sans-serif;
    }
    .title {
        text-align: center;
        color: #4169E1;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .upload-box {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    .image-preview {
        text-align: center;
        margin-top: 2rem;
    }
    .caption {
        font-style: italic;
        color: #555555;
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    .result-box {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.1);
        margin-top: 2rem;
        text-align: center;
    }
    .number-display {
        font-size: 2rem;
        font-weight: 600;
        color: #4169E1;
        margin-top: 1rem;
    }
    .precision-display {
        font-size: 1.2rem;
        font-weight: 500;
        color: #333333;
        margin-top: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title of the app
st.markdown('<div class="title">YOLOv5 Object Detection Demo</div>', unsafe_allow_html=True)

# Upload box
#st.markdown('<div class="upload-box">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.markdown('<div class="image-preview">', unsafe_allow_html=True)
    st.image(image, caption='Uploaded Image', use_column_width=True, output_format="JPEG")
    st.markdown('<div class="caption">Running YOLOv5 model...</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Perform inference using the model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
    
    # Perform inference on the image
    results = model(image, size=352)
    
    # Convert the image to numpy array
    img_np = np.array(image)

    # Get labels and bounding boxes
    labels, cords = results.xyxyn[0][:, -1].cpu().numpy(), results.xyxyn[0][:, :-1].cpu().numpy()
    
    # List to store detected digits, their x-coordinates, and precisions
    detected_digits = []

    # Function to generate darker, more saturated colors
    def get_class_color(idx):
        colors = [
            (0, 0, 139),   # Dark Red
            (0, 100, 0),   # Dark Green
            (139, 0, 0),   # Dark Blue
            (255, 255, 0), # Dark Yellow
            (0, 139, 139), # Dark Cyan
            (139, 0, 139), # Dark Magenta
            (85, 0, 0),    # Darker Red
            (0, 85, 0),    # Darker Green
            (0, 0, 85),    # Darker Blue
            (85, 85, 0),   # Dark Olive
            (0, 85, 85),   # Dark Teal
            (85, 0, 85)    # Dark Purple
        ]
        return colors[idx % len(colors)]  # Cycle through colors

    # Loop over all detections
    for i in range(len(labels)):
        x1, y1, x2, y2, conf = cords[i]
        if conf > 0.4:  # confidence threshold
            # Convert coordinates to absolute values
            x1_abs, y1_abs, x2_abs, y2_abs = int(x1 * img_np.shape[1]), int(y1 * img_np.shape[0]), int(x2 * img_np.shape[1]), int(y2 * img_np.shape[0])
            
            # Get the label name
            label_name = str(model.names[int(labels[i])])  # Convert to string
            
            # Store the digit, its x-coordinate, and precision (confidence)
            detected_digits.append((x1_abs, label_name, conf))
            
            # Draw the bounding box with darker color
            color = get_class_color(int(labels[i]))
            img_np = cv2.rectangle(img_np, (x1_abs, y1_abs), (x2_abs, y2_abs), color=color, thickness=1)
            
            # Put label text above the bounding box with a smaller font size
            font_scale = 0.6  # Smaller font size
            font_thickness = 1  # Keep thickness small to avoid clutter
            label_size = cv2.getTextSize(label_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            label_y_position = max(y1_abs, label_size[1] + 2)
            img_np = cv2.putText(img_np, label_name, (x1_abs, label_y_position), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness, cv2.LINE_AA)

    # Sort the detected digits by their x-coordinates
    detected_digits.sort(key=lambda x: x[0])

    # Skip the first (leftmost) digit and prepare for display
    detected_number = ''.join([digit[1] for digit in detected_digits[1:]])
    detected_precisions = [f"{digit[2]:.2f}" for digit in detected_digits[1:]]

    # Convert back to PIL image
    detected_image = Image.fromarray(img_np)

    # Display the image with detections
    st.markdown('<div class="result-box">', unsafe_allow_html=True)
    st.image(detected_image, caption='Detected Image', use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Display the detected number
    if detected_number:
        st.markdown('<div class="number-display">Detected Number:</div>', unsafe_allow_html=True)
        st.write(f"**{detected_number}**")

        # Create a dataframe to display digits and their precision
        data = {
            "Digit": list(detected_number),
            "Precision": detected_precisions
        }
        df = pd.DataFrame(data)
        
        # Display the table in Streamlit
        st.markdown('<div class="number-display">Detected Digits and Their Precision:</div>', unsafe_allow_html=True)
        st.write("\n")
        st.table(df)
