import streamlit as st
import time
import random
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt


st.set_page_config(page_title="Ai tools", layout="wide")

selected_farm = st.sidebar.selectbox("Select the farm:", ["Image", "Machine Learning"])

if selected_farm == 'Image':
    import cv2
    import numpy as np
    from PIL import Image
    import streamlit as st

    import cv2
    import numpy as np


    def detect_crops_and_health_week2(image, min_pixels=900, min_health_percentage=40, min_component_area=500):
        """
        Detect crops and analyze their health based on green pixels in the image.

        Parameters:
            image (numpy.ndarray): The input image (RGB or BGR format).
            min_pixels (int): Minimum green pixels to classify a crop as healthy.
            min_health_percentage (float): Minimum percentage of green pixels for a crop to be considered healthy.
            min_component_area (int): Minimum area (in pixels) for a detected crop.

        Returns:
            processed_image (numpy.ndarray): The input image with crops annotated.
            crop_health (list): A list of dictionaries containing health data for each crop.
            healthy_crops (int): The number of healthy crops detected.
            total_crops (int): The total number of crops detected.
            improved_mask (numpy.ndarray): Binary mask highlighting green areas in the image.
        """

        # Ensure the input image is a numpy array
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a numpy array.")

        # Convert to BGR format if the image is in RGB
        if len(image.shape) == 3 and image.shape[2] == 3:  # Check for color image
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Convert to HSV for color-based segmentation
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define green color range for segmentation
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([90, 255, 255])

        # Create a mask for green areas
        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

        # Morphological operations to remove noise and close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        cleaned_green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

        # Detect connected components in the cleaned mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_green_mask, connectivity=8)

        # Initialize variables for crop health analysis
        processed_image = image.copy()
        crop_health = []
        total_crops = 0
        healthy_crops = 0

        # Analyze each detected component
        for i in range(1, num_labels):  # Skip the background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_component_area:  # Filter components by size
                total_crops += 1

                # Get bounding box coordinates
                x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[
                    i, cv2.CC_STAT_HEIGHT]

                # Extract the crop mask and calculate green pixel percentage
                crop_mask = cleaned_green_mask[y:y + h, x:x + w]
                green_pixels = cv2.countNonZero(crop_mask)
                total_pixels = w * h
                green_percentage = (green_pixels / total_pixels) * 100

                # Determine health status
                if green_percentage >= min_health_percentage and green_pixels >= min_pixels:
                    health_status = "Healthy"
                    healthy_crops += 1
                    border_color = (0, 255, 0)  # Green for healthy crops
                else:
                    health_status = "Unhealthy"
                    border_color = (255, 0, 0)  # Red for unhealthy crops

                # Annotate the processed image
                cv2.rectangle(processed_image, (x, y), (x + w, y + h), border_color, 3)
                cv2.putText(processed_image, f"Crop {total_crops}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            border_color, 2)

                # Append crop health data
                crop_health.append({
                    "Crop": total_crops,
                    "Health Status": health_status,
                    "Green Pixels": green_pixels,
                    "Total Pixels": total_pixels,
                    "Green Percentage": green_percentage
                })

        # Create a mask highlighting green areas
        improved_mask = np.zeros_like(image)
        improved_mask[cleaned_green_mask > 0] = [0, 255, 0]

        return processed_image, crop_health, healthy_crops, total_crops, improved_mask


    def generate_html_report(crop_health, healthy_crops, total_crops, desired_yield=250):
        """
        Generate an HTML report for crop health analysis.

        Parameters:
            crop_health (list): List of dictionaries containing crop health data.
            healthy_crops (int): Number of healthy crops detected.
            total_crops (int): Total number of crops detected.
            desired_yield (int): Desired yield for the area (in grams).

        Returns:
            str: HTML content for the report.
        """
        healthy_percentage = (healthy_crops / total_crops) * 100 if total_crops > 0 else 0

        # Create a header for the overall summary
        html_content = f"""<div style="border: 2px solid #4CAF50; border-radius: 10px; padding: 15px; background-color: #f9f9f9;">
            <h4 style="color: #4CAF50; text-align: center;">Health Status of Area</h4>
            <p><b>Total Crops Detected:</b> {total_crops}</p>
            <p><b>Healthy Crops:</b> {healthy_crops}/{total_crops} ({healthy_percentage:.2f}%)</p>
            <p><b>Desired Yield for the Area:</b> {desired_yield} grams</p>
        </div>"""

        # Add a table for detailed crop data
        html_content += """<table style="width: 100%; border-collapse: collapse; margin-top: 15px;">
            <tr style="background-color: #4CAF50; color: white;">
                <th style="border: 1px solid #ddd; padding: 8px;">Crop</th>
                <th style="border: 1px solid #ddd; padding: 8px;">Health Status</th>
                <th style="border: 1px solid #ddd; padding: 8px;">Green Pixels</th>
                <th style="border: 1px solid #ddd; padding: 8px;">Total Pixels</th>
                <th style="border: 1px solid #ddd; padding: 8px;">Green Percentage (%)</th>
                <th style="border: 1px solid #ddd; padding: 8px;">Canopy Size (mm²)</th>
            </tr>"""

        # Add rows for each crop
        for crop in crop_health:
            canopy_size = crop["Total Pixels"] * 0.1  # Assuming each pixel represents 0.1 mm²
            html_content += f"""<tr>
                <td style="border: 1px solid #ddd; padding: 8px;">Crop {crop['Crop']}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{crop['Health Status']}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{crop['Green Pixels']}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{crop['Total Pixels']}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{crop['Green Percentage']:.2f}</td>
                <td style="border: 1px solid #ddd; padding: 8px;">{canopy_size:.2f} mm²</td>
            </tr>"""

        # Close the table
        html_content += "</table>"

        return html_content


    import streamlit as st
    from PIL import Image
    import numpy as np

    # Streamlit app title
    st.title("Crop Health Detection and Reporting")

    # File uploader to upload an image
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Open the uploaded image using PIL
        image = Image.open(uploaded_image)

        # Convert the PIL image to a numpy array
        image_array = np.array(image)

        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Call your crop health detection function here
        processed_image, crop_health, healthy_crops, total_crops, improved_mask = detect_crops_and_health_week2(
            image_array)

        # Generate the HTML report
        html_report = generate_html_report(crop_health, healthy_crops, total_crops)

        # Display the report
        st.markdown(html_report, unsafe_allow_html=True)

        # Optionally display the processed image and mask
        st.image(processed_image, caption="Processed Image", use_column_width=True)
        st.image(improved_mask, caption="Improved Mask", use_column_width=True)
    else:
        st.write("Please upload an image to analyze.")

if selected_farm == 'Machine Learning':
    import streamlit as st
    import pandas as pd
    import time
    import random

    # Mock model performance and parameter results
    mock_tuning_results = {
        'RandomForest': {'best_params': {'n_estimators': 100, 'max_depth': 10}, 'accuracy': 0.91},
        'AdaBoost': {'best_params': {'n_estimators': 200, 'learning_rate': 0.1}, 'accuracy': 0.89},
        'KNN': {'best_params': {'n_neighbors': 5}, 'accuracy': 0.85},
        'MLP': {'best_params': {'hidden_layer_sizes': (100,), 'max_iter': 500}, 'accuracy': 0.88},
        'LogisticRegression': {'best_params': {'C': 1.0, 'penalty': 'l2'}, 'accuracy': 0.86},
        'DecisionTree': {'best_params': {'max_depth': 10, 'min_samples_split': 2}, 'accuracy': 0.84},
        'SVM': {'best_params': {'C': 1.0, 'kernel': 'rbf'}, 'accuracy': 0.87},
        'AutoML_H2O': {'best_params': 'Auto-tuned by H2O', 'accuracy': 0.92},
        'AutoML_TPOT': {'best_params': 'Auto-tuned by TPOT', 'accuracy': 0.90},
        'AutoML_AutoSklearn': {'best_params': 'Auto-tuned by Auto-Sklearn', 'accuracy': 0.91}
    }

    # Streamlit app
    st.title("Simulated ML Dashboard with Hyperparameter Tuning")

    # Styled Introduction
    st.markdown(
        """
        <div style="background-color:#f0f8ff; padding:15px; border-radius:10px; border:1px solid #d3d3d3;">
            <h2 style="color:#007acc;">Welcome to the Simulated ML Dashboard</h2>
            <p>This tool allows you to simulate the process of training and tuning Machine Learning models. Here's how it works:</p>
            <ol>
                <li><strong>Step 1:</strong> Upload your dataset in CSV format.</li>
                <li><strong>Step 2:</strong> Select the input features and target column.</li>
                <li><strong>Step 3:</strong> Choose models and a tuning method (Grid Search or Random Search).</li>
                <li><strong>Step 4:</strong> Run the simulation to view the best parameters and performance metrics.</li>
            </ol>
            <p style="color: #333;">This dashboard is for demonstration purposes and does not perform actual training.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info("Please follow the steps below to proceed.")

    # Step 1: Dataset Upload
    st.write("### Step 1: Upload Dataset")
    uploaded_file = st.file_uploader("Upload your CSV file to get started:", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        # Display Dataset Preview
        st.write("### Dataset Preview")
        st.dataframe(df.head(10))  # Display the first 10 rows of the dataset

        # Step 2: Feature and Target Selection
        st.write("### Step 2: Select Features and Target")
        columns = df.columns.tolist()

        # Select Features
        features = st.multiselect("Select Feature Columns", options=columns,
                                  help="These columns will be used as input features.")
        # Update Target Options
        target_options = [col for col in columns if col not in features]
        target = st.selectbox("Select Target Column", options=target_options,
                              help="This column will be the output/target variable.")

        if features and target:
            X = df[features]
            y = df[target]

            # Step 3: Model Selection and Tuning Method
            st.write("### Step 3: Select ML Models and Tuning Method")
            model_options = [
                'RandomForest', 'AdaBoost', 'KNN', 'MLP',
                'LogisticRegression', 'DecisionTree', 'SVM',
                'AutoML_H2O', 'AutoML_TPOT', 'AutoML_AutoSklearn'
            ]
            selected_models = st.multiselect("Select Models to Simulate", options=model_options,
                                             help="Select one or more models for simulation.")
            tuning_method = st.radio("Select Tuning Method", options=['Grid Search', 'Random Search'],
                                     help="Choose the hyperparameter tuning method.")

            if selected_models:
                # Step 4: Simulate Model Training and Tuning
                st.write("### Step 4: Simulate Model Training and Tuning")
                simulate_button = st.button("Simulate Training and Tuning")

                if simulate_button:
                    st.write("Preparing Data...")
                    progress = st.progress(0)
                    time.sleep(1)  # Simulate data preparation
                    progress.progress(20)

                    st.write(f"Simulating Model Training and Tuning using {tuning_method}...")
                    tuned_results = {}
                    for i, model_name in enumerate(selected_models):
                        time.sleep(2)  # Simulate tuning time for each model
                        progress.progress(20 + (i + 1) * int(80 / len(selected_models)))

                        # Simulate tuning results
                        tuned_results[model_name] = mock_tuning_results[model_name]

                    progress.progress(100)
                    st.success("Simulation Complete!")

                    # Display Tuned Results
                    st.write("### Tuned Model Results")
                    for model_name, result in tuned_results.items():
                        st.write(f"**{model_name}**")
                        st.write(f"- Best Parameters: {result['best_params']}")
                        st.write(f"- Accuracy: {result['accuracy']:.2f}")

    else:
        st.warning("Please upload a dataset to proceed.")
