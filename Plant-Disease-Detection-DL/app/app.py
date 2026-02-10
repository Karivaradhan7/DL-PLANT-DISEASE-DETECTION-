"""
Streamlit Web Application for Plant Disease Detection
Interactive interface for model inference and disease prediction
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from pathlib import Path
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import PlantDiseaseDataLoader


def load_model(model_path):
    """Load trained model"""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def preprocess_image(image, img_size=224):
    """Preprocess uploaded image"""
    image = image.resize((img_size, img_size))
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def predict_disease(model, image_array, class_names):
    """Make prediction on image"""
    predictions = model.predict(image_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    predicted_class = class_names[predicted_class_idx]
    
    return predicted_class, confidence, predictions[0]


def main():
    st.set_page_config(
        page_title="Plant Disease Detection",
        page_icon="🌿",
        layout="wide"
    )
    
    st.title("🌿 Plant Disease Detection System")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Settings")
    selected_page = st.sidebar.radio(
        "Choose Mode:",
        ["Home", "Single Image Prediction", "Batch Processing", "Model Information"]
    )
    
    #  Model path for demo (can be customized)
    saved_models_dir = Path(__file__).parent.parent / "saved_models"
    
    # ============== HOME ==============
    if selected_page == "Home":
        st.markdown("""
        ## Welcome to Plant Disease Detection System
        
        This application uses deep learning models to detect and classify plant diseases.
        
        ### Features:
        - **Single Image Prediction**: Upload a plant image to get disease classification
        - **Batch Processing**: Process multiple images at once
        - **Model Information**: View details about available models
        
        ### How to Use:
        1. Select a mode from the sidebar
        2. Upload or select an image
        3. Click predict to see the results
        
        ### Available Models:
        - MLP (Multi-Layer Perceptron)
        - CNN (Convolutional Neural Network)
        - Transfer Learning (MobileNetV2)
        - Autoencoder
        - DCGAN
        """)
        
        st.info("ℹ️ Tip: For best results, use clear images of plant leaves with good lighting.")
    
    # ============== SINGLE IMAGE PREDICTION ==============
    elif selected_page == "Single Image Prediction":
        st.header("Single Image Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload Image")
            uploaded_file = st.file_uploader(
                "Choose an image of a plant leaf",
                type=["jpg", "jpeg", "png", "bmp"]
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.subheader("Model Settings")
            
            # Get available models
            available_models = []
            if saved_models_dir.exists():
                available_models = [f.name for f in saved_models_dir.glob("*.h5")]
            
            if not available_models:
                st.warning("No trained models found in saved_models directory.")
                st.stop()
            
            selected_model = st.selectbox(
                "Select Model:",
                available_models
            )
            
            # Class names (should be loaded from config in production)
            class_names = st.text_input(
                "Class Names (comma-separated):",
                value="Healthy,Disease1,Disease2,Disease3"
            ).split(",")
            class_names = [name.strip() for name in class_names]
            
            if st.button("🔍 Predict Disease"):
                if uploaded_file is not None:
                    with st.spinner("Loading model and making prediction..."):
                        model_path = saved_models_dir / selected_model
                        model = load_model(str(model_path))
                        
                        if model is not None:
                            image_array = preprocess_image(image, img_size=224)
                            predicted_class, confidence, all_predictions = predict_disease(
                                model, image_array, class_names
                            )
                            
                            # Display results
                            st.success("Prediction Complete!")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("Predicted Disease", predicted_class)
                                st.metric("Confidence", f"{confidence*100:.2f}%")
                            
                            with col2:
                                # Show all class probabilities
                                st.subheader("All Predictions:")
                                for class_name, prob in zip(class_names, all_predictions):
                                    st.write(f"{class_name}: {prob*100:.2f}%")
                            
                            # Recommendation
                            if predicted_class != "Healthy":
                                st.warning(f"⚠️ **Alert**: Plant appears to have {predicted_class}")
                                st.info("**Recommendation**: Consult an agricultural expert for treatment options.")
                            else:
                                st.success("✅ Plant appears healthy!")
    
    # ============== BATCH PROCESSING ==============
    elif selected_page == "Batch Processing":
        st.header("Batch Image Processing")
        
        st.info("Upload multiple images to process them in batch")
        
        uploaded_files = st.file_uploader(
            "Choose multiple plant images",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            # Get available models
            available_models = []
            if saved_models_dir.exists():
                available_models = [f.name for f in saved_models_dir.glob("*.h5")]
            
            if not available_models:
                st.warning("No trained models found in saved_models directory.")
            else:
                selected_model = st.selectbox("Select Model:", available_models)
                class_names = st.text_input(
                    "Class Names (comma-separated):",
                    value="Healthy,Disease1,Disease2,Disease3"
                ).split(",")
                class_names = [name.strip() for name in class_names]
                
                if st.button("🔍 Process Batch"):
                    model_path = saved_models_dir / selected_model
                    model = load_model(str(model_path))
                    
                    if model is not None:
                        results = []
                        progress_bar = st.progress(0)
                        
                        for idx, uploaded_file in enumerate(uploaded_files):
                            image = Image.open(uploaded_file)
                            image_array = preprocess_image(image, img_size=224)
                            predicted_class, confidence, _ = predict_disease(
                                model, image_array, class_names
                            )
                            
                            results.append({
                                'filename': uploaded_file.name,
                                'prediction': predicted_class,
                                'confidence': f"{confidence*100:.2f}%"
                            })
                            
                            progress_bar.progress((idx + 1) / len(uploaded_files))
                        
                        # Display results table
                        st.subheader("Batch Results:")
                        import pandas as pd
                        results_df = pd.DataFrame(results)
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="batch_results.csv",
                            mime="text/csv"
                        )
    
    # ============== MODEL INFORMATION ==============
    elif selected_page == "Model Information":
        st.header("Model Information")
        
        # Get available models
        available_models = []
        if saved_models_dir.exists():
            available_models = [f.name for f in saved_models_dir.glob("*.h5")]
        
        if not available_models:
            st.warning("No trained models found.")
        else:
            selected_model = st.selectbox("Select Model:", available_models)
            
            model_path = saved_models_dir / selected_model
            model = load_model(str(model_path))
            
            if model is not None:
                st.subheader("Model Summary:")
                
                # Display model info
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Parameters", f"{model.count_params():,}")
                
                with col2:
                    st.metric("Layers", len(model.layers))
                
                with col3:
                    st.metric("Model Name", selected_model)
                
                # Display architecture
                st.subheader("Model Architecture:")
                model_summary = []
                model.summary(print_fn=lambda x: model_summary.append(x))
                st.code('\n'.join(model_summary), language='text')


if __name__ == "__main__":
    main()
