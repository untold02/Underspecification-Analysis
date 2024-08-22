import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, jsonify
import base64
from PIL import Image
from skimage.segmentation import mark_boundaries
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
import os
import contextlib
from flask import Flask, render_template, jsonify, request
import cv2
import io
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
from collections import Counter
from tensorflow.keras.utils import plot_model


model1_path = 'cnn_mnist.h5'
model2_path = 'mobilenet_mnist.h5'
model3_path = 'densenet_mnist.h5'

# Load models
try:
    model1 = tf.keras.models.load_model(model1_path)
    print("Model 1 loaded successfully")
except Exception as e:
    print(f"Failed to load model 1: {e}")

try:
    model2 = tf.keras.models.load_model(model2_path)
    print("Model 2 loaded successfully")
except Exception as e:
    print(f"Failed to load model 2: {e}")

try:
    model3 = tf.keras.models.load_model(model3_path)
    print("Model 3 loaded successfully")
except Exception as e:
    print(f"Failed to load model 3: {e}")


# Store models in a dictionary if successfully loaded
models = {
	'CNN': model1 if 'model1' in locals() else None,
	'MobileNet': model2 if 'model2' in locals() else None,
	'DenseNet': model3 if 'model3' in locals() else None,
}

models['CNN']._name = 'CNN'
models['MobileNet']._name = 'MobileNetV2'
models['DenseNet']._name = 'DenseNet121'

def preprocess_image(image):
    if type(image) != np.ndarray:
        image = np.array(image)

    if image.dtype != np.float32:
        image = image.astype(np.float32)

    if np.max(image) > 1:
        image = image / np.max(image)

    image = tf.image.resize(image,(32,32))
    image = np.reshape(image, (1,32,32,3))

    return image

def prediction_fn(images, model):
    images = np.array(images)

    if len(images.shape) == 3: 
        images = images.reshape(-1, 32, 32, 3)
    elif len(images.shape) == 4: 
        if images.shape[1:] != (32, 32, 3):
            raise ValueError("Expected images with shape (32, 32, 3)")

    predictions = model.predict(images, verbose=0)

    return predictions


segmenter_quick = SegmentationAlgorithm("slic", kernal_size=3, ratio=0.2) 

explainer = lime_image.LimeImageExplainer(verbose=False, random_state =42)

def get_lime_explanation(image, model, no_of_samples):
    with open(os.devnull, 'w') as f:
        with contextlib.redirect_stdout(f):
            explanation = explainer.explain_instance(image,
                                                    lambda x: prediction_fn(x, model),
                                                    top_labels = 3,
                                                    hide_color = 0,
                                                    num_samples = no_of_samples,
                                                    segmentation_fn=segmenter_quick
                                                            )
            
            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], 
                                                        positive_only=True,
                                                        num_features=5,
                                                        hide_rest=True)
            
            img_boundry = mark_boundaries(temp, mask)

            return np.array(img_boundry)
        

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image'].split(',')[1]

    # Decode the base64 image data
    decoded_image = base64.b64decode(image_data)
    image_array = np.frombuffer(decoded_image, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    processed_image = preprocess_image(image)

    explanations = {}
    predictions = {}
    predicted_labels = []
    feature_vectors = []
    model_names = []

    for model_name, model in models.items():
        prediction = model.predict(processed_image.reshape(-1, 32, 32, 3))
        predicted_digit = np.argmax(prediction)
        probability = np.max(prediction)

        predictions[model_name] = {
            'label' : int(predicted_digit),
            'probability' : float(probability)
        }

        predicted_labels.append(predicted_digit)
        model_names.append(model_name)
        print(model_names)

        # Get LIME explanation
        lime_explanation = get_lime_explanation(processed_image.squeeze(), model, 1000)
        feature_vectors.append(lime_explanation.flatten())

        # Convert the LIME explanation to a PIL image and then to Base64
        pil_img = Image.fromarray((lime_explanation * 255).astype(np.uint8))
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        explanation_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        explanations[model_name] = explanation_base64

    # Calculate cosine similarity matrix
    sim = cosine_similarity(feature_vectors)
    cosine_distances = 1 - sim

    model_labels = ['CNN', 'MobileNet', 'DenseNet']
    most_common_label, _ = Counter(predicted_labels).most_common(1)[0]

    plt.figure(figsize=(8, 8))
    sns.heatmap(cosine_distances, annot=True, fmt=".2f", cmap="viridis",
                xticklabels=model_labels, yticklabels=model_labels, cbar_kws={'label': 'Cosine Distance'})
    plt.title(f"Pairwise Cosine Distance Confusion Matrix for Digit {most_common_label}")

    buf_mat = io.BytesIO()
    plt.savefig(buf_mat, format='png')
    buf_mat.seek(0)
    plt.close()

    encoded_mat = base64.b64encode(buf_mat.getvalue()).decode('utf-8')

    response = {
        'predictions': predictions,
        'explanations': explanations,
        'cosine_distance_img': encoded_mat,
        'model_names' : model_names
    }

    return jsonify(response)

@app.route('/model-summary', methods=['GET'])
def model_summary():
    model_name = request.args.get('model')
    if model_name in models:
        model = models[model_name]

        # Generate and save the model architecture as an image
        plot_model(model, to_file='arc_img.png', show_shapes=True, show_layer_names=True, dpi=96)
        buf_arc = io.BytesIO()
        arc_img = plt.imread('arc_img.png')
        plt.imshow(arc_img)
        plt.axis('off')
        plt.savefig(buf_arc, format='png', bbox_inches='tight')
        buf_arc.seek(0)

        model_plot_base64 = base64.b64encode(buf_arc.getvalue()).decode('utf-8')

        os.remove('arc_img.png') #remove the temporary img

        summary_io = io.StringIO()
        model.summary(print_fn=lambda x: summary_io.write(x + '\n'))
        model_summary_str = summary_io.getvalue()
        

        return jsonify({
            "model_plot": model_plot_base64,
            "model_summary": model_summary_str
            })
                 
    else:
        return jsonify({"error": "Model not found"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
