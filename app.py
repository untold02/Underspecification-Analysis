import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, jsonify
import base64
from skimage.segmentation import mark_boundaries
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
import os
import contextlib
import cv2
import io
from PIL import Image
import boto3


# def download_model_from_s3(bucket_name, model_key, download_path):
#     if not os.path.exists(download_path):
#         s3 = boto3.client('s3')
#         s3.download_file(bucket_name, model_key, download_path)
#     else:
#         print(f"Model {download_path} already exists. Skipping download.")

# bucket_name = 'myunderspec'
# models_info = {
#     'model1': {'key': 'cnn_mnist_std.h5', 'path': '/tmp/cnn_mnist_std.h5'},
#     'model2': {'key': 'mobilenet_mnist_std.h5', 'path': '/tmp/mobilenet_mnist_std.h5'},
#     # 'model3': {'key': 'resnet_mnist_std.h5', 'path': '/tmp/resnet_mnist_std.h5'},
#     'model3': {'key': 'densenet_mnist_std.h5', 'path': '/tmp/densenet_mnist_std.h5'}
# }

# for model_name, model_info in models_info.items():
#     download_model_from_s3(bucket_name, model_info['key'], model_info['path'])

app = Flask(__name__)

models = {model_name: tf.keras.models.load_model(model_info['path']) for model_name, model_info in models_info.items()}
models = {
    'model1': tf.keras.models.load_model('/std/cnn_mnist_std.h5'),
    'model2': tf.keras.models.load_model('/std/mobilenet_mnist_std.h'),
    # 'model3': tf.keras.models.load_model('/std/resnet_mnist_std.h5'),
    'model3': tf.keras.models.load_model('/std/densenet_mnist_std.h5')
                                        
}
segmenter_quick = SegmentationAlgorithm("slic", kernal_size=3, ratio=0.2) 

explainer = lime_image.LimeImageExplainer(verbose=False, random_state =42)

def preprocess_image(image):
    if type(image) != np.ndarray:
        image = np.array(image)

    if image.dtype != np.float32:
        image = image.astype(np.float32)
    print(f"before Normalize image min: {image.min()}, max: {image.max()}") #-> before Normalize image min: 0.0, max: 0.0

    if np.max(image) > 1:
        image = image / np.max(image)
        print(f"Normalized image min: {image.min()}, max: {image.max()}")

    image = tf.image.resize(image,(32,32))
    print(f"Resized image min: {np.array(image).min()}, max: {np.array(image).max()}")
    print(f'Resized image shape {image.shape}')

    image = np.reshape(image, (1,32,32,3))
    print(f"Final preprocessed image shape: {image.shape}, type: {type(image)} dtype: {image.dtype}")

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
        

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image'].split(',')[1]
    model_name = data['model']

    # Decode the base64 image data
    decoded_image = base64.b64decode(image_data)
    image_array = np.frombuffer(decoded_image, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    processed_image = preprocess_image(image)


    if model_name in models:
        model = models[model_name]
        prediction = model.predict(processed_image.reshape(-1,32,32,3))
        predicted_digit = np.argmax(prediction)

        # Get LIME explanation
        lime_explanation = get_lime_explanation(processed_image.squeeze(), model, 1000)

        # Convert the LIME explanation to a PIL image and then to Base64
        pil_img = Image.fromarray((lime_explanation * 255).astype(np.uint8))
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        explanation_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    else:
        return jsonify({'error': 'Model not found'}), 404

    response = {
        'prediction': {model_name: int(predicted_digit)},
        'explanation': {model_name: explanation_base64}
    }

    return jsonify(response)

if __name__ == '__main__':
    # app.run(debug=False, use_reloader=False)
    app.run(host='0.0.0.0', port=8080)