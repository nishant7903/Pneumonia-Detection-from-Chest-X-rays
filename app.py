import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.cm as cm
from flask import Flask, render_template, request,url_for
from tensorflow.keras.models import load_model

# Ensure uploads folder exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model
model = load_model('pneumonia_classifier.h5')

# Preprocess image for prediction
def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Generate Grad-CAM heatmap
def generate_gradcam(img_path, model, layer_name='conv2d_4'):
    import tensorflow as tf
    import matplotlib.cm as cm

    # Load and preprocess image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img_array = tf.convert_to_tensor(np.expand_dims(img / 255.0, axis=0), dtype=tf.float32)

    # Rebuild functional model from Sequential
    input_shape = (224, 224, 3)
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    for layer in model.layers:
        x = layer(x)
        if layer.name == layer_name:
            target_output = x
    outputs = x
    grad_model = tf.keras.Model(inputs=inputs, outputs=[target_output, outputs])

    # Use GradientTape to record operations
    with tf.GradientTape() as tape:
        tape.watch(img_array)
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise ValueError("Gradients could not be computed. Check layer name and model structure.")

    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1)

    # Normalize and convert to heatmap
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (224, 224))

    heatmap = cm.jet(cam)[:, :, :3]
    heatmap = np.uint8(255 * heatmap)
    overlay = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0.6, heatmap, 0.4, 0)

    # ✅ These lines must be indented
    filename = 'gradcam_' + os.path.basename(img_path)
    gradcam_path = os.path.join('static', filename)
    cv2.imwrite(gradcam_path, overlay)

    # Return a URL that Flask can serve
    return url_for('static', filename=filename)






# Main route
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    gradcam = None

    if request.method == 'POST':
        file = request.files['image']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        img = preprocess_image(filepath)
        result = model.predict(img)[0][0]
        prediction = "Pneumonia Detected" if result > 0.5 else "Normal"

        gradcam = generate_gradcam(filepath, model)

    return render_template('index.html', prediction=prediction, gradcam=gradcam)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
