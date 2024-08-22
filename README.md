# Underspecification-Analysis
## Overview:
This is a demo version of our work on analyzing underspecification in deep learning models, leveraging the post-hoc explanation tool LIME (Local Interpretable Model-agnostic Explanations). 
The tool is designed to provide insight into how different deep learning models behave when presented with the same input data, specifically focusing on digit classification.

## Key Features:

* **Interactive Analysis**: 
Users can draw digits directly on the canvas (e.g., 0 ~ 9), submit them for prediction, and see how different models interpret the input.

* **Model Comparison**: 
The tool allows you to compare the predictions and explanations of several deep learning models side by side, giving a clear view of how each model's decision-making process differs.

* **LIME Explanations**:
For each model, the tool generates LIME explanations, which visually highlight the areas of the input that were most influential in the model's prediction. This helps in understanding which parts of the input data each model relies on.

* **Focus on Underspecification**: 
This tool is particularly useful for exploring the concept of underspecification—where models may produce the same outputs despite learning different underlying features—by providing a detailed comparison of how various models interpret the same input data.

* **Purpose**:
The tool is intended to be a hands-on way for users to explore and analyze the concept of underspecification in deep learning models, particularly within the context of the digit classification.
By interacting with the tool, users can gain a deeper understanding of how different models approach the same problem and the potential consequences of underspecification in model training and deployment.

## Step-by-Step Guide on how to use the web app:

1. **Draw a Digit**:
    * Navigate to the canvas area titled "Draw a Digit".
    * Use your mouse to draw a digit inside the canvas (e.g., 0 to 9).
    * Once you finishied writing a digit, click the Submit button.
  
2. **View Predictions and Explanations**:
    * After you submit your drawing, the system will process the image using multiple pre-trained models (CNN, MobileNet, and DenseNet).
    * You will see the predictions from each model displayed below their respective LIME Explanations.
    * For each model, the predicted digit and the associated probability will be shown in a caption beneath the explanation image.
   
3. **Model Comparion**:
    * The explanations displayed show which parts of the image each model considered important when making its prediction.
    * By comparing the explanations across models, you can identify differences in how the models interpret the same input image, highlighting potential underspecification (where different models make similar predictions for different reasons).

4. **Pairwise Cosine Distance**:
    * Below the explanations, the tool will display a Pairwise Cosine Distance Confusion Matrix
    * This matrix shows how different the models are in their internal feature representations based on their LIME explanations.
    * A lower value in the matrix indicates that two models are more similar in behavior, while a higher value indicates greater divergence in their feature space, suggesting underspecification.
   
5. **Model Summary and Architecture Visualization**:
    * On the right side of the page, you can explore the architecture and complexity of each model. By default it shows the total number of parameters and the architecture of the CNN model.
    * To view the model complexity and architecture of different models, select the model from the **Select Model** dropdown menu under the **Model Architecture section**.
    * Once selected, the models summary and an image representing the model’s architecture will appear, helping you understand the structural differences between models.

6. **Analyze Underspecification**:
    * Compare the predictions, explanations, and cosine distance between models.
    * Identify cases where different models make similar predictions but focus on different parts of the input image (if the variation is significant, a sign of underspecification).
    * Use the confusion matrix to quantify the level of divergence between model behaviors.

## Additional Notes:
* **Clear the Canvas** : If you want to submit another digit, click the Clear button to erase the canvas before drawing a new digit. This will clear the left panal of the screen inlcuding LIME explanations and confusion matrix.
* **Use Cases** : This tool can be useful for identifying underspecification in machine learning models by comparing how different models arrive at their predictions.
      It's especially beneficial in analyzing robustness, interpretability, and generalization behavior in models.
* **Limiataions** : Keep in mind that this tool relies on visual explanations (LIME) and cosine distance comparisons. These methods provide insight into model behavior but should be supplemented with other analysis techniques for a complete evaluation of underspecification.

## Model Overview:
Here’s the refined version:

---

## Model Overview:
In this demo, we used three deep learning models with varying levels of architectural complexity: Custom CNN, MobileNetV2, and DenseNet121.
All of these models were selected based on their diverse architectural design and their performance during the training and validation phases. Where their performances were higher than 0.90 within 10 epochs, meeting our threshold criteria. These models, while performing similarly during training, were chosen to highlight different aspects of underspecification by demonstrating how models with distinct architectures and complexities interpret the same input data differently during generalization. The diversity in architecture and complexity among these three models makes them well-suited for underspecification analysis. By comparing how each model interprets the same input, we can identify instances where different models, despite producing similar outputs, rely on different parts of the input data. This divergence in model interpretation is a key indicator of underspecification, where the models' learned features may not fully capture the true underlying patterns in the data.
Overall, the chosen models provide a comprehensive framework for exploring underspecification, offering insights into how different levels of model complexity and architectural design influence the learning process and the resulting predictions.

## **Custom CNN:**
* **Simplicity and Interpretability**: The Custom CNN is a straightforward, easy-to-understand model trained from scratch on the MNIST dataset. 
Its architecture consists of a few convolutional layers, making it less complex compared to other deep learning models. 
This simplicity allows us to clearly observe how the model processes input data and makes predictions.
* **Baseline for Comparison**: As a simple model, the Custom CNN serves as a baseline for comparison with more sophisticated models like MobileNetV2 and DenseNet121.
By comparing the Custom CNN's behavior with that of more complex models, we can investigate whether the simplicity of the model leads to different interpretations of the same input,
thus providing insights into underspecification.

## **MobileNetV2:**
* **Efficiency and Compactness**: MobileNetV2 is a lightweight model optimized for mobile and edge devices. It utilizes depthwise separable convolutions to reduce the number of parameters 
and computational requirements, all while maintaining high accuracy. This model's design makes it interesting for underspecification analysis, as its architecture is focused on efficiency 
rather than complexity.

* **Architectural Diversity**: MobileNetV2 introduces a different architectural paradigm compared to the Custom CNN, particularly through its use of inverted residuals and linear bottlenecks. 
Analyzing this model alongside the others helps us explore how architectural innovations aimed at efficiency might impact the model's interpretability and behavior, potentially leading to 
different forms of underspecification.

## **DenseNet121:**
* **Complexity and Feature Reuse**: DenseNet121 is the most complex model in this demo, characterized by its dense connectivity pattern where each layer receives input from all previous layers. 
This design promotes feature reuse, reduces the need to relearn redundant features, and improves gradient flow.
* **Understanding Complex Models**: Due to its complexity, DenseNet121 can capture more intricate patterns in the data, making it ideal for analyzing whether complex models are more prone to 
underspecification. By comparing DenseNet121's behavior with that of simpler models, we can assess whether complexity leads to more robust or more divergent model interpretations.

