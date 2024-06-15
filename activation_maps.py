import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
import cv2
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk

# Load the pre-trained model
model = load_model('Age_sex_Detection.keras')

# Print the model summary to find the correct layer name
model.summary()

# Select a convolutional layer for visualization
layer_name = 'conv2d_1'  # Replace with your actual layer name
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

# Load and preprocess the image
img_path = 'C:/Users/matha/OneDrive/Desktop/C PROGRAM/gender_age/UTKFace/1_0_0_20161219140623097.jpg.chip.jpg'  # Replace with your actual image path
img = image.load_img(img_path, target_size=(48, 48))  # Adjust target_size if needed
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # Normalize the image if needed

# Generate activation maps
intermediate_output = intermediate_layer_model.predict(img_array)

# Explanation to be displayed
explanation_text = """
The activation maps shown here represent the response of different filters in the convolutional layer '{0}'.
Bright areas in the maps indicate strong activation, meaning these regions are important for the model's prediction.
By examining these maps, we can understand which features the model focuses on, such as edges, textures, or specific facial features.
Colors indicate the activation intensity:
- Dark blue: Low activation
- Light blue to green: Medium activation
- Yellow to red: High activation
""".format(layer_name)

# Display explanations in the terminal
print(explanation_text)

# Plot the activation maps and create GUI
num_filters = intermediate_output.shape[-1]

# Create a simple GUI window
root = tk.Tk()
root.title("Activation Maps and Color Bar")
root.geometry("1200x900")

# Display the explanation text in the GUI
explanation_label = Label(root, text=explanation_text, wraplength=1100, justify="left")
explanation_label.pack()

# Display activation maps in the GUI
canvas = tk.Canvas(root, width=1100, height=700)
canvas.pack()

# Prepare activation maps for display
activation_images = []
epsilon = 1e-10  # Small value to prevent division by zero
for i in range(num_filters):
    activation_map = intermediate_output[0, :, :, i]
    activation_map -= activation_map.mean()
    activation_map /= (activation_map.std() + epsilon)  # Add epsilon to prevent division by zero
    activation_map *= 64
    activation_map += 128
    activation_map = np.clip(activation_map, 0, 255).astype('uint8')
    
    # Overlay activation map on the original image
    heatmap = cv2.resize(activation_map, (img_array.shape[2], img_array.shape[1]))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img_array[0] * 255
    activation_images.append(ImageTk.PhotoImage(image=Image.fromarray(superimposed_img.astype('uint8'))))

# Arrange activation maps in the GUI
rows = num_filters // 8 + 1
for i in range(num_filters):
    row = i // 8
    col = i % 8
    canvas.create_image(col * 120, row * 120, anchor='nw', image=activation_images[i])

# Display color bar for activation intensity
color_bar_img = Image.open('colorbar.png')
color_bar_img = color_bar_img.resize((30, 200))  # Adjust size if needed
color_bar_tk_img = ImageTk.PhotoImage(color_bar_img)
color_bar_label = Label(root, image=color_bar_tk_img)
color_bar_label.pack(side='right', padx=10, pady=10)

root.mainloop()
