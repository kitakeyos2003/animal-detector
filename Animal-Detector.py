import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import ImageTk, Image

# Load the trained model
model = tf.keras.models.load_model("animal_classifier_model.keras")

# Define class names
class_names = ["Bear", "Bird", "Cat", "Cow", "Deer", "Dog", "Dolphin", "Elephant", "Giraffe", "Horse", "Kangaroo", "Lion", "Panda", "Tiger", "Zebra"]

def classify_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    if confidence > 0.5:
        result_text = f"This image is a {class_names[predicted_class]} with {confidence * 100:.2f}% confidence."
    else:
        result_text = "This image is not an animal."
    result_label.config(text=result_text)

def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        img = Image.open(file_path)
        img = img.resize((200, 200))
        img_tk = ImageTk.PhotoImage(img)
        img_label.config(image=img_tk)
        img_label.image = img_tk
        classify_image(file_path)

# Set up the GUI
root = tk.Tk()
root.title("Animal Classifier")
root.geometry("400x400")

# Image display
img_label = Label(root)
img_label.pack(pady=10)

# Result display
result_label = Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

# Button to open file
select_button = Button(root, text="Select an Image", command=open_file)
select_button.pack(pady=10)

root.mainloop()
