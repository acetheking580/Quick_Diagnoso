import cv2
import numpy as np
from sklearn.metrics import pairwise_distances
import os
import pandas as pd

data = pd.read_csv('autism/autistic_quiz/data.csv')

user_values = {}
user_values['Symptom 1'] = int(
    input("Enter your value for Symptom 1: "))
user_values['Symptom 2'] = int(
    input("Enter your value for Symptom 2: "))
user_values['Symptom 3'] = int(
    input("Enter your value for Symptom 3: "))
user_values['Symptom 4'] = int(
    input("Enter your value for Symptom 4: "))
user_values['Symptom 5'] = int(
    input("Enter your value for Symptom 5: "))

data['Distance'] = np.sqrt(
    np.sum((data.iloc[:, :-1] - list(user_values.values()))**2, axis=1))
sorted_data = data.sort_values(by='Distance')

closest_entry = sorted_data.iloc[0]

result = closest_entry['Autism Spectrum']

if result == "Likely in Spectrum":
    print("Do whatever in the GUI")
    autism1 = 1
elif result == "Not in Spectrum":
    print("You may not be on the autism spectrum, but if you have concerns, consult a healthcare professional for further evaluation.")
    autism1 = 0
else:
    print("Code to rely solely on the picture scanning")
    autism1 = 3

def load_and_compute_descriptors(dataset_path):
    dataset = []
    for image_filename in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, image_filename)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hog = cv2.HOGDescriptor()
        descriptors = hog.compute(gray)
        dataset.append(descriptors.flatten())
    
    return np.array(dataset)

non_autistic_dataset_path = r'autism\non_autistic'
autistic_dataset_path = r'autism\autistic'

non_autistic_dataset = load_and_compute_descriptors(non_autistic_dataset_path)
autistic_dataset = load_and_compute_descriptors(autistic_dataset_path)

distances = pairwise_distances(non_autistic_dataset, autistic_dataset, metric='euclidean')

threshold = 0.4

matching_images = np.where(distances < threshold)
matching_image_pairs = [(i, j) for i, j in zip(*matching_images)]


if matching_image_pairs:
    print("Matching images found:")
    for i, j in matching_image_pairs:
        non_autistic_image_path = os.path.join(non_autistic_dataset_path, os.listdir(non_autistic_dataset_path)[i])
        autistic_image_path = os.path.join(autistic_dataset_path, os.listdir(autistic_dataset_path)[j])
        print(f"Non-autistic image ({non_autistic_image_path}) matches with autistic image ({autistic_image_path})")
else:
    print("No matching images found.")

def classify_image(image_path, threshold):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor()
    descriptors_to_classify = hog.compute(gray).flatten()
    distances_to_autistic = pairwise_distances(descriptors_to_classify.reshape(1, -1), autistic_dataset, metric='euclidean')
    distances_to_non_autistic = pairwise_distances(descriptors_to_classify.reshape(1, -1), non_autistic_dataset, metric='euclidean')
    
    if np.min(distances_to_autistic) < threshold and np.min(distances_to_autistic) < np.min(distances_to_non_autistic):
        return "autistic"
    elif np.min(distances_to_non_autistic) < threshold and np.min(distances_to_non_autistic) < np.min(distances_to_autistic):
        return "non-autistic"
    else:
        return "unclassified"

image_to_classify_path = 'image.jpg'

classification_result = classify_image(image_to_classify_path, threshold)

if classification_result == "autistic":
    print(f"The image at {image_to_classify_path} is classified as autistic.")
    autism_1 = 1
elif classification_result == "non-autistic":
    print(f"The image at {image_to_classify_path} is classified as non-autistic.")
    autism_1 = 0
else:
    print(f"The image at {image_to_classify_path} could not be definitively classified.")
    autism_1 = 2

if autism_1 == 1 == autism1:
    autism = "True"

if autism_1 == 0 == autism1:
    autism = "False"

if autism_1 == autism1 - 1:
    autism = "Uncertain"

if autism_1 == 2:
    autism = "True"
