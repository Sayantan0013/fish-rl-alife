# %% CELL 1
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np
import io

import os
print(os.getcwd())
# %% CELL 2
# Specify the directory where your TensorBoard logs are stored
log_dir = 'analysis_data/' # Replace with your actual log directory

# Initialize EventAccumulator
ea = EventAccumulator(log_dir, size_guidance={
    event_accumulator.IMAGES: 0  # 0 means load all images
})

# Reload the events from the log files.
ea.Reload()

print("=" * 80)
print("COMPLETE EVENTACCUMULATOR CONTENTS")
print("=" * 80)

# Print basic information about the EventAccumulator
print(f"\nLog directory: {log_dir}")
print(f"EventAccumulator object: {ea}")

# Print all available tags
print("\n" + "="*50)
print("ALL AVAILABLE TAGS")
print("="*50)
tags = ea.Tags()
print(f"Tags object: {tags}")
print(f"Available tag types: {list(tags.keys())}")
# %% CELL
# Print image data
#
all_images = []
attr = "query"

for i in range(5):
    image_events = ea.Images(f'{attr}/run_{i}')
    print(f"  Number of image events: {len(image_events)}")
    for image_event in image_events:
        image = np.array(Image.open(io.BytesIO(image_event.encoded_image_string))).mean(axis=2)
        all_images.append(image)
# %% CELL
cumulted_messages = np.array(all_images).reshape(-1, all_images[0].shape[-1])

from sklearn.decomposition import PCA
pca = PCA(n_components=2)

principal_components = pca.fit_transform(cumulted_messages)

print(f"Shape of principal components: {principal_components.shape}")
plt.figure(figsize=(10, 8))

plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.8)
plt.show()
# %% CELL


index = 1
print(np.array(all_images).shape)
print(np.array(all_images)[:,index,:].shape)
agent_i = pca.transform(np.array(all_images)[:,index,:])

print(f"Shape of principal components: {principal_components.shape}")
plt.figure(figsize=(10, 8))

plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.8)
plt.scatter(agent_i[:, 0], agent_i[:, 1], alpha=0.8)

plt.show()
