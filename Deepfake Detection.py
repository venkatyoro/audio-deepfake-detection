#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install torch torchaudio transformers datasets librosa numpy pandas scikit-learn


# ## Load the Pretrained Wav2Vec 2.0 Model

# In[2]:


from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import torch

# Load pre-trained processor and model
model_name = "facebook/wav2vec2-base"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 2 labels: Real or Deepfake


# In[3]:


import librosa
import numpy as np
import torch

# Function to load and preprocess audio
def preprocess_audio(file_path):
    speech_array, sampling_rate = librosa.load(file_path, sr=16000)
    inputs = processor(speech_array, sampling_rate=16000, return_tensors="pt", padding=True)
    return inputs.input_values[0]

# Example: Load a sample file
sample_audio_path = "fake4_3.wav"
audio_features = preprocess_audio(sample_audio_path)


# ### Create the Training Data Loader

# In[4]:


from torch.utils.data import Dataset, DataLoader

class DeepfakeAudioDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels
    
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        audio_features = preprocess_audio(self.file_paths[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return audio_features, label

# Example: File paths and labels (1 = real, 0 = deepfake)
file_paths = ["speaker1_1.wav", "fake1_1.wav", "speaker1_2.wav", "fake1_2.wav"]
labels = [1, 0, 1, 0]

# Create dataset
dataset = DeepfakeAudioDataset(file_paths, labels)

# Data loader
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)


# In[5]:


from transformers import Wav2Vec2ForSequenceClassification

# Define your model (adjust based on your use case)
model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", num_labels=2)


# ### Define Training Loop
Use Adam optimizer and cross-entropy loss.
# In[6]:


import torch
import torch.nn.functional as F

def train_model(model, train_loader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()  # Set model to training mode
    
    for epoch in range(epochs):
        for batch in train_loader:
            inputs, labels = batch

            # Debug input shape
            print(f"Before Processing - Input Shape: {inputs.shape}, Data Type: {inputs.dtype}")

            # Ensure correct shape
            inputs = inputs.squeeze(1)  # Remove channel dim if present
            inputs = inputs.to(torch.float32)  # Ensure correct dtype

            # Debug after processing
            print(f"After Processing - Input Shape: {inputs.shape}")

            # Forward pass
            outputs = model(inputs)

            # Check if model output is a dict
            if isinstance(outputs, dict) and "logits" in outputs:
                outputs = outputs["logits"]

            loss = F.cross_entropy(outputs, labels.long())  # Ensure labels are long type

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")




# In[7]:


pip install dagshub


# In[9]:


# Train the model
train_model(model, train_loader, epochs=5)


# ### Evaluate the Model

# In[11]:


def predict(audio_file):
    model.eval()
    inputs = preprocess_audio(audio_file).unsqueeze(0)
    with torch.no_grad():
        logits = model(inputs).logits
        prediction = torch.argmax(logits, dim=-1).item()
    return "Real" if prediction == 1 else "Deepfake"


print(predict("fake1_1.wav"))


# In[12]:


def predict(audio_file):
    model.eval()
    inputs = preprocess_audio(audio_file).unsqueeze(0)
    with torch.no_grad():
        logits = model(inputs).logits
        prediction = torch.argmax(logits, dim=-1).item()
    return "Real" if prediction == 1 else "Deepfake"


print(predict("speaker1_1.wav"))


# ### Save the Model

# In[13]:


model.save_pretrained("wav2vec2-deepfake-detector")
processor.save_pretrained("wav2vec2-deepfake-detector")


# In[ ]:




