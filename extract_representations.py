import numpy as np
from collections import defaultdict

import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Speech2Text2Processor, SpeechEncoderDecoderModel
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.manifold import TSNE

import os
import librosa
import pickle


# helper funtions 
# 1. extract representations from model
def get_layer_representations(speech_samples, model):
    """
    Get layer representations for speech samples using a given model.

    Parameters:
    - speech_samples (dict): A dictionary containing speech samples.
    - model: The model used for obtaining layer representations.

    Returns:
    - layer_reprs (defaultdict): A dictionary containing layer representations for each speech sample.
    """
    layer_reprs = defaultdict()

    for i, sample_ID in enumerate(speech_samples):

        # Tokenize and forward pass through the model
        processed_sample = processor(
            speech_samples[sample_ID].squeeze().numpy(), 
            return_tensors="pt", 
            sampling_rate=16000
        ).input_values

        with torch.no_grad():
            all_layers_reprs = model(processed_sample, output_hidden_states=True).hidden_states

        num_layers = len(all_layers_reprs)

        layer_reprs[sample_ID] = [
            all_layers_reprs[layer_idx] for layer_idx in range(0, num_layers) #.numpy()
        ]

        print(f"{(100*i)/len(speech_samples):.2f}% completed!", end='\r')

        #if i == 100:
            #break   

    return layer_reprs


# 2. compute t-SNE 
def calculate_tsne_layer_representations(layer_reprs):
    """
    Calculates T-SNE layer representations for each layer in the given layer representations.

    Args:
        layer_reprs (dict): A dictionary containing layer representations for each sample ID.

    Returns:
        list: A list of T-SNE layer representations for each layer.
    """
    tsne_layer_representations = []

    num_layers = len(layer_reprs[list(layer_reprs.keys())[0]])

    for layer_index in range(0, num_layers):
        # Extract representations from the specified layer
        layer_representations = []

        for sample_ID in layer_reprs:
            layer_representations.append(layer_reprs[sample_ID][layer_index].mean(dim=1).squeeze().numpy())

        # Convert the representations to a NumPy array
        layer_representations = np.array(layer_representations)

        # Perform T-SNE dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42)
        tsne_representation = tsne.fit_transform(layer_representations)

        tsne_layer_representations.append(tsne_representation)

    return tsne_layer_representations


# 3. Visualize and plot t-SNE

def plot_tsne(tsne_representation, labels, label2word, figsave_name):
    """
    Plots 2D t-SNE representations for each layer.

    Parameters:
    - tsne_representation (list of numpy arrays): List of t-SNE representations for each layer.
    - labels (numpy array): Array of labels for each data point.
    - label2word (dict): Dictionary mapping label indices to corresponding words.

    Returns:
    - None

    This function plots t-SNE representations for each layer in a grid of subplots. Each subplot represents a layer,
    and the t-SNE representations are plotted as scatter plots. The color of each data point in the scatter plot
    corresponds to its label. The legend at the bottom of the plot shows the color labels and their corresponding words.
    """
    
    # Define the number of rows and columns for the subplot matrix
    num_rows = len(tsne_representation) // 6
    num_cols = 6

    # Create a new figure and set the size
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 2*num_rows))

    # Flatten the axes array
    axes = axes.flatten()

    # Iterate over the layers and plot t-sne representations
    for i, tsne_rep in enumerate(tsne_representation):

        # Plot the representations in the current subplot
        scatter = axes[i].scatter(tsne_rep[:, 0], tsne_rep[:, 1], c=labels, cmap='tab10', alpha=0.1, marker='.')
        axes[i].set_title(f'Layer {i+1}', fontsize=10)
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    # Remove any extra subplots
    for j in range(len(axes)):
        if j >= len(tsne_representation):
            fig.delaxes(axes[j])

    # Adjust the spacing between subplots
    #fig.tight_layout()

    # Add legend with color labels at the bottom
    legend_labels = [label2word[i].upper() for i in np.unique(labels)]

    legend_handles = [
        plt.Line2D(
            [0], 
            [0], 
            marker='.', 
            color='w', 
            markerfacecolor=scatter.get_cmap()(scatter.norm(label)), markersize=10) for label in np.unique(labels)
        ]

    fig.legend(legend_handles, legend_labels, loc='lower center', ncol=len(legend_labels), bbox_to_anchor=(0.5, -0.1))

    fig.tight_layout()
    fig.savefig('Figures/' + figsave_name, dpi=300)

    plt.show()

# read all speech files 
print("Read audio files from desk...")
audio_folder = 'spoken_digit/speech_data/'
audio_files = []

for file in os.listdir(audio_folder):
    if file.endswith('.wav') or file.endswith('.mp3'):
        audio_files.append(os.path.join(audio_folder, file))

# collect metadata
print("Collect metadata...")
audio_data = {}
metadata = {}

for file in audio_files:
    file_name = os.path.basename(file)
    label, speaker_id, sample_number = file_name.split('_')
    
    # Read audio file
    audio, sr = librosa.load(file, sr=8000)
    
    # Store audio data
    audio_data[file_name] = audio
    
    # Store metadata
    metadata[file_name] = {
        'label': label,
        'speaker_id': speaker_id,
        'sample_number': sample_number.split('.')[0]
    }


# apply model-specific preprocessing...
print("Apply preprocessing...")
resampled_samples = defaultdict()

for i, sample_ID in enumerate(metadata):


    # Get audio data
    file_name = audio_folder + '/' + sample_ID

    waveform, sample_rate = torchaudio.load(file_name)
    
    resampler = torchaudio.transforms.Resample(8_000, 16_000) 

    waveform = resampler(waveform)

    resampled_samples[sample_ID] = waveform

    print(f"{(100*i)/len(audio_files):.2f}%", end='\r')


# Load pre-trained model and processor
#w2v2_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60")
w2v2_model = SpeechEncoderDecoderModel.from_pretrained("facebook/s2t-wav2vec2-large-en-de")    
w2v2_model = w2v2_model.encoder
processor = Speech2Text2Processor.from_pretrained("facebook/s2t-wav2vec2-large-en-de")
#processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-lv60")



# extract representations...
print("Extract representations...")


layer_reprs = get_layer_representations(resampled_samples, w2v2_model)


# Get metadata in lists 
# Iterate over the layers and plot the representations
speaker_ids, labels = [], []

for sample_ID in layer_reprs:
    speaker_id = metadata[sample_ID]['speaker_id']
    label = int(metadata[sample_ID]['label'])

    speaker_ids.append(speaker_id)
    labels.append(label)


# Specify the file path for saving the pickle object
pickle_file_path = "../wav2vec2_vectors/pt/layer_reprs_large_ft_s2t.pkl"

# Save the layer_reprs_large_ft as a pickle object
print("Save representations to desk...")
with open(pickle_file_path, "wb") as file:
    pickle.dump(layer_reprs, file)


# Calculate mean pooling at the utterance level
layer_reprs_pooled = defaultdict(list)

for sample_ID in layer_reprs:

    num_layers = len(layer_reprs[list(layer_reprs.keys())[0]])

    for layer_index in range(0, num_layers): 

        layer_reprs_pooled[sample_ID].append(
            layer_reprs[sample_ID][layer_index].mean(dim=1).squeeze().numpy()
        )



# Specify the file path for saving the pickle object for pooled representations 
pickle_file_path = "../wav2vec2_vectors/pt/layer_reprs_large_pooled_ft_s2t.pkl"

# Save the layer_reprs_large_ft as a pickle object
print("Save pooled representations to desk...")
with open(pickle_file_path, "wb") as file:
    pickle.dump(layer_reprs_pooled, file)


