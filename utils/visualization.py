import matplotlib.pyplot as plt
import numpy as np

def plot_embeddings(embeddings, labels, title='Speaker Embeddings'):
    """
    Visualizes the speaker embeddings using a scatter plot.

    Parameters:
    - embeddings: A 2D numpy array of shape (n_samples, n_features) containing the embeddings.
    - labels: A 1D array of labels corresponding to the embeddings.
    - title: Title of the plot.
    """
    unique_labels = np.unique(labels)
    plt.figure(figsize=(10, 8))

    for label in unique_labels:
        idx = np.where(labels == label)
        plt.scatter(embeddings[idx, 0], embeddings[idx, 1], label=f'Speaker {label}')

    plt.title(title)
    plt.xlabel('Embedding Dimension 1')
    plt.ylabel('Embedding Dimension 2')
    plt.legend()
    plt.grid()
    plt.show()

def plot_rttm(rttm_data, title='RTTM Visualization'):
    """
    Visualizes the RTTM data as a timeline.

    Parameters:
    - rttm_data: A list of tuples containing (start_time, end_time, speaker_id).
    - title: Title of the plot.
    """
    plt.figure(figsize=(12, 6))

    for start, end, speaker in rttm_data:
        plt.plot([start, end], [speaker, speaker], linewidth=6)

    plt.title(title)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Speaker ID')
    plt.yticks(np.unique([speaker for _, _, speaker in rttm_data]))
    plt.grid()
    plt.show()