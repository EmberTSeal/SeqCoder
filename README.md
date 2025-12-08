# SeqCoder

SeqCoder is a novel lossless compressor for prokaryotic genome sequences. It leverages a deep learning autoencoder combined with residual encoding to achieve high compression ratios while ensuring perfect reconstruction of the original data.

The core of SeqCoder is a 1D Convolutional Autoencoder with Self-Attention blocks designed specifically for sequential genomic data.

## How It Works

The compression process is a two-stage pipeline designed for lossless reconstruction:

1.  **Stage 1: Autoencoder Compression (Lossy)**
    *   The input DNA sequence is divided into chunks.
    *   A 1D Convolutional Autoencoder's encoder compresses each chunk into a compact latent representation (float32).
    *   This latent representation is then quantized to 8-bit integers (`int8`) to significantly reduce its size.
    *   Finally, the `int8` latent data is further compressed losslessly using the Blosc library with the Zstandard (zstd) codec.

2.  **Stage 2: Residual Encoding (Lossless Correction)**
    *   The quantized latent data from Stage 1 is passed through the autoencoder's decoder to produce a reconstructed sequence. This reconstruction is an approximation of the original.
    *   The differences (residuals) between the original sequence and the reconstructed sequence are identified.
    *   The positions and original base values of these mismatches are efficiently encoded using a combination of delta encoding and VarInt encoding.
    *   This compact residual data is then compressed using Zstandard.

The final compressed file consists of the compressed latent representation from Stage 1 and the compressed residuals from Stage 2. During decompression, the autoencoder first generates the approximate sequence, which is then corrected using the residual data to achieve a bit-for-bit identical, lossless reconstruction of the original genome.

## Model Architecture

The deep learning model, `ConvolutionalAutoEncoder1D`, is built with PyTorch and consists of:

*   **Encoder**: A series of 1D convolutional layers with `ReLU` activation and batch normalization that progressively downsample the input sequence chunks into a latent space.
*   **Self-Attention Bottleneck**: A self-attention block (`_SelfAttnBlock1D`) is applied to the latent representation. This allows the model to capture long-range dependencies and complex patterns within the DNA sequence.
*   **Decoder**: A series of 1D transposed convolutional layers that upsample the latent representation back to the original sequence length. The decoder also incorporates self-attention blocks to refine the reconstruction.

## Results

The model was trained on a subset of the prokaryotic DNA corpus and evaluated on both seen (train) and unseen (test) genomes. The two-stage compression process achieves perfect, lossless reconstruction (100% accuracy) for all files.

The overall compression ratio (total compressed size / original size) is detailed below.

### Train Set Results
| file | Original Size (bytes) | Compressed Size (bytes) | Compression Ratio |
|:-----|----------------------:|------------------------:|------------------:|
| AeCa |               1591049 |                  735252 |          0.462118 |
| EsCo |               4641652 |                 2000302 |          0.430946 |
| HaHi |               3890005 |                 1754356 |          0.450991 |
| WaMe |               9144432 |                 4158044 |          0.454708 |

### Test Set Results
| file | Original Size (bytes) | Compressed Size (bytes) | Compression Ratio |
|:-----|----------------------:|------------------------:|------------------:|
| AgPh |                 43970 |                   20727 |          0.471390 |
| BuEb |                 18940 |                    9233 |          0.487487 |
| HePy |               1667825 |                  753662 |          0.451883 |
| YeMi |                 73689 |                   34620 |          0.469812 |

## Usage

The entire workflow, from data preprocessing and model training to compression and evaluation, is contained within the `SeqCoder_Inter.ipynb` Jupyter Notebook.

### Prerequisites

*   Python 3.x
*   PyTorch
*   NumPy
*   Pandas
*   Blosc
*   Zstandard
*   Matplotlib / Seaborn (for visualization)

You can install the required packages using pip:
```bash
pip install torch numpy pandas blosc zstandard matplotlib seaborn
```

### Running the Notebook

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/EmberTSeal/SeqCoder.git
    cd SeqCoder
    ```
2.  **Dataset:** The notebook is configured to use the [DNACorpus](?) dataset. You will need to download the prokaryotic genome files and place them in a directory structure that matches the paths in the notebook (e.g., `/kaggle/input/dnacorpus/DNACorpus/Prokaryotic`). You may need to adjust the file paths in the notebook for local execution.
3.  **Execute the notebook:** Open and run the cells in `SeqCoder_Inter.ipynb` using Jupyter Lab or Jupyter Notebook. The notebook will:
    *   Train the autoencoder model.
    *   Compress each file in the dataset.
    *   Generate evaluation results, including compression ratios and accuracy checks.
    *   Perform a final verification to confirm lossless reconstruction.

