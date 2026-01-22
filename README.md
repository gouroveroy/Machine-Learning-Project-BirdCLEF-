# BirdCLEF 2025: Species Identification in Noisy Soundscapes

## Project Overview
This notebook implements a baseline deep learning model for identifying bird species from audio recordings in the BirdCLEF 2025 competition. The project uses audio signal processing techniques combined with computer vision models to classify bird calls from noisy environmental soundscapes.

---

## Notebook Workflow - Step by Step

### **Cell 1: Package Installation**

**What it does:**
- Installs the `timm` (PyTorch Image Models) library which provides pre-trained EfficientNet models
- This cell uses the quiet mode (`-q`) to minimize installation output
- Verifies the Python executable path to confirm the environment being used
- Prints confirmation message once all packages are successfully installed

**Why it's important:**
The `timm` library is essential for loading pre-trained EfficientNet models that will be used as the backbone for bird species classification. Installing it at the start ensures all dependencies are available before running subsequent cells.

---

### **Cell 2: Import Libraries and Configuration**

**What it does:**
1. **Suppress Warnings:** Filters out pydantic-related warnings that don't affect functionality
2. **Import Essential Libraries:**
   - `os`, `pandas`, `numpy`: For data manipulation and file handling
   - `matplotlib`, `seaborn`: For data visualization
   - `torch`, `torchaudio`: For deep learning and audio processing
   - `timm`: For pre-trained EfficientNet models
   - `Dataset`, `DataLoader`: For creating custom datasets and batch processing
   - `tqdm`: For progress bars during training
   - `train_test_split`: For splitting data into training and validation sets

3. **Define Configuration Class:**
   - `SR = 32000`: Sample rate of 32kHz (standard for bird audio)
   - `N_MELS = 128`: Number of Mel frequency bands (height of spectrogram image)
   - `FMIN = 20`: Minimum frequency to capture (20 Hz)
   - `FMAX = 16000`: Maximum frequency (Nyquist frequency for 32kHz sampling)
   - `DURATION = 5`: Process 5-second audio chunks
   - `BATCH_SIZE = 32`: Number of samples per training batch
   - `EPOCHS = 3`: Number of complete passes through the dataset
   - `MODEL_NAME = 'efficientnet_b0'`: Lightweight baseline architecture
   - `NUM_CLASSES = 206`: Total bird species in the competition
   - `TRAIN_AUDIO_DIR`: Path to audio files on Kaggle
   - `TRAIN_METADATA`: Path to CSV file containing labels and metadata

4. **Set Device:**
   - Automatically detects if CUDA GPU is available
   - Uses GPU if available, otherwise falls back to CPU
   - Prints which device will be used for training

**Why it's important:**
This cell establishes the entire foundation of the project. The configuration parameters are carefully chosen based on audio processing best practices and the competition requirements. Using GPU acceleration significantly speeds up training time.

---

### **Cell 3: Load and Visualize Dataset Distribution**

**What it does:**
1. **Load Metadata:**
   - Reads the `train.csv` file containing information about each audio recording
   - Each row includes filename, species label, recording location, and other metadata

2. **Count Species Distribution:**
   - Counts how many audio files exist for each bird species
   - Sorts species by frequency (most common to rarest)
   - This reveals the class imbalance problem in the dataset

3. **Visualize Top 30 Species:**
   - Creates a bar plot showing the 30 most common species
   - Uses color-coded bars (viridis palette) for better visualization
   - Rotates x-axis labels 90 degrees for readability
   - Shows total number of species in the title

4. **Display Rarest Species:**
   - Prints the 5 species with the fewest audio recordings
   - Helps identify potential challenges with rare species classification

**Why it's important:**
Understanding the dataset distribution is crucial because:
- Some species have thousands of recordings while others have very few
- This "long tail" distribution means the model may learn to predict common species well but struggle with rare ones
- It informs decisions about data augmentation, sampling strategies, and evaluation metrics

---

### **Cell 4: Create Custom Dataset and Audio Processing Pipeline**

**What it does:**

1. **Define BirdCLEFDataset Class:**
   - Inherits from PyTorch's `Dataset` class for seamless integration with DataLoader
   - Creates a mapping from species names to integer indices for classification

2. **Audio Processing Pipeline (in `__getitem__` method):**

   **Step 1 - Load Audio:**
   - Uses `torchaudio.load()` to read the audio file
   - Returns waveform (audio signal) and sample rate

   **Step 2 - Resample if Necessary:**
   - Checks if the audio sample rate matches the target (32kHz)
   - If different, resamples the audio to 32kHz for consistency

   **Step 3 - Crop or Pad to Fixed Duration:**
   - Calculates target length (32000 samples/sec × 5 sec = 160,000 samples)
   - If audio is longer: randomly crops a 5-second segment (data augmentation)
   - If audio is shorter: pads with zeros to reach 5 seconds

   **Step 4 - Convert to Mel-Spectrogram:**
   - Transforms the 1D audio waveform into a 2D time-frequency representation
   - Uses 128 Mel frequency bands spanning 20Hz to 16kHz
   - Mel scale better represents human/bird perception of sound

   **Step 5 - Convert to Logarithmic Scale:**
   - Applies `AmplitudeToDB` transformation (converts to decibels)
   - Makes quiet sounds more visible to the model
   - Mimics logarithmic nature of biological hearing

   **Step 6 - Create 3-Channel Image:**
   - Repeats the grayscale spectrogram across 3 channels (RGB format)
   - Necessary because EfficientNet expects 3-channel inputs
   - Final shape: [3, 128, 313] where 313 is the time dimension

   **Step 7 - Return Image and Label:**
   - Returns the processed spectrogram and corresponding species label index

3. **Test the Pipeline:**
   - Creates a dataset instance using the loaded metadata
   - Loads the first audio sample and processes it
   - Visualizes the mel-spectrogram using matplotlib
   - Displays with 'inferno' colormap showing intensity in decibels
   - Verifies the entire pipeline works correctly before training

**Why it's important:**
This is the core data preprocessing pipeline. Converting audio to mel-spectrograms allows us to use powerful computer vision models (like EfficientNet) on audio data. The fixed duration ensures consistent input sizes, and the augmentation (random cropping) helps the model generalize better.

---

### **Cell 5: Model Definition and Training**

**What it does:**

1. **Split Data into Train/Validation Sets:**
   - Uses 80% of data for training, 20% for validation
   - `stratify=df['primary_label']` ensures each species is proportionally represented in both sets
   - `random_state=42` makes the split reproducible

2. **Create Data Loaders:**
   - `train_loader`: Shuffles data each epoch, uses 2 worker processes for parallel loading
   - `val_loader`: No shuffling needed for validation
   - Batches samples into groups of 32 for efficient GPU processing

3. **Define BirdClassifier Model:**
   - Uses EfficientNet-B0 as the backbone architecture
   - `pretrained=True`: Loads weights pre-trained on ImageNet
   - Transfer learning: The model already knows general image features
   - Replaces final layer to output predictions for 206 bird species
   - Moves model to GPU/CPU based on available hardware

4. **Setup Training Components:**
   - **Loss Function:** CrossEntropyLoss for multi-class classification
   - **Optimizer:** Adam with learning rate of 0.001 (good default for fine-tuning)
   - Adam adapts learning rates per parameter for better convergence

5. **Training Loop (Runs for 3 Epochs):**

   **For each epoch:**
   - Sets model to training mode (enables dropout, batch normalization updates)

   **For each batch:**
   - Moves images and labels to GPU/CPU
   - **Forward Pass:** Model predicts species for each audio spectrogram
   - **Compute Loss:** Compares predictions to true labels
   - **Backward Pass:** Calculates gradients via backpropagation
   - **Update Weights:** Optimizer adjusts model parameters
   - **Track Progress:** Updates progress bar with current loss

   **After each epoch:**
   - Prints average loss across all batches
   - Lower loss indicates better training performance

6. **Save Trained Model:**
   - Saves model weights to `baseline_model.pth`
   - Can be loaded later for inference or further training

**Why it's important:**
This cell brings everything together for actual model training. The transfer learning approach (using pretrained EfficientNet) is crucial because:
- It requires less training data to achieve good performance
- The model already understands visual patterns, edges, and textures
- We only need to teach it to recognize bird-specific patterns in spectrograms
- Training is faster and more stable compared to training from scratch

The validation split is essential for monitoring overfitting and ensuring the model generalizes to unseen data.

---

## Key Concepts Explained

### **Why Convert Audio to Images?**
- Bird calls create unique patterns in spectrograms (visual fingerprints)
- State-of-the-art computer vision models can recognize these patterns
- Mel-spectrograms preserve perceptually important frequency information
- Transfer learning from ImageNet gives us a powerful starting point

### **What is Transfer Learning?**
- Using a model pre-trained on one task (ImageNet classification) for another task (bird identification)
- The early layers learn general features (edges, textures, patterns)
- Only the final layers need to be retrained for bird species
- Dramatically reduces training time and required data

### **Why EfficientNet?**
- Excellent balance between accuracy and computational efficiency
- EfficientNet-B0 is the smallest variant, perfect for a baseline
- Can scale up to larger variants (B1-B7) for better performance
- Widely used in audio classification competitions

---

## Expected Output

After running all cells successfully:
- ✅ All required packages installed
- ✅ Dataset loaded and visualized (showing class imbalance)
- ✅ Sample mel-spectrogram displayed (verifies audio processing works)
- ✅ Model trained for 3 epochs
- ✅ Training loss decreases over time
- ✅ Model weights saved to `baseline_model.pth`

---

## Next Steps for Improvement

1. **Add Validation Metrics:** Implement accuracy, F1-score tracking during training
2. **Data Augmentation:** Add mixup, time/frequency masking for robustness
3. **Handle Class Imbalance:** Use weighted sampling or focal loss
4. **Longer Training:** Increase epochs with learning rate scheduling
5. **Model Ensemble:** Combine predictions from multiple models
6. **Larger Models:** Try EfficientNet-B3 or B4 for better accuracy
7. **Test Time Augmentation:** Average predictions over multiple crops

---

## Competition Context

BirdCLEF 2025 challenges participants to identify bird species from short audio recordings in noisy, real-world environments. The dataset includes:
- 206 different bird species
- Thousands of audio recordings from various locations
- Background noise from rain, wind, insects, and other birds
- Varying audio quality and recording conditions

This baseline provides a solid foundation for building more sophisticated models to tackle these challenges.