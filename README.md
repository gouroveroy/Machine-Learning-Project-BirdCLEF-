# Species Identification in Noisy Soundscapes - BirdCLEF 2025

## Project Overview
This notebook implements an **improved deep learning model** for identifying bird species from audio recordings in noisy soundscapes. The project uses state-of-the-art audio signal processing and data augmentation techniques combined with computer vision models to classify bird calls from noisy environmental soundscapes.

## 🎯 Project Status: **75% Complete**

### Key Achievements:
- ✅ Data loading and visualization
- ✅ Spectrogram pre-generation pipeline
- ✅ **Improved training** with Mixup, SpecAugment, and weighted sampling
- ✅ **Upgraded to EfficientNet-B3** (more accurate than B0)
- ✅ Test-Time Augmentation (TTA) for better predictions
- ✅ Complete inference pipeline
- ✅ **Baseline training cells REMOVED** (were fundamentally broken)

### Expected Performance:
- **Improved (Current):** 35-45% accuracy ✅ (after 10 epochs with B3)
- **Advanced (Next):** 60-70% accuracy (with ensemble + TTA + longer training)

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
   - `EPOCHS = 10`: Number of complete passes through the dataset
   - `MODEL_NAME = 'efficientnet_b3'`: **Upgraded model** for better accuracy
   - `NUM_CLASSES = 206`: Total bird species in the competition
   - `LR = 1e-4`: Lower learning rate for stable fine-tuning
   - `MIXUP_ALPHA = 0.2`: Mixup augmentation strength
   - `LABEL_SMOOTHING = 0.1`: Prevents overconfidence
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

### **Cell 4: Generate and Save All Spectrograms**

**What it does:**

1. **Create Output Directory:**
   - Sets up `/kaggle/working/spectrograms` directory
   - Uses `Path().mkdir()` to create the directory structure
   - `parents=True, exist_ok=True` ensures directories are created without errors

2. **Process All Audio Files:**
   - Iterates through every audio file in the dataset (all rows in `train.csv`)
   - Shows a progress bar using `tqdm` to track processing status
   - Handles thousands of files efficiently in a batch process

3. **Organize by Species:**
   - Creates a subdirectory for each species (e.g., `spectrograms/species1/`)
   - Groups spectrograms by their primary_label for easy organization
   - Makes it simple to count and review samples per species

4. **Audio Processing (Same Pipeline as Training):**
   - **Load Audio:** Reads the audio file using torchaudio
   - **Resample:** Converts to 32kHz if needed for consistency
   - **Crop/Pad:** Standardizes all audio to exactly 5 seconds
     - Takes first 5 seconds if longer (for consistency in pre-processing)
     - Pads with zeros if shorter
   - **Generate Mel-Spectrogram:** Converts audio to 2D time-frequency representation
   - **Convert to dB Scale:** Makes quiet sounds more visible

5. **Save as Images:**
   - Converts mel-spectrogram to numpy array
   - Normalizes values to 0-255 range (standard image format)
   - Saves as PNG files using OpenCV
   - Filename matches original audio file (e.g., `audio123.ogg` → `audio123.png`)

6. **Error Handling:**
   - Wraps processing in try-except block
   - Continues processing if one file fails
   - Prints error messages for debugging

7. **Summary Output:**
   - Prints total number of spectrograms generated
   - Shows output directory location
   - Displays count of species folders created

**Why it's important:**
Pre-generating spectrograms offers several advantages:
- **Faster Training:** No need to process audio on-the-fly during training
- **Consistency:** All spectrograms processed the same way (deterministic cropping)
- **Reusability:** Can use spectrograms for multiple experiments without reprocessing
- **Disk Space vs Speed Trade-off:** Uses more disk space but dramatically speeds up training
- **Debugging:** Can visually inspect spectrograms to verify quality
- **Data Analysis:** Easy to browse and analyze spectrogram patterns per species

**Output Directory Structure:**
```
spectrograms/
├── species1/
│   ├── audio1.png
│   ├── audio2.png
│   └── ...
├── species2/
│   ├── audio3.png
│   └── ...
└── species206/
    └── ...
```

---

### **Cell 5: Create Custom Dataset and Audio Processing Pipeline**

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

**Note:** If you've already run Cell 4 to pre-generate spectrograms, you could modify this dataset class to load from saved PNG files instead of processing audio on-the-fly for faster training.

---

### **Cell 6: Helper Functions for Improved Training**

**What it does:**

1. **Mixup Data Augmentation:**
   - `mixup_data()`: Blends two random training samples together
   - Creates virtual training examples by interpolating images and labels
   - Formula: `mixed_image = λ * image1 + (1-λ) * image2`
   - Helps model learn smoother decision boundaries
   - Reduces overfitting significantly

2. **SpecAugment Class:**
   - Applies time and frequency masking to spectrograms
   - Randomly masks 20% of time steps and 10% of frequency bands
   - Forces model to not rely on specific time/frequency regions
   - Proven technique from speech recognition research

3. **Weighted Sampler:**
   - `create_weighted_sampler()`: Balances class distribution in training
   - Rare species get sampled more frequently
   - Common species get sampled less frequently
   - Prevents model from only learning common species

4. **ImprovedBirdDataset:**
   - Enhanced dataset class with SpecAugment applied during training
   - Only applies augmentation to training set, not validation
   - Loads from pre-saved PNG spectrograms (10-50x faster)

**Why it's important:**
These techniques are used by competition winners and research papers. They address the fundamental problems in the baseline:
- Mixup: Reduces overfitting, improves generalization
- SpecAugment: Makes model robust to variations in audio
- Weighted Sampling: Solves class imbalance (206 species, some rare)

---

### **Cell 7: Improved Training Loop**

**What it does:**

1. **Split Data with Stratification:**
   - 80% training, 20% validation
   - Ensures each species proportionally represented

2. **Create Weighted Sampler:**
   - Balances training data so all species get equal representation
   - Critical for learning rare species

3. **Setup Improved Configuration:**
   - **Learning Rate:** 1e-4 (10x lower than baseline)
   - **Label Smoothing:** 0.1 (prevents overconfidence)
   - **Mixup Alpha:** 0.2 (optimal mixing ratio)
   - **Cosine Annealing Scheduler:** Gradually reduces learning rate

4. **Training Loop (10 Epochs):**

   **Training Phase (each epoch):**
   - Apply Mixup to each batch
   - Calculate mixed loss using both labels
   - Update model weights
   - Track training loss

   **Validation Phase (each epoch):**
   - Evaluate on clean validation data (no augmentation)
   - Calculate validation loss and accuracy
   - Save best model checkpoint

   **Learning Rate Scheduling:**
   - Cosine annealing reduces LR smoothly over training
   - Helps model converge to better minima

5. **Results Visualization:**
   - Plots training/validation loss curves
   - Plots validation accuracy over epochs
   - Compares to broken baseline (5-10% vs 30-40%)

6. **Best Model Saving:**
   - Automatically saves model when validation accuracy improves
   - File: `best_model_improved.pth`
   - Can be loaded for inference

**Expected Performance with EfficientNet-B3:**
- Epoch 1-3: 20-25% accuracy (learning basics)
- Epoch 5-7: 30-40% accuracy (learning patterns)
- Epoch 10: 35-45% accuracy (final performance)

**Why it's important:**
This is the complete fix for the broken baseline. Every technique here is battle-tested from competition winners:
- Lower LR prevents overshooting optima
- Mixup + SpecAugment provide robust augmentation
- Weighted sampling handles class imbalance
- Label smoothing prevents overconfidence
- Cosine annealing improves final convergence

---

### **Cell 8: Test-Time Augmentation (TTA)**

**What it does:**

1. **predict_with_tta() Function:**
   - Makes predictions on N augmented versions of each test sample
   - Each augmentation uses different random crop from audio
   - Averages predictions across all augmentations
   - Typically improves accuracy by 2-5%

2. **How TTA Works:**
   - Load test audio file
   - Create 5 different random crops (different 5-second segments)
   - Process each crop through model
   - Average softmax probabilities
   - Return final averaged prediction

**Why it's important:**
- Single random crop might miss the bird call
- Averaging multiple crops is more robust
- Competition winners always use TTA
- Simple technique with guaranteed improvement

---

### **Cell 9: Inference Pipeline**

**What it does:**

1. **Load Best Model:**
   - Loads `best_model_improved.pth` weights
   - Sets model to evaluation mode

2. **Process Test Files:**
   - Finds all test audio files
   - Applies TTA to each file (optional, configurable)
   - Generates species prediction + confidence

3. **Create Submission:**
   - Formats predictions as CSV for Kaggle competition
   - Columns: filename, predicted_species, confidence
   - Saves to `submission.csv`

4. **Analysis:**
   - Shows sample predictions
   - Plots confidence distribution
   - High confidence = model is certain
   - Low confidence = model is uncertain

**Why it's important:**
This is how you actually submit predictions to Kaggle. The TTA option gives you a speed vs accuracy tradeoff:
- TTA enabled: Slower but more accurate
- TTA disabled: Faster but slightly less accurate

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
- ✅ All spectrograms generated and saved to disk (organized by species)
- ✅ Sample mel-spectrogram displayed (verifies audio processing works)
- ✅ **EfficientNet-B3 model trained for 10 epochs** (35-45% accuracy expected)
- ✅ Validation metrics tracked (loss + accuracy plots)
- ✅ Test-Time Augmentation function loaded
- ✅ Inference pipeline complete with submission.csv generated
- ✅ Model weights saved to `best_model_improved.pth`

---

## Next Steps for Further Improvement (To reach 90-100%)

### Completed (75%):
- ✅ Mixup data augmentation
- ✅ SpecAugment (time/frequency masking)
- ✅ Weighted sampling for class balance
- ✅ Label smoothing
- ✅ Lower learning rate with cosine annealing
- ✅ Test-Time Augmentation (TTA)
- ✅ Complete inference pipeline

### Remaining (25%):
1. **Upgrade Model Architecture:**
   - Use EfficientNet-B3 or B4 (larger, more accurate)
   - Experiment with other architectures (ResNet, ConvNeXt)

2. **Model Ensemble:**
   - Train 3-5 different models
   - Average their predictions
   - Typically adds 3-5% accuracy

3. **Advanced Audio Processing:**
   - PCEN (Per-Channel Energy Normalization) instead of log mel
   - Multi-scale spectrograms (different window sizes)
   - Bird-specific frequency filtering

4. **Better Training:**
   - Train for 30-50 epochs
   - Cross-validation (5-fold)
   - Advanced augmentation (pitch shift, time stretch)

5. **Post-Processing:**
   - Confidence thresholding
   - Species co-occurrence filtering (some birds appear together)
   - Temporal smoothing for soundscape predictions
6. **Model Ensemble:** Combine predictions from multiple models
7. **Larger Models:** Try EfficientNet-B3 or B4 for better accuracy
8. **Test Time Augmentation:** Average predictions over multiple crops

---

## Context

In this project we need to identify bird species from short audio recordings in noisy, real-world environments. The dataset includes:
- 206 different bird species
- Thousands of audio recordings from various locations
- Background noise from rain, wind, insects, and other birds
- Varying audio quality and recording conditions

This baseline provides a solid foundation for building more sophisticated models to tackle these challenges.