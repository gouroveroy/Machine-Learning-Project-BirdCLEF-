# =============================================================================
# CELL 12 — MARKDOWN (paste this as a Markdown cell above the code cell)
# =============================================================================
# # 🔄 Cell 12: Delta-Shift Test-Time Augmentation (TTA)

# ## What is Delta-Shift TTA?
# A temporal augmentation technique applied **during inference only**.
# The prediction is computed at the original position AND at ±2 frame offsets,
# then the three predictions are blended together.

# ## How it works:
# ```
# Original prediction:    [..., p(t-2), p(t-1), p(t), p(t+1), p(t+2), ...]
# Left-shifted (-2):      [..., p(t), p(t+1), p(t+2), p(t+3), p(t+4), ...]
# Right-shifted (+2):     [..., p(t-4), p(t-3), p(t-2), p(t-1), p(t), ...]

# Final = 0.25 * left_shifted + 0.50 * original + 0.25 * right_shifted
# ```

# ## Why this works:
# - Each time frame gets predictions from **3 different temporal contexts**
# - Reduces sensitivity to exact segment boundaries
# - Acts as a form of **free TTA** with no additional model cost
# - Combined with overlap-averaging → very robust predictions

# ## Pipeline: Audio → Overlap-Average → Delta-Shift → Gaussian Smoothing

# =============================================================================
# CELL 12 — CODE (paste this as a Code cell)
# =============================================================================

# ============================================================================
# DELTA-SHIFT TEST-TIME AUGMENTATION
# ============================================================================


def delta_shift_tta(segment_preds, delta=2, weights=None):
    """
    Delta-Shift TTA: shift predictions by ±delta frames and blend.

    This technique computes predictions at
    the original position AND at ±delta offsets, then blends them.

    Args:
        segment_preds: [num_segments, num_classes] predictions
        delta: number of frames to shift (default: 2)
        weights: blending weights [left, center, right] (default: [0.25, 0.5, 0.25])

    Returns:
        blended_preds: [num_segments, num_classes] TTA-enhanced predictions
    """
    if weights is None:
        weights = [0.25, 0.5, 0.25]

    num_segments, num_classes = segment_preds.shape

    # Center prediction (original)
    center = segment_preds.copy()

    # Left-shifted: shift predictions left by delta positions
    left_shifted = np.zeros_like(segment_preds)
    if delta < num_segments:
        left_shifted[:-delta] = segment_preds[delta:]
        left_shifted[-delta:] = segment_preds[-delta:]  # Replicate edge
    else:
        left_shifted = segment_preds.copy()

    # Right-shifted: shift predictions right by delta positions
    right_shifted = np.zeros_like(segment_preds)
    if delta < num_segments:
        right_shifted[delta:] = segment_preds[:-delta]
        right_shifted[:delta] = segment_preds[:delta]  # Replicate edge
    else:
        right_shifted = segment_preds.copy()

    # Blend
    blended = (
        weights[0] * left_shifted + weights[1] * center + weights[2] * right_shifted
    )

    return blended


def enhanced_inference(model, audio_path, device, use_delta_shift=True, use_smoothing=True):
    """
    Full inference pipeline combining all techniques:
    1. Overlap-average framewise predictions
    2. Delta-shift TTA
    3. Gaussian smoothing

    Args:
        model: BirdClassifierSED model with forward_framewise method
        audio_path: path to soundscape audio
        device: torch device
        use_delta_shift: whether to apply delta-shift TTA
        use_smoothing: whether to apply Gaussian smoothing

    Returns:
        segment_preds: [NUM_SEGMENTS, num_classes]
    """
    # Step 1: Overlap-average inference (no smoothing yet — we apply it last)
    segment_preds = overlap_average_inference(
        model, audio_path, device, apply_smoothing=False
    )

    # Step 2: Delta-shift TTA
    if use_delta_shift:
        segment_preds = delta_shift_tta(segment_preds, delta=2)

    # Step 3: Gaussian smoothing (applied last)
    if use_smoothing:
        segment_preds = gauss_smooth(segment_preds)

    return segment_preds


# --- Demo ---
print("=" * 70)
print("🔄 Delta-Shift TTA + Enhanced Inference Pipeline")
print("=" * 70)

# Demo with synthetic data
print("\n📌 Demo with synthetic predictions:")
np.random.seed(42)
demo_preds = np.random.rand(12, 206).astype(np.float32)  # 12 segments, 206 species
demo_preds = demo_preds / demo_preds.sum(axis=1, keepdims=True)  # Normalize

# Apply delta-shift
demo_shifted = delta_shift_tta(demo_preds, delta=2)

print(f"  Input shape:  {demo_preds.shape}")
print(f"  Output shape: {demo_shifted.shape}")
print(f"  Max change:   {np.abs(demo_preds - demo_shifted).max():.4f}")
print(f"  Mean change:  {np.abs(demo_preds - demo_shifted).mean():.6f}")

# Demo on real soundscapes if available
test_soundscape_dir = Path(Config.DATA_DIR) / "test_soundscapes"
train_soundscape_dir = Path(Config.DATA_DIR) / "train_soundscapes"

soundscape_dir = None
if test_soundscape_dir.exists() and len(list(test_soundscape_dir.glob("*.ogg"))) > 0:
    soundscape_dir = test_soundscape_dir
elif train_soundscape_dir.exists() and len(list(train_soundscape_dir.glob("*.ogg"))) > 0:
    soundscape_dir = train_soundscape_dir

if soundscape_dir is not None:
    print(f"\n🔊 Enhanced Inference Demo (with Delta-Shift TTA)")
    print("-" * 70)

    # Load the trained model
    eval_model = create_sed_model(pretrained_path="best_model_sed.pth")
    eval_model.eval()

    soundscape_files = sorted(list(soundscape_dir.glob("*.ogg")))[:2]
    for audio_path in soundscape_files:
        print(f"\n📂 {audio_path.name}")

        # Compare: without vs with delta-shift
        preds_basic = overlap_average_inference(eval_model, str(audio_path), DEVICE, apply_smoothing=True)
        preds_enhanced = enhanced_inference(eval_model, str(audio_path), DEVICE, use_delta_shift=True, use_smoothing=True)

        print(f"  {'Seg':>5} | {'Basic Top Species':<20} {'Conf':>6} | {'Enhanced Top Species':<20} {'Conf':>6}")
        print(f"  {'-'*5}-+-{'-'*20}-{'-'*6}-+-{'-'*20}-{'-'*6}")
        for seg_idx in range(min(preds_basic.shape[0], 6)):
            # Basic
            b_idx = preds_basic[seg_idx].argmax()
            b_sp = IDX_TO_SPECIES[b_idx]
            b_conf = preds_basic[seg_idx][b_idx]
            # Enhanced
            e_idx = preds_enhanced[seg_idx].argmax()
            e_sp = IDX_TO_SPECIES[e_idx]
            e_conf = preds_enhanced[seg_idx][e_idx]
            t = (seg_idx + 1) * 5
            print(f"  {t:>3}s  | {b_sp:<20} {b_conf:.4f} | {e_sp:<20} {e_conf:.4f}")

    del eval_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
else:
    print("\n💡 No soundscape files found for enhanced inference demo.")
    print("   The delta-shift TTA will be used during full inference on soundscapes.")

print("\n✅ Delta-Shift TTA ready!")
print("   Pipeline: Overlap-Average → Delta-Shift (±2 frames) → Gaussian Smoothing")
