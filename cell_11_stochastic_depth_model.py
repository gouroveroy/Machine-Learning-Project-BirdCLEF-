# =============================================================================
# CELL 11 — MARKDOWN (paste this as a Markdown cell above the code cell)
# =============================================================================
# # 🧠 Cell 11: Enhanced SED Model with Stochastic Depth

# ## What is Stochastic Depth?
# Stochastic Depth (DropPath) randomly drops entire residual blocks during training.
# Unlike standard dropout (which drops individual neurons), this drops **whole layers**.

# ## Why it matters:
# - Only helps during **self-training** (not supervised training alone)
# - Acts as strong regularization, forcing the model to learn robust features
# - Prevents the student model from overfitting to noisy pseudo-labels
# - **Consistently boosts validation accuracy** when used with self-training

# ## Key parameters:
# | Parameter | Value | Reason |
# |-----------|-------|--------|
# | `drop_path_rate` | 0.15 | Optimal value found through experimentation |
# | When to use | Self-training only | No improvement in supervised-only training |

# ## How it works:
# ```
# Standard forward:   Input → Block1 → Block2 → Block3 → Output
# With DropPath:       Input → Block1 → [SKIP] → Block3 → Output  (random)
# ```
# The model must learn to produce good predictions even when some
# intermediate representations are missing — this forces robustness.

# =============================================================================
# CELL 11 — CODE (paste this as a Code cell)
# =============================================================================

# ============================================================================
# ENHANCED SED MODEL WITH STOCHASTIC DEPTH
# ============================================================================

class BirdClassifierSED_V2(nn.Module):
    """
    Enhanced SED-based Bird Species Classifier with Stochastic Depth support.

    Improvements over V1:
    - Supports drop_path_rate for Noisy Student self-training
    - Stochastic Depth randomly drops entire residual blocks during training
    - This prevents overfitting to noisy pseudo-labels during self-training
    - drop_path_rate=0.15 was found to be optimal across different backbones

    Usage:
    - Supervised training:    BirdClassifierSED_V2(..., drop_path_rate=0.0)
    - Self-training:         BirdClassifierSED_V2(..., drop_path_rate=0.15)
    """

    def __init__(self, num_classes, backbone_name=None, dropout=0.5, drop_path_rate=0.0):
        super().__init__()
        if backbone_name is None:
            backbone_name = Config.MODEL_NAME

        # Backbone with feature extraction only + optional stochastic depth
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            features_only=True,
            drop_path_rate=drop_path_rate,  # Key addition for self-training
        )
        backbone_dim = self.backbone.feature_info.channels()[-1]

        # SED Head (same as V1)
        self.head = AttHead(in_chans=backbone_dim, p=dropout, num_class=num_classes)

        self.backbone_name = backbone_name
        self.drop_path_rate = drop_path_rate

        print(f"  Backbone: {backbone_name} (features_only, dim={backbone_dim})")
        print(f"  Head: SED AttHead (hidden=512, dropout={dropout})")
        print(f"  Stochastic Depth: drop_path_rate={drop_path_rate}")

    def forward(self, x):
        """Training forward: returns clip-level logits for CrossEntropy loss."""
        features = self.backbone(x)[-1]
        head_out = self.head(features)
        framewise_logit = head_out["framewise_logit"]
        clipwise_logit = framewise_logit.max(dim=-1)[0]
        return clipwise_logit

    def forward_framewise(self, x):
        """Inference forward: returns framewise probabilities for overlap-averaging."""
        features = self.backbone(x)[-1]
        head_out = self.head(features)
        framewise_prob = head_out["framewise_logit"].sigmoid()
        return framewise_prob


def create_sed_model(backbone_name=None, drop_path_rate=0.0, pretrained_path=None):
    """
    Factory function to create SED models with configurable stochastic depth.

    Args:
        backbone_name: timm model name (default: Config.MODEL_NAME)
        drop_path_rate: 0.0 for supervised, 0.15 for self-training
        pretrained_path: path to load pretrained weights

    Returns:
        model on DEVICE
    """
    model = BirdClassifierSED_V2(
        num_classes=len(ALL_SPECIES),
        backbone_name=backbone_name or Config.MODEL_NAME,
        dropout=Config.SED_DROPOUT,
        drop_path_rate=drop_path_rate,
    ).to(DEVICE)

    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path, map_location=DEVICE)
        try:
            model.load_state_dict(state_dict, strict=True)
            print(f"  ✅ Loaded standard weights from: {pretrained_path}")
        except RuntimeError as e:
            if "Unexpected key(s)" in str(e) and "Missing key(s)" not in str(e):
                # This perfectly handles Nikitababich's 1st-place models which simply contain unused extra keys
                model.load_state_dict(state_dict, strict=False)
                print(f"  ✅ Loaded 1st-place extended weights from: {pretrained_path}")
            else:
                print(f"  ⚠️ Warning loading weights: {e}")
                model.load_state_dict(state_dict, strict=False)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,} | Trainable: {trainable_params:,}")

    return model


# --- Demo: Create model with stochastic depth ---
print("=" * 70)
print("🧠 Enhanced SED Model with Stochastic Depth")
print("=" * 70)

print("\n📌 Supervised training model (drop_path_rate=0.0):")
demo_model_supervised = create_sed_model(drop_path_rate=0.0)

print("\n📌 Self-training model (drop_path_rate=0.15):")
demo_model_selftrain = create_sed_model(drop_path_rate=0.15)

# Clean up demo models
del demo_model_supervised, demo_model_selftrain
torch.cuda.empty_cache() if torch.cuda.is_available() else None

print("\n✅ Stochastic Depth model ready!")
print("   Use drop_path_rate=0.0 for supervised training")
print("   Use drop_path_rate=0.15 for self-training (Noisy Student)")
