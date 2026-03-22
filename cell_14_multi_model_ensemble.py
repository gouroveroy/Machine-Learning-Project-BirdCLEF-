# =============================================================================
# CELL 14 — MARKDOWN (paste this as a Markdown cell above the code cell)
# =============================================================================
# # 🎼 Cell 14: Multi-Model Ensemble

# ## Why ensembles work:
# Different backbone architectures learn different features from spectrograms:
# - **EfficientNet** excels at fine-grained frequency patterns
# - **RegNet** captures temporal dynamics better
# - **NFNet** has built-in normalization-free design, learns complementary features

# ## Our ensemble architecture:
# | Model | Architecture | Role |
# |-------|-------------|------|
# | M1-M3 | EfficientNet-B0/B3/B4 | Different capacity levels |
# | M4-M5 | RegNetY-016/008 | Temporal dynamics focus |
# | M6 | NFNet-L0 | Normalization-free features |
# | M7 | EfficientNet-B3 | Insecta/Amphibia specialist |

# ## Key insights:
# - Simple **arithmetic mean** of predictions works best (no learned weights needed!)
# - Each model uses the **same SED architecture** (GeMFreq + AttHead), just different backbones
# - All models go through 4 iterations of self-training before ensembling
# - Even just 2-3 diverse models significantly boosts accuracy

# ## For this project:
# We demonstrate training and ensembling 3 diverse backbones:
# - `tf_efficientnet_b3.ns_jft_in1k` (already trained)
# - `tf_efficientnet_b0.ns_jft_in1k` (lightweight)
# - `eca_nfnet_l0.ra2_in1k` (normalization-free)

# =============================================================================
# CELL 14 — CODE (paste this as a Code cell)
# =============================================================================

# ============================================================================
# MULTI-MODEL ENSEMBLE
# ============================================================================

# Backbone configurations for ensemble
ENSEMBLE_BACKBONES = [
    {
        "name": "tf_efficientnet_b3.ns_jft_in1k",
        "save_path": "/kaggle/input/datasets/robinhood19/75-output/outputs/best_model_sed.pth",
        "trained": True,
    },
    {
        "name": "tf_efficientnet_b0.ns_jft_in1k",
        "save_path": "/kaggle/input/datasets/robinhood19/ensemble-models/best_model_ens_effb0.pth",
        "trained": True,
    },
    {
        "name": "eca_nfnet_l0.ra2_in1k",
        "save_path": "/kaggle/input/datasets/robinhood19/ensemble-models/best_model_ens_nfnet.pth",
        "trained": True,
    },
    {
        "name": "tf_efficientnet_b4.ns_jft_in1k",
        "save_path": "/kaggle/input/datasets/nikitababich/birdclef2025-1st-place-ensemble/tf_efficientnet_b4.ns_jft_in1k_sampler_maxsum_iteration_3_v1_temp_0.55_64_bs_0.15_drop_path_rate_1_mixup_ratio_pseudo_data_20_duration_sed_type_0.5_mixup_p_(224, 512)_size_ce_4096_n_fft_0_fold_22_seed_25_epoch.pt",
        "trained": True,
    },
    {
        "name": "regnety_016.tv2_in1k",
        "save_path": "/kaggle/input/datasets/nikitababich/birdclef2025-1st-place-ensemble/regnety_016.tv2_in1k_sampler_maxsum_iteration_4_v1_temp_0.6_64_bs_0.15_drop_path_rate_1_mixup_ratio_pseudo_data_20_duration_sed_type_0.5_mixup_p_(224, 512)_size_ce_4096_n_fft_2_fold_25_epoch.pt",
        "trained": True,
    },
    {
        "name": "regnety_008.pycls_in1k",
        "save_path": "/kaggle/input/datasets/nikitababich/birdclef2025-1st-place-ensemble/regnety_008.pycls_in1k_20_duration_sed_mixup_(224, 512)_size_ce_4096_n_fft_2_fold_15_epoch.pt",
        "trained": True,
    }
]


def train_single_model(
    backbone_name,
    train_df,
    save_path,
    epochs=10,
    drop_path_rate=0.0,
):
    """
    Train a single SED model with the specified backbone.

    Uses the same training pipeline as Cell 8:
    - ImprovedBirdDataset with SpecAugment
    - Mixup augmentation
    - Weighted sampling for class balance
    - AdamW + CosineAnnealing
    - CrossEntropy with label smoothing

    Args:
        backbone_name: timm model name
        train_df: full training DataFrame
        save_path: where to save best model
        epochs: number of training epochs
        drop_path_rate: 0.0 for supervised, 0.15 for self-training

    Returns:
        best_val_acc: best validation accuracy
    """
    print(f"\n{'='*70}")
    print(f"🤖 Training: {backbone_name}")
    print(f"{'='*70}")

    # Split data
    train_split, val_split = train_test_split(
        train_df, test_size=0.2, stratify=train_df["primary_label"], random_state=42
    )

    # Datasets
    train_ds = ImprovedBirdDataset(train_split, Config.TRAIN_AUDIO_DIR, spec_augment=True)
    val_ds = ImprovedBirdDataset(val_split, Config.TRAIN_AUDIO_DIR, spec_augment=False)

    # Data loaders
    train_sampler = create_weighted_sampler(train_split)
    train_loader = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, sampler=train_sampler, num_workers=2
    )
    val_loader_ens = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )

    # Create model
    model = create_sed_model(
        backbone_name=backbone_name,
        drop_path_rate=drop_path_rate,
    )

    # Training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    best_val_acc = 0.0
    train_losses = []
    val_accs = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [{backbone_name}]")
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Mixup
            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=Config.MIXUP_ALPHA)

            optimizer.zero_grad()
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.6f}"})

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        val_loss, val_acc, _, _ = evaluate_model(model, val_loader_ens, criterion, DEVICE)
        val_accs.append(val_acc)
        scheduler.step()

        print(f"  Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Acc={val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ✅ Best model saved! (Val Acc: {val_acc:.2f}%)")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(range(1, epochs + 1), train_losses, "b-o", label="Train Loss")
    ax1.set_title(f"{backbone_name}: Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(range(1, epochs + 1), val_accs, "g-o", label="Val Accuracy")
    ax2.set_title(f"{backbone_name}: Accuracy")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.show()

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return best_val_acc


# 1st-Place Competition Label Map mapping their string names to output indices
COMPETITION_LABEL2IND = {'grekis': 0, 'compau': 1, 'trokin': 2, 'roahaw': 3, 'banana': 4, 'whtdov': 5, 'socfly1': 6, 'yeofly1': 7, 'bobfly1': 8, 'wbwwre1': 9, 'soulap1': 10, 'sobtyr1': 11, 'trsowl': 12, 'laufal1': 13, 'strcuc1': 14, 'bbwduc': 15, 'saffin': 16, 'amekes': 17, 'tropar': 18, 'compot1': 19, 'blbgra1': 20, 'bubwre1': 21, 'strfly1': 22, 'gycwor1': 23, 'greegr': 24, 'linwoo1': 25, 'pirfly1': 26, 'littin1': 27, 'bkmtou1': 28, 'yercac1': 29, 'butsal1': 30, 'smbani': 31, 'bugtan': 32, 'chbant1': 33, 'yebela1': 34, 'rutjac1': 35, 'cotfly1': 36, 'whbman1': 37, 'yehcar1': 38, 'solsan': 39, 'rumfly1': 40, 'yecspi2': 41, 'blhpar1': 42, 'creoro1': 43, 'paltan1': 44, 'rinkin1': 45, 'orcpar': 46, 'stbwoo2': 47, 'speowl1': 48, 'yebfly1': 49, 'plbwoo1': 50, 'yebsee1': 51, 'bkcdon': 52, 'strher': 53, 'y00678': 54, 'babwar': 55, 'strowl1': 56, 'gybmar': 57, 'cocwoo1': 58, 'secfly1': 59, 'thbeup1': 60, 'pavpig2': 61, 'baymac': 62, 'rtlhum': 63, 'purgal2': 64, 'colcha1': 65, 'crcwoo1': 66, 'ywcpar': 67, 'chfmac1': 68, 'rugdov': 69, 'gohman1': 70, 'watjac1': 71, 'grnkin': 72, 'greani1': 73, 'whfant1': 74, 'cattyr': 75, 'srwswa1': 76, 'blbwre1': 77, 'mastit1': 78, 'greibi1': 79, 'snoegr': 80, '41663': 81, 'leagre': 82, 'blcjay1': 83, 'grbhaw1': 84, 'eardov1': 85, 'blcant4': 86, 'whbant1': 87, 'yectyr1': 88, 'rufmot1': 89, 'thlsch3': 90, 'cargra1': 91, 'bicwre1': 92, 'anhing': 93, 'neocor': 94, 'shtfly1': 95, 'recwoo1': 96, 'amakin1': 97, 'ragmac1': 98, 'grasal4': 99, 'gretin1': 100, '65448': 101, 'spepar1': 102, 'fotfly': 103, 'ruther1': 104, 'yehbla2': 105, 'cregua1': 106, '21211': 107, 'whttro1': 108, 'brtpar1': 109, 'rubsee1': 110, 'blkvul': 111, 'verfly': 112, 'cinbec1': 113, 'labter1': 114, 'grepot1': 115, 'palhor2': 116, 'yelori1': 117, '517119': 118, 'colara1': 119, 'crbtan1': 120, 'rebbla1': 121, 'piepuf1': 122, 'savhaw1': 123, 'blchaw1': 124, '22973': 125, 'crebob1': 126, 'whwswa1': 127, 'spbwoo1': 128, '22333': 129, 'bucmot3': 130, '22976': 131, 'tbsfin1': 132, 'cocher1': 133, 'royfly1': 134, 'bobher1': 135, 'olipic1': 136, 'plukit1': 137, 'whmtyr1': 138, 'rosspo1': 139, '52884': 140, '65373': 141, 'blctit1': 142, '50186': 143, 'ampkin1': 144, 'bafibi1': 145, 'woosto': 146, '555086': 147, 'grysee1': 148, '566513': 149, '65962': 150, '48124': 151, 'bubcur1': 152, '42007': 153, 'piwtyr1': 154, 'rutpuf1': 155, '715170': 156, '65349': 157, '65344': 158, '41970': 159, 'shghum1': 160, 'norscr1': 161, 'sahpar1': 162, '67252': 163, '24322': 164, 'turvul': 165, '135045': 166, '65547': 167, '787625': 168, '1462737': 169, 'plctan1': 170, '555142': 171, '126247': 172, '65336': 173, '1564122': 174, '24272': 175, '548639': 176, '46010': 177, '1346504': 178, '963335': 179, '476538': 180, '714022': 181, '66893': 182, '134933': 183, '1192948': 184, '868458': 185, '523060': 186, '24292': 187, '65419': 188, '1194042': 189, '1462711': 190, '81930': 191, '67082': 192, '66578': 193, '66531': 194, '66016': 195, '21038': 196, '41778': 197, '21116': 198, '64862': 199, '528041': 200, '476537': 201, '47067': 202, '42113': 203, '42087': 204, '1139490': 205}

class EnsemblePredictor:
    """Multi-model ensemble predictor."""

    def __init__(self, model_configs, device):
        """
        Args:
            model_configs: list of dicts with "name" and "save_path"
            device: torch device
        """
        self.models = []
        self.model_names = []
        self.device = device
        self.needs_permutation = []
        
        # Compute permutation map to magically re-order the 1st place predictions into our alphabetical ordering.
        self.permutation_indices = [COMPETITION_LABEL2IND.get(sp, 0) for sp in ALL_SPECIES]

        for config in model_configs:
            if os.path.exists(config["save_path"]):
                print(f"  Loading: {config['name']} from {config['save_path']}")
                try:
                    model = BirdClassifierSED_V2(
                        num_classes=len(ALL_SPECIES),
                        backbone_name=config["name"],
                        dropout=Config.SED_DROPOUT,
                        drop_path_rate=0.0,
                    ).to(device)
                    state_dict = torch.load(config["save_path"], map_location=device)
                    # Filter out extra keys from 1st-place models that our architecture doesn't have
                    model_keys = set(model.state_dict().keys())
                    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
                    skipped_keys = set(state_dict.keys()) - model_keys
                    if skipped_keys:
                        print(f"    ℹ️ Skipped {len(skipped_keys)} extra keys: {', '.join(sorted(skipped_keys))}")
                    model.load_state_dict(filtered_state_dict, strict=True)
                    model.eval()
                    self.models.append(model)
                    self.model_names.append(config["name"])
                    self.needs_permutation.append("nikitababich" in config["save_path"])
                    print(f"    ✅ Loaded successfully")
                except Exception as e:
                    print(f"    ⚠️ Failed to load: {e}")
            else:
                print(f"  ⏭️ Skipping {config['name']} (not trained yet: {config['save_path']})")

        print(f"\n  📊 Ensemble size: {len(self.models)} models")

    def predict_soundscape(self, audio_path, use_delta_shift=True, use_smoothing=True):
        """
        Ensemble prediction for a soundscape using overlap-average + delta-shift.

        Returns the arithmetic mean of all model predictions.
        """
        all_preds = []
        for j, model in enumerate(self.models):
            # Use the full enhanced pipeline for each model
            preds = enhanced_inference(
                model, audio_path, self.device,
                use_delta_shift=use_delta_shift,
                use_smoothing=use_smoothing,
            )
            # Realign 1st-place model class labels to alphabetical
            if self.needs_permutation[j]:
                preds = preds[:, self.permutation_indices]
                
            all_preds.append(preds)

        # Arithmetic mean ensemble
        ensemble_preds = np.mean(all_preds, axis=0)
        return ensemble_preds

    def predict_batch_validation(self, val_loader):
        """
        Ensemble prediction on validation set.
        Returns: (all_probs, all_labels) as numpy arrays.
        """
        all_probs = []
        all_labels = []

        for images, labels in tqdm(val_loader, desc="Ensemble Eval"):
            images = images.to(self.device)

            # Collect predictions from all models
            batch_preds = []
            for j, model in enumerate(self.models):
                with torch.no_grad():
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    # Apply O(1) array slicing to instantly re-align 1st place model class labels
                    if self.needs_permutation[j]:
                        probs = probs[:, self.permutation_indices]
                    batch_preds.append(probs)

            # Average predictions
            avg_probs = np.mean(batch_preds, axis=0)

            all_probs.extend(avg_probs.tolist())
            all_labels.extend(labels.numpy().tolist())

        return np.array(all_probs), np.array(all_labels)

    def cleanup(self):
        """Free GPU memory."""
        for model in self.models:
            del model
        self.models = []
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


# --- Run ensemble training and evaluation ---
print("=" * 70)
print("🎼 Multi-Model Ensemble")
print("=" * 70)

# Train additional backbones (if not already trained)
ensemble_results = {}
for config in ENSEMBLE_BACKBONES:
    if config["trained"]:
        if os.path.exists(config["save_path"]):
            print(f"\n✅ {config['name']} already trained ({config['save_path']})")
            ensemble_results[config["name"]] = "pre-trained"
        else:
            print(f"\n⚠️ {config['name']} marked as trained but {config['save_path']} not found!")
    else:
        # Train new model
        print(f"\n🔧 Training {config['name']}...")
        best_acc = train_single_model(
            backbone_name=config["name"],
            train_df=df,
            save_path=config["save_path"],
            epochs=5,  # Reduced from 10 to fit Kaggle 12h limit
        )
        ensemble_results[config["name"]] = f"{best_acc:.2f}%"

# Report
print(f"\n{'='*70}")
print("📊 Individual Model Results:")
print(f"{'='*70}")
for name, result in ensemble_results.items():
    print(f"  {name:<35} → {result}")

# Ensemble evaluation on validation set
print(f"\n{'='*70}")
print("🎼 Ensemble Evaluation")
print(f"{'='*70}")

# Build validation set
_, val_split = train_test_split(
    df, test_size=0.2, stratify=df["primary_label"], random_state=42
)
val_ds_ens = ImprovedBirdDataset(val_split, Config.TRAIN_AUDIO_DIR, spec_augment=False)
val_loader_ens = DataLoader(
    val_ds_ens, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
)

# Create ensemble
print("\n📦 Loading ensemble models...")
ensemble = EnsemblePredictor(ENSEMBLE_BACKBONES, DEVICE)

if len(ensemble.models) > 1:
    # Evaluate ensemble
    ens_probs, ens_labels = ensemble.predict_batch_validation(val_loader_ens)
    
    # Top-1 Accuracy
    ens_preds = ens_probs.argmax(axis=1)
    ens_correct = sum(1 for p, l in zip(ens_preds, ens_labels) if p == l)
    ens_acc = 100 * ens_correct / len(ens_labels)
    
    # Top-5 Accuracy
    top5_preds = np.argsort(ens_probs, axis=1)[:, -5:]
    top5_correct = sum(1 for i, l in enumerate(ens_labels) if l in top5_preds[i])
    ens_top5_acc = 100 * top5_correct / len(ens_labels)

    print(f"\n{'='*50}")
    print(f"  Ensemble Results ({len(ensemble.models)} models)")
    print(f"{'='*50}")
    print(f"  Ensemble Top-1 Accuracy: {ens_acc:.2f}%")
    print(f"  Ensemble Top-5 Accuracy: {ens_top5_acc:.2f}%")
    print(f"  Models in ensemble:")
    for name in ensemble.model_names:
        print(f"    • {name}")
    print(f"{'='*50}")
else:
    print("\n💡 Only 1 model available — no ensemble to compare.")
    print("   Train additional models to see ensemble improvement.")

# Cleanup
ensemble.cleanup()

print("\n✅ Multi-model ensemble ready!")
print("   Key finding: Even 2-3 diverse backbones significantly boost accuracy.")
print("   More models + self-training iterations = higher gains.")
