import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.linear_model import LogisticRegression


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

# For this model:
# channel 2 = left stimulus
# channel 3 = right stimulus
LEFT_IDX = 2
RIGHT_IDX = 3

DELTA_COL = "delta"
SUBJECT_COL = "subject_idx"
BLOCK_COL = "block_uid"
TRANSITION_COL = "transition_prob"

NON_REPETITIVE_VALUE = 0.5

# Used to sample choices from the model
SEED = 42


# ------------------------------------------------------------
# Load the model and behavioral data
# ------------------------------------------------------------

cfg_pk, model_pk, X_pk, Y_pk, mask_pk, block_ids_pk, train_idx_pk, val_idx_pk = load_run(RUN_DIR)

df_pk = load_all_subjects(
    cfg_pk["mat_file"],
    cfg_pk["placebo_col"],
    cfg_pk["n_subjects"]
)


# ------------------------------------------------------------
# Get the model's choices
# ------------------------------------------------------------

with torch.no_grad():
    output = model_pk(
        X_pk.to(device),
        mask_pk.to(device),
        sample=False
    )

logits = (
    output["choice_logit"][:, :, 0]
    .cpu()
    .numpy()
)

# The model's sigmoid output is P(left)
p_left = 1 / (1 + np.exp(-logits))

# Sample a choice from that probability
rng = np.random.RandomState(SEED)

choice_left = rng.binomial(
    1,
    p_left
)

# Convert to:
# 0 = left
# 1 = right
choice_right = 1 - choice_left


# ------------------------------------------------------------
# Get the left/right stimulus samples
# ------------------------------------------------------------

X_np = X_pk.cpu().numpy()
mask_np = mask_pk.cpu().numpy().astype(bool)

left = X_np[:, :, :, LEFT_IDX]
right = X_np[:, :, :, RIGHT_IDX]

# Right-minus-left contrast for every sample
sample_contrast = right - left


# ------------------------------------------------------------
# Remove any non-stimulus/padding sample
# ------------------------------------------------------------

delta_all = sample_contrast[mask_np]

column_std = delta_all.std(axis=0)
median_std = np.median(column_std)

keep_columns = (
    column_std >= 0.05 * median_std
)

if not np.all(keep_columns):
    removed = np.where(~keep_columns)[0]
    print("Removing degenerate columns:", removed.tolist())

sample_contrast = (
    sample_contrast[..., keep_columns]
)

n_samples = sample_contrast.shape[-1]

print("Using", n_samples, "samples.")


# ------------------------------------------------------------
# Put the true trial-wise delta into the same
# block/trial structure as the model
# ------------------------------------------------------------

n_blocks, n_trials = mask_np.shape

generative_mean = np.full(
    (n_blocks, n_trials),
    np.nan
)

for block, block_id in enumerate(block_ids_pk):

    block_data = df_pk[
        df_pk[BLOCK_COL] == block_id
    ]

    valid_trials = np.where(
        mask_np[block]
    )[0]

    n = min(
        len(valid_trials),
        len(block_data)
    )

    generative_mean[
        block,
        valid_trials[:n]
    ] = block_data[
        DELTA_COL
    ].to_numpy()[:n]


# ------------------------------------------------------------
# Find the subject belonging to each model block
# ------------------------------------------------------------

block_subjects = np.array([
    df_pk.loc[
        df_pk[BLOCK_COL] == block_id,
        SUBJECT_COL
    ].iloc[0]
    for block_id in block_ids_pk
])


# ------------------------------------------------------------
# Fit one kernel per subject
# ------------------------------------------------------------

kernels = {}

subjects = sorted(
    df_pk[SUBJECT_COL]
    .dropna()
    .unique()
)


for subject in subjects:

    subject_blocks = np.where(
        block_subjects == subject
    )[0]

    fluctuations = []
    means = []
    choices = []
    block_types = []


    for block in subject_blocks:

        valid = (
            mask_np[block]
            & np.isfinite(
                generative_mean[block]
            )
        )

        if not np.any(valid):
            continue


        # Subtract the trial's generative mean
        # from each sample's actual contrast.
        fluctuation = (
            sample_contrast[
                block,
                valid
            ]
            - generative_mean[
                block,
                valid,
                None
            ]
        )

        fluctuations.append(
            fluctuation
        )

        means.append(
            generative_mean[
                block,
                valid
            ]
        )

        choices.append(
            choice_right[
                block,
                valid
            ]
        )


        # Mark repetitive blocks as 1
        # and non-repetitive blocks as 0.
        transition = df_pk.loc[
            df_pk[BLOCK_COL] == block_ids_pk[block],
            TRANSITION_COL
        ].iloc[0]

        repetitive = (
            abs(
                transition
                - NON_REPETITIVE_VALUE
            ) > 1e-6
        )

        block_types.append(
            np.full(
                valid.sum(),
                float(repetitive)
            )
        )


    if not fluctuations:
        continue


    # Put all blocks for this subject together
    fluctuations = np.concatenate(
        fluctuations
    )

    means = np.concatenate(
        means
    )

    choices = np.concatenate(
        choices
    )

    block_types = np.concatenate(
        block_types
    )


    # --------------------------------------------------------
    # Logistic regression
    #
    # choice ~ generative mean
    #        + sample fluctuations
    #        + block type
    # --------------------------------------------------------

    design = np.column_stack([
        means,
        fluctuations,
        block_types
    ])

    clf = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=10000
    )

    clf.fit(
        design,
        choices
    )

    beta = clf.coef_[0]

    # First coefficient = generative mean
    # Next coefficients = psychophysical kernel
    # Last coefficient = block effect

    kernels[subject] = beta[
        1 : 1 + n_samples
    ]

    print(
        f"Subject {subject}: "
        f"{len(choices)} trials"
    )


# ------------------------------------------------------------
# Average across subjects
# ------------------------------------------------------------

kernel_matrix = np.stack([
    kernels[s]
    for s in sorted(kernels)
])

kernel_mean = kernel_matrix.mean(
    axis=0
)

kernel_sem = (
    kernel_matrix.std(
        axis=0,
        ddof=1
    )
    / np.sqrt(
        kernel_matrix.shape[0]
    )
)


# ------------------------------------------------------------
# Print the group kernel
# ------------------------------------------------------------

print()
print("Group psychophysical kernel:")

for sample, beta in enumerate(
    kernel_mean,
    start=1
):

    print(
        f"Sample {sample}: "
        f"{beta:.4f} +/- "
        f"{kernel_sem[sample - 1]:.4f}"
    )


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

samples = np.arange(
    1,
    n_samples + 1
)

plt.figure(
    figsize=(7, 5)
)

plt.fill_between(
    samples,
    kernel_mean - kernel_sem,
    kernel_mean + kernel_sem,
    alpha=0.2
)

# Connect the kernel points with straight lines
plt.plot(
    samples,
    kernel_mean,
    "o-",
    linewidth=2.5,
    markersize=7
)

plt.xlabel("Sample number")
plt.ylabel("Contrast fluctuations weight")

plt.xticks(samples)

plt.xlim(
    0.5,
    n_samples + 0.5
)

plt.axhline(
    0,
    linewidth=0.8,
    alpha=0.5
)

plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

plt.tight_layout()
plt.show()