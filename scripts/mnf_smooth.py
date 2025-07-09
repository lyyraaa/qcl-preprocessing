# %%
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import sys
from sklearn.model_selection import KFold

sys.path.append('/mnt/iusers01/eee01/w37262do/scratch/PyIR/src')
from pyir_spectralcollection import PyIR_SpectralCollection as pir
import pyir_pca as pir_pca

# %%
# Experiment
is_local = False
seed = 0 if is_local else int(sys.argv[-1])
np.random.seed(seed=seed)
balanced = True
even_or_odd = "even" if balanced else "odd"
annotation_class_colors = np.array([[0, 255, 0], [128, 0, 128], [255, 0, 255], [0, 0, 255], [255, 165, 0], [255, 0, 0]])
annotation_class_names = np.array(['epithelium_n', 'stroma_n', 'epithelium_c', 'stroma_c', 'corpora_amylacea', 'blood'])


# %%
def fast_mnf_denoise(hyperspectraldata, SNR=5, bands=0):
    # Check if the input is 3D and reshape to 2D if needed
    if hyperspectraldata.ndim == 3:
        m, n, s = hyperspectraldata.shape
        X = np.reshape(hyperspectraldata, (-1, s))  # Reshape to 2D
    elif hyperspectraldata.ndim == 2:
        X = hyperspectraldata
        m, n = X.shape
        s = n  # If already 2D, assume second dimension is the spectral dimension
    else:
        raise ValueError("Input C must be either 2D or 3D.")

    # Step 2: Create the dX matrix
    dX = np.zeros((m, s))
    for i in range(m - 1):
        dX[i, :] = X[i, :] - X[i + 1, :]

    # Step 3: Perform eigenvalue decomposition of dX' * dX
    S1, U1 = np.linalg.eigh(dX.T @ dX)
    ix = np.argsort(S1)[::-1]  # Sort in descending order
    U1 = U1[:, ix]
    D1 = S1[ix]
    diagS1 = 1.0 / np.sqrt(D1)

    # Step 4: Compute weighted X
    wX = X @ U1 @ np.diag(diagS1)

    # Step 5: Perform eigenvalue decomposition of wX' * wX
    S2, U2 = np.linalg.eigh(wX.T @ wX)
    iy = np.argsort(S2)[::-1]  # Sort in descending order
    U2 = U2[:, iy]
    D2 = S2[iy]

    # Step 6: Retain top K components according to input SNR threshold
    S2_diag = D2 - 1
    if bands != 0:
        K = bands
    else:
        K = np.sum(S2_diag > SNR)
    U2 = U2[:, :K]

    # Step 7: Compute Phi_hat and Phi_tilde
    Phi_hat = U1 @ np.diag(diagS1) @ U2
    Phi_tilde = U1 @ np.diag(np.sqrt(D1)) @ U2

    # Step 8: Project data onto MNF components and reshape to original dimensions
    mnfX = X @ Phi_hat
    Xhat = mnfX @ Phi_tilde.T

    if hyperspectraldata.ndim == 3:
        clean_data = np.reshape(Xhat, (m, n, s))  # Reshape back to 3D if input was 3D
    else:
        clean_data = Xhat  # Keep 2D if input was 2D

    return clean_data


# %%
kit = pir()
n_comp = 30 # todo


def preprocess_0(spectra, wavenumbers):
    spectra = kit.fast_mnf_denoise(spectra, bands=n_comp)
    return spectra, wavenumbers


def preprocess_1(spectra, wavenumbers):
    spectra, wavenumbers = kit.remove_range(1360, 1490, spectra, wavenumbers)
    spectra = kit.all_spec_min2zero(spectra)
    spectra = kit.vector_norm(spectra)
    spectra = kit.fast_mnf_denoise(spectra, bands=n_comp)
    return spectra, wavenumbers


def preprocess_2(spectra, wavenumbers):
    spectra, wavenumbers = kit.remove_range(1360, 1490, spectra, wavenumbers)
    spectra = kit.all_spec_min2zero(spectra)
    spectra = kit.vector_norm(spectra)
    spectra = kit.fast_mnf_denoise(spectra, bands=n_comp)
    spectra, wavenumbers = kit.data_deriv(spectra, wavenumbers, 21, 5, 1)
    return spectra, wavenumbers


def preprocess_3(spectra, wavenumbers):
    spectra, wavenumbers = kit.remove_range(1360, 1490, spectra, wavenumbers)
    spectra = kit.all_spec_min2zero(spectra)
    spectra = kit.vector_norm(spectra)
    spectra = kit.fast_mnf_denoise(spectra, bands=n_comp)
    spectra, wavenumbers = kit.data_deriv(spectra, wavenumbers, 21, 5, 2)
    return spectra, wavenumbers


pipelines = [
    preprocess_3,
    preprocess_1,
    preprocess_2,
    preprocess_3,
]
# %%
spectra = np.load(f"data_{even_or_odd}.npy")
labels = np.load(f"labels_{even_or_odd}.npy")
pids = np.load(f"pids_{even_or_odd}.npy")
wavenumbers = np.load("wavenumbers.npy")
print(
    f"Assembled data with {len(spectra)} samples and {len(labels)} labels and {len(pids)} pids and {len(wavenumbers)} wavenumbers")


# %% md
## Run experiment for every pipeline
# %%
def train_model(train_data, train_labels):
    rf_model = Pipeline([
        ("normalise", StandardScaler(),),
        ("randomforest", RandomForestClassifier(n_estimators=500, min_samples_leaf=10))
    ])
    rf_model.fit(train_data, train_labels)
    return rf_model


def test_model(model, test_data, test_labels, dataset_name=None):
    pred = model.predict(test_data)
    acc = accuracy_score(test_labels, pred)
    f1m = f1_score(test_labels, pred, average='macro')
    f1 = f1_score(test_labels, pred, average=None)
    print(f"\t{dataset_name} | OA: {acc:.4f} | f1: {f1m:.4f}")
    return acc, f1m, f1


# %%
fig, ax = plt.subplots(figsize=(14, 4))
for cls in range(len(annotation_class_names)):
    ax.plot(
        wavenumbers,
        np.mean(spectra[np.where(labels == cls)], axis=0),
        color=annotation_class_colors[cls] / 255.0,
        label=annotation_class_names[cls]
    )
ax.legend()
ax.set_title("mean raw spectra per-class")
# %%
for t_idx in range(0, len(pipelines)):
    ### RUN K-FOLD SPLIT
    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    unique_pids = np.unique(pids)
    accs = []
    f1ms = []
    f1s = []
    for i, (train_pids, test_pids) in enumerate(kfold.split(unique_pids)):
        where_train = np.where(np.isin(pids, unique_pids[train_pids]))[0]
        where_test = np.where(np.isin(pids, unique_pids[test_pids]))[0]

        ### TRANSFORM SPECTRA
        processed_spectra_where_train, _ = pipelines[t_idx](spectra[where_train], wavenumbers)
        processed_spectra_where_test, _ = pipelines[t_idx](spectra[where_test], wavenumbers)

        model = train_model(processed_spectra_where_train, labels[where_train])
        print(f"On split {i + 1}/5", end=" ")
        test_model(model, processed_spectra_where_train, labels[where_train], dataset_name="TRAIN ---")
        print(f"On split {i + 1}/5", end=" ")
        acc, f1m, f1 = test_model(model, processed_spectra_where_test, labels[where_test], dataset_name="TEST ----")
        print()
        accs.append(acc);
        f1ms.append(f1m);
        f1s.append(f1)

    print(f"Average metrics over 5-fold CV for pipeline {t_idx}:")
    print(f"Accuracy: {np.mean(accs):.4f}")
    print(f"Macro F1: {np.mean(f1ms):.4f}")
    for cls_idx, f1 in enumerate(np.stack(f1s, axis=0).mean(axis=0)):
        print(f"\t\t{annotation_class_names[cls_idx]}{(20 - len(annotation_class_names[cls_idx])) * ' '} : {f1:.4f}")
    print("\n\n\n")
# %%
