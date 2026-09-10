"""
Uso de framework o biblioteca de aprendizaje máquina
para la implementación de una solución.

MODULO 2
Mario Alberto Pérez Barrera A01799928
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score,
    homogeneity_completeness_v_measure, silhouette_score,
    confusion_matrix, classification_report,
)

RANDOM_STATE = 42



def load_dataset():
    # Utilizo media lunas que funcionan bien con DBSCAN
    X, y = make_moons(n_samples=3000, noise=0.07, random_state=RANDOM_STATE)
    return X, y


def plot_k_distance(X_train, min_samples, out_path):
    """
    Tecnica estandar para elegir 'eps' en DBSCAN: se calcula la
    distancia al k-esimo vecino mas cercano (k = min_samples) para
    cada punto y se ordenan de forma ascendente. 
    """
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors.fit(X_train)
    distances, _ = neighbors.kneighbors(X_train)
    k_distances = np.sort(distances[:, -1])  # ascendente

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(k_distances)
    ax.set_xlabel("Puntos ordenados (ascendente)")
    ax.set_ylabel(f"Distancia al {min_samples}-esimo vecino mas cercano")
    ax.set_title("K-distance plot")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return k_distances


def find_eps_from_knee(k_distances, tail_fraction=0.15):
    """
    Se busca el salto (codo) en la cola de la curva, que es la diferencia
    de puntos consecutivos mas grande
"""
    n = len(k_distances)
    tail_start = int((1 - tail_fraction) * n)
    diffs = np.diff(k_distances[tail_start:])
    jump_idx = np.argmax(diffs) + tail_start
    return float(k_distances[jump_idx])


def assign_test_clusters(X_train, train_labels, X_test, eps):
    """
    Se busca el 'core point' de entrenamiento mas
    cercano; si esta a una distancia <= eps, el punto nuevo hereda su
    cluster; si no, se marca como ruido (-1).
    """
    core_mask = train_labels != -1
    core_points = X_train[core_mask]
    core_labels = train_labels[core_mask]

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(core_points)
    distances, indices = nn.kneighbors(X_test)

    test_labels = np.where(
        distances.flatten() <= eps,
        core_labels[indices.flatten()],
        -1,
    )
    return test_labels


#Se mapean los clusters encontrados
def map_clusters_to_true_labels(cluster_labels, true_labels):

    mapping = {}
    for cluster_id in np.unique(cluster_labels):
        if cluster_id == -1:
            continue
        mask = cluster_labels == cluster_id
        majority_label = np.bincount(true_labels[mask]).argmax()
        mapping[cluster_id] = majority_label

    mapped = np.array([
        mapping.get(c, -1) for c in cluster_labels
    ])
    return mapped, mapping


def plot_clusters(X, labels, title, out_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    unique_labels = np.unique(labels)
    for lab in unique_labels:
        mask = labels == lab
        if lab == -1:
            ax.scatter(X[mask, 0], X[mask, 1], c="lightgray", s=15,
                       label="Ruido", marker="x")
        else:
            ax.scatter(X[mask, 0], X[mask, 1], s=15, label=f"Cluster {lab}")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def eps_sensitivity_analysis(X_train_scaled, y_train, eps_candidates, min_samples):
    """
    Se prueba distintos valores de eps para ver su influencia entre que sea muy chico o muy grande
    """
    rows = []
    for eps in eps_candidates:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X_train_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1))
        ari = adjusted_rand_score(y_train, labels)
        rows.append((eps, n_clusters, n_noise, ari))
    return rows


def plot_confusion_matrix(cm, out_path, title="Matriz de confusion"):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Cluster asignado (mapeado)")
    ax.set_ylabel("Etiqueta real")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def main():
    print("Cargando dataset (make_moons)...")
    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    # DBSCAN es sensible a la escala -> estandarizar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    #                 ALGORITMO
    min_samples = 5
    print(f"\nGenerando k-distance plot para elegir 'eps' "
          f"(min_samples={min_samples})...")
    k_distances = plot_k_distance(X_train_scaled, min_samples,
                                   "k_distance_plot.png")
    # eps elegido automaticamente a partir del salto (codo) en la cola de la curva
    eps = find_eps_from_knee(k_distances)
    print(f"eps elegido (codo de la curva): {eps:.4f}")

    print("\n=== Analisis de sensibilidad de eps (mismo min_samples) ===")
    eps_candidates = [0.05, 0.10, eps, 0.40, 0.60]
    sensitivity_rows = eps_sensitivity_analysis(
        X_train_scaled, y_train, eps_candidates, min_samples
    )
    for eps_val, n_c, n_noise_pts, ari_val in sensitivity_rows:
        print(f"eps={eps_val:.3f} -> clusters={n_c}, ruido={n_noise_pts}, "
              f"ARI={ari_val:.3f}")

    print(f"\nEntrenando DBSCAN(eps={eps:.4f}, min_samples={min_samples}) "
          f"con scikit-learn...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    train_cluster_labels = dbscan.fit_predict(X_train_scaled)

    n_clusters = len(set(train_cluster_labels)) - (1 if -1 in train_cluster_labels else 0)
    n_noise = np.sum(train_cluster_labels == -1)
    print(f"Clusters encontrados: {n_clusters} | Puntos de ruido: {n_noise} "
          f"de {len(train_cluster_labels)}")

 
    print("\nClusters a los puntos de test "
          "(via core point mas cercano)...")
    test_cluster_labels = assign_test_clusters(
        X_train_scaled, train_cluster_labels, X_test_scaled, eps
    )
    train_mapped, mapping = map_clusters_to_true_labels(
        train_cluster_labels, y_train
    )
    test_mapped = np.array([mapping.get(c, -1) for c in test_cluster_labels])


    print("\n=== Metricas no supervisadas (train) ===")
    mask_clustered = train_cluster_labels != -1
    if n_clusters > 1:
        sil = silhouette_score(X_train_scaled[mask_clustered],
                                train_cluster_labels[mask_clustered])
        print(f"Silhouette sin ruido: {sil:.4f}")

    ari = adjusted_rand_score(y_train, train_cluster_labels)
    nmi = normalized_mutual_info_score(y_train, train_cluster_labels)
    homog, complet, v_measure = homogeneity_completeness_v_measure(
        y_train, train_cluster_labels
    )
    print(f"Adjusted Rand Index (vs etiquetas reales): {ari:.4f}")
    print(f"Normalized Mutual Information: {nmi:.4f}")
    print(f"Homogeneity: {homog:.4f} | Completeness: {complet:.4f} | "
          f"V-measure: {v_measure:.4f}")


    #Metricas
    print("\nEvaluacion en test: ")
    cm = confusion_matrix(y_test, test_mapped, labels=[0, 1])
    print("Matriz de confusion:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_test, test_mapped, digits=3))

    noise_test = np.sum(test_cluster_labels == -1)
    print(f"Puntos marcados como ruido: {noise_test} de {len(X_test)}")

  
    print("\nGraficas...")
    plot_clusters(X_train_scaled, train_cluster_labels,
                  "DBSCAN - Clusters encontrados (train)",
                  "clusters_train.png")
    plot_clusters(X_test_scaled, test_cluster_labels,
                  "DBSCAN - Clusters asignados (test)",
                  "clusters_test.png")
    plot_confusion_matrix(cm, "confusion_matrix.png")
    print("Listo: k_distance_plot.png, clusters_train.png, "
          "clusters_test.png, confusion_matrix.png guardadas.")


if __name__ == "__main__":
    main()
