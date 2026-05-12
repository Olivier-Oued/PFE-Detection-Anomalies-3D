# 🔍 Détection d'Anomalies dans des Objets Industriels 3D

> **Projet de Fin d'Études (PFE) — Stage Ingénierie**  
> 🎓 **SUPMTI** — Filière : Ingénierie des Systèmes Informatiques (ISI)  
> 🏢 **3D SMART FACTORY** — Mohammedia, Maroc  
> 📅 Période de stage : **15 Février 2025 → 03 Juillet 2025**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![ResNet18](https://img.shields.io/badge/ResNet18-ImageNet-orange)](https://pytorch.org/vision/stable/models.html)
[![Gradio](https://img.shields.io/badge/Gradio-Interface-FF7C00?logo=gradio&logoColor=white)](https://gradio.app)
[![Open3D](https://img.shields.io/badge/Open3D-0.18-green)](http://www.open3d.org)
[![Colab](https://img.shields.io/badge/Google%20Colab-GPU%20T4-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com)
[![MVTec 3D-AD](https://img.shields.io/badge/Dataset-MVTec%203D--AD-blue)](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad)
[![Status](https://img.shields.io/badge/Statut-Terminé-brightgreen)]()
[![License](https://img.shields.io/badge/Licence-MIT-green)](LICENSE)

---

## 👤 Auteur

| | |
|---|---|
| **Stagiaire** | Olivier OUEDRAOGO |
| **École** | SUPMTI — Ingénierie des Systèmes Informatiques (ISI) |
| **Entreprise d'accueil** | 3D SMART FACTORY, Mohammedia |
| **Directeur de stage** | M. Bertin Thierry Maurice Leon |
| **Encadrants académiques** | M. Hamza MOUNCIF · Mme Chaymae BENHAMMACHT |
| **Période** | 15 Février 2025 — 03 Juillet 2025 |

---

## 📋 Résumé du projet

Dans un contexte de **contrôle qualité industriel automatisé**, ce projet développe un système de détection d'anomalies sur des objets 3D à partir de données RGB et de profondeur. Le système apprend la distribution normale d'une pièce industrielle et détecte automatiquement toute déviation anormale — sans avoir jamais vu de défaut pendant l'entraînement (**apprentissage non supervisé**).

### Problématique

> Comment détecter automatiquement des défauts structurels (fissures, trous, contaminations) sur des pièces industrielles 3D à partir de leurs données RGB et géométriques, en utilisant uniquement des exemples de pièces normales pour l'entraînement ?

---

## 🏗️ Architecture retenue — BTF (Back to Feature)

Après comparaison de plusieurs approches, l'approche **BTF (Back to Feature)** a été retenue :

```
┌─────────────────────────────────────────────────────────────┐
│                    PIPELINE BTF                             │
│                                                             │
│  Image RGB (.png)          Carte Depth Z (.tiff)            │
│       │                          │                          │
│  ┌────▼──────────────────────────▼────┐                     │
│  │   ResNet18 pré-entraîné ImageNet   │                     │
│  │   Layer1 (64ch) · Layer2 (128ch)   │                     │
│  │   Layer3 (256ch) — poids figés     │                     │
│  └────┬──────────────────────────┬────┘                     │
│       │  Features RGB            │  Features Depth          │
│  ┌────▼──────────────────────────▼────┐                     │
│  │   Fusion multi-échelle (896 dim)   │                     │
│  │   PCA → 128 dimensions             │                     │
│  └────┬───────────────────────────────┘                     │
│       │                                                     │
│  ┌────▼──────────────────────────────┐                      │
│  │  KNN cosine (k=9)                 │                      │
│  │  Banque de 244 pièces normales    │                      │
│  └────┬──────────────────────────────┘                      │
│       │                                                     │
│  ┌────▼──────────────────────────────┐                      │
│  │  Score = max distance locale      │                      │
│  │  Score > seuil → DÉFAUT détecté   │                      │
│  └───────────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

### Pourquoi BTF ?

| Approche testée | AUC | Statut |
|---|---|---|
| Autoencodeur Chamfer Distance | 0.657 | Abandonné |
| PatchCore XYZ-only | 0.468 | Abandonné |
| Fusion RGB+XYZ globale | 0.662 | Abandonné |
| **BTF RGB + Depth (retenu)** | **0.9514** | ✅ Retenu |

---

## 📊 Dataset — MVTec 3D-AD · Catégorie `bagel`

| Split | Échantillons | Composition |
|-------|-------------|-------------|
| `train/` | **244** | 100% normaux — apprentissage non supervisé |
| `validation/` | **22** | 100% normaux — calibration du seuil |
| `test/` | **110** | 20% normaux · 80% défectueux |

| Défaut | Description | Échantillons | AUC spécifique |
|--------|-------------|-------------|----------------|
| `crack` | Fissures nettes en surface | 22 | 0.9917 |
| `hole` | Trous anormaux dans la structure | 21 | 0.9459 |
| `contamination` | Dépôts ou corps étrangers | 22 | 0.9360 |
| `combined` | Plusieurs défauts simultanés | 23 | 0.9901 |

> ⚠️ Le dataset n'est pas inclus dans ce dépôt (14.2 GB).  
> Téléchargement : https://www.mvtec.com/company/research/datasets/mvtec-3d-ad

---

## 🗂️ Structure du dépôt

```
PFE-Detection-Anomalies-3D/
│
├── 📓 notebooks/
│   ├── Phase1_Exploration_MVTec3D.ipynb       ✅ Exploration & analyse
│   ├── Phase2_Pretraitement_MVTec3D.ipynb     ✅ Pipeline prétraitement
│   ├── Phase3_Modelisation_MVTec3D.ipynb      ✅ Modélisation BTF
│   ├── Phase4_Evaluation_MVTec3D.ipynb        ✅ Évaluation & cartes
│   └── Phase5_Demo_Gradio.ipynb               ✅ Interface Gradio
│
├── 🐍 src/
│   ├── preprocessing/
│   │   ├── loader.py       # Chargement fichiers .tiff MVTec
│   │   ├── normalize.py    # Normalisation + augmentation
│   │   └── dataset.py      # BagelDataset + DataLoaders
│   ├── models/
│   │   └── autoencoder.py  # PointNet Encoder + Decoder (v1)
│   ├── evaluation/
│   │   ├── metrics.py      # AUC · F1 · IoU · PRO-score
│   │   └── visualize.py    # Cartes d'anomalie
│   └── utils/
│       └── chamfer.py      # Chamfer Distance loss
│
├── 📊 results/
│   ├── figures/            # Courbes et visualisations
│   └── metrics/            # metrics_final.json
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Avancement du projet — TOUTES PHASES TERMINÉES

| Phase | Description | Résultats clés | Statut |
|-------|-------------|----------------|--------|
| **Phase 1** | Exploration du dataset MVTec 3D-AD | 540 854 pts/pièce · bagel retenu · 4 types défauts | ✅ |
| **Phase 2** | Prétraitement & DataLoaders PyTorch | Tenseur (16,2048,3) · pipeline validé · 0 NaN/Inf | ✅ |
| **Phase 3** | Modélisation BTF RGB+Depth | AUC=0.9514 · F1=0.9392 · Précision=0.9140 | ✅ |
| **Phase 4** | Évaluation · IoU · cartes d'anomalie | Cartes heatmap · démo interactive · métriques JSON | ✅ |
| **Phase 5** | Déploiement — Interface Gradio | Interface live · upload custom · démo soutenance | ✅ |

---

## 📈 Résultats finaux

### Métriques de détection globale

| Métrique | Objectif cahier des charges | Résultat obtenu | |
|----------|----------------------------|-----------------|---|
| **AUC-ROC** | > 0.85 | **0.9514** | ✅ +10pts |
| **F1-Score** | > 0.80 | **0.9392** | ✅ +14pts |
| **Précision** | > 0.80 | **0.9140** | ✅ +11pts |
| **Rappel** | > 0.80 | **0.9659** | ✅ +17pts |

### AUC par type de défaut

| Défaut | AUC | Interprétation |
|--------|-----|----------------|
| crack | 0.9917 | Fissures très bien détectées (visibles en RGB) |
| combined | 0.9901 | Défauts multiples = score plus élevé |
| hole | 0.9459 | Trous bien détectés |
| contamination | 0.9360 | Dépôts subtils — bonne détection |

### Matrice de confusion (test — 110 pièces)

| | Prédit Normal | Prédit Défaut |
|---|---|---|
| **Réel Normal** | 14 ✅ | 8 |
| **Réel Défaut** | 3 | 85 ✅ |

### Validation du modèle

| Test | Résultat |
|------|----------|
| Anti-fuite (train/test) | ✅ 0 chevauchement |
| Robustesse (image aléatoire) | ✅ Score 2× supérieur aux normaux |
| Variation sémantique | ✅ AUC varie de 0.936 à 0.992 par type |

---

## 🖥️ Interface de démonstration — Gradio

L'interface permet :
- **Onglet 1** : sélection depuis le dataset (15 exemples couvrant tous les types)
- **Onglet 2** : upload d'une image RGB externe → analyse instantanée

```python
# Lancer l'interface depuis le notebook Phase5_Demo_Gradio.ipynb
# Nécessite la session Phase 3 active (extractor + knn_btf + pca + THRESHOLD)
demo.launch(share=True)  # génère un lien public gradio.live
```

---

## ⚙️ Installation

```bash
git clone https://github.com/VOTRE_USERNAME/PFE-Detection-Anomalies-3D.git
cd PFE-Detection-Anomalies-3D
pip install -r requirements.txt
```

### Utilisation sur Google Colab (recommandé)

```python
# 1. Monter le Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Extraire le dataset (une seule fois)
import subprocess
subprocess.run(['tar', '-xf',
    '/content/drive/MyDrive/Detection_Anomalie_3D/mvtec_3d_anomaly_detection.tar.xz',
    '-C', '/content/mvtec3d'])

# 3. Exécuter les notebooks dans l'ordre
# Phase1 → Phase2 → Phase3 → Phase4 → Phase5
```

---

## 📦 Dépendances principales

```
torch >= 2.0.0
torchvision >= 0.15.0
open3d >= 0.18.0
tifffile >= 2023.7.0
numpy >= 1.24.0
scikit-learn >= 1.3.0
matplotlib >= 3.7.0
plotly >= 5.15.0
gradio >= 4.0.0
Pillow >= 10.0.0
```

---

## 📚 Références scientifiques

- Bergmann et al. — *The MVTec 3D-AD Dataset for Unsupervised 3D Anomaly Detection* (VISAPP 2022)
- Horwitz & Hoshen — *Back to the Feature: Classical 3D Features are (Almost) All You Need for 3D Anomaly Detection* (CVPR 2023)
- Qi et al. — *PointNet: Deep Learning on Point Sets for 3D Classification* (CVPR 2017)
- Roth et al. — *Towards Total Recall in Industrial Anomaly Detection (PatchCore)* (CVPR 2022)

---

## 📄 Licence

Ce projet est sous licence MIT — voir [LICENSE](LICENSE).

---

<div align="center">
<strong>3D SMART FACTORY</strong> · Mohammedia, Maroc<br>
<em>Stage PFE — SUPMTI ISI · Olivier OUEDRAOGO · 2025</em><br><br>
<img src="https://img.shields.io/badge/Toutes%20phases-Terminées-brightgreen?style=for-the-badge"/>
</div>
