# 🔍 Détection d'Anomalies dans des Objets Industriels 3D

> **Projet de Fin d'Études (PFE) — Stage Ingénierie**  
> 🎓 **SUPMTI** — Filière : Ingénierie des Systèmes Informatiques (ISI)  
> 🏢 **3D SMART FACTORY** — Mohammedia, Maroc  
> 📅 Période de stage : **15 Février 2025 → 03 Juillet 2025**

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Open3D](https://img.shields.io/badge/Open3D-0.18-green)](http://www.open3d.org)
[![Colab](https://img.shields.io/badge/Google%20Colab-GPU%20T4-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com)
[![MVTec 3D-AD](https://img.shields.io/badge/Dataset-MVTec%203D--AD-blue)](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad)
[![Status](https://img.shields.io/badge/Statut-En%20cours-orange)]()
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

Dans un contexte de **contrôle qualité industriel automatisé**, ce projet développe un système de détection d'anomalies sur des objets 3D à partir de nuages de points. Le système apprend la géométrie normale d'une pièce industrielle et détecte automatiquement toute déviation anormale — sans avoir jamais vu de défaut pendant l'entraînement.

### Problématique

> Comment détecter automatiquement des défauts structurels (fissures, trous, contaminations) sur des pièces industrielles 3D à partir de leurs données géométriques, en utilisant uniquement des exemples de pièces normales pour l'entraînement ?

### Approche retenue

- **Apprentissage non supervisé** : le modèle apprend uniquement sur des pièces normales
- **Architecture** : Autoencodeur basé sur PointNet (encoder + decoder)
- **Score d'anomalie** : Chamfer Distance entre pièce originale et reconstruction
- **Localisation** : carte d'erreur point à point pour localiser précisément les défauts

---

## 🏗️ Architecture du système

```
┌─────────────────────────────────────────────────────────────┐
│                    PIPELINE COMPLET                         │
│                                                             │
│  Fichier .tiff (H×W×3)                                      │
│       │                                                     │
│  ┌────▼──────────────┐                                      │
│  │   PRÉTRAITEMENT   │  normalisation · 2048 pts · augment  │
│  └────┬──────────────┘                                      │
│       │  Tenseur (B, 2048, 3)                               │
│  ┌────▼──────────────┐                                      │
│  │     ENCODER       │  Conv1D : 3→64→128→256               │
│  │   (PointNet)      │  MaxPool global → z ∈ ℝ¹²⁸          │
│  └────┬──────────────┘                                      │
│  ┌────▼──────────────┐                                      │
│  │     DECODER       │  MLP : 128→256→512→1024→N×3         │
│  └────┬──────────────┘                                      │
│  ┌────▼──────────────────────────────┐                      │
│  │  SCORE D'ANOMALIE                 │                      │
│  │  Chamfer Distance(orig, recon)    │                      │
│  │  Score élevé → pièce défectueuse  │                      │
│  └───────────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset — MVTec 3D-AD · Catégorie `bagel`

| Split | Échantillons | Composition |
|-------|-------------|-------------|
| `train/` | **244** | 100% normaux |
| `validation/` | **22** | 100% normaux |
| `test/` | **110** | 20% normaux · 80% défectueux |

| Défaut | Description | Échantillons |
|--------|-------------|-------------|
| `crack` | Fissures nettes en surface | 22 |
| `hole` | Trous anormaux dans la structure | 21 |
| `contamination` | Dépôts ou corps étrangers | 22 |
| `combined` | Plusieurs défauts simultanés | 23 |

> ⚠️ Le dataset n'est pas inclus dans ce dépôt (14.2 GB).  
> Téléchargement : https://www.mvtec.com/company/research/datasets/mvtec-3d-ad

---

## 🗂️ Structure du dépôt

```
PFE-Detection-Anomalies-3D/
│
├── 📓 notebooks/
│   ├── Phase1_Exploration_MVTec3D.ipynb       ✅ Terminé
│   ├── Phase2_Pretraitement_MVTec3D.ipynb     ✅ Terminé
│   ├── Phase3_Modelisation_MVTec3D.ipynb      ⏳ En cours
│   └── Phase4_Evaluation_MVTec3D.ipynb        🔜 À venir
│
├── 🐍 src/
│   ├── preprocessing/
│   │   ├── loader.py       # Chargement fichiers .tiff MVTec
│   │   ├── normalize.py    # Normalisation + augmentation
│   │   └── dataset.py      # BagelDataset + DataLoaders
│   ├── models/
│   │   └── autoencoder.py  # PointNet Encoder + Decoder
│   ├── evaluation/
│   │   ├── metrics.py      # AUC · F1 · IoU · PRO-score
│   │   └── visualize.py    # Cartes d'anomalie
│   └── utils/
│       └── chamfer.py      # Chamfer Distance loss
│
├── 📊 results/
│   ├── figures/            # Courbes et visualisations
│   └── metrics/            # Résultats numériques
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🚀 Avancement

| Phase | Description | Résultats clés | Statut |
|-------|-------------|----------------|--------|
| **Phase 1** | Exploration du dataset | 540 854 pts/pièce · 4 types défauts | ✅ |
| **Phase 2** | Prétraitement & DataLoaders | Tenseur (16,2048,3) · 0 NaN/Inf | ✅ |
||**Phase 3** | Modélisation BTF RGB+Depth | AUC=0.9664 · F1=0.9609 · Précision=0.9451 | ✅ |
| **Phase 4** | Évaluation AUC · F1 · IoU | — | 🔜 |
| **Phase 5** | Déploiement & démo finale | — | 🔜 |

---

## ⚙️ Installation

```bash
git clone https://github.com/VOTRE_USERNAME/PFE-Detection-Anomalies-3D.git
cd PFE-Detection-Anomalies-3D
pip install -r requirements.txt
```

---

## 📈 Métriques cibles

| Métrique | Objectif |
|----------|----------|
| AUC-ROC | > 0.85 |
| F1-Score | > 0.80 |
| Précision | > 0.80 |
| Rappel | > 0.80 |
| IoU | > 0.60 |

---

## 📚 Références

- Bergmann et al. — *The MVTec 3D-AD Dataset for Unsupervised 3D Anomaly Detection* (VISAPP 2022)
- Qi et al. — *PointNet: Deep Learning on Point Sets for 3D Classification* (CVPR 2017)
- Qi et al. — *PointNet++: Deep Hierarchical Feature Learning on Point Sets* (NeurIPS 2017)

---

<div align="center">
<strong>3D SMART FACTORY</strong> · Mohammedia, Maroc<br>
<em>Stage PFE — SUPMTI ISI · Olivier OUEDRAOGO · 2025</em>
</div>
