# 📅 Journal de développement

> Ce fichier documente la chronologie réelle du projet,
> depuis le démarrage du stage jusqu'à la publication du dépôt GitHub.

---

## Contexte de publication

Ce dépôt a été publié en **mai 2026** lors de la finalisation de la
documentation et du rapport de soutenance.  
Le projet a été intégralement réalisé durant le stage chez
**3D SMART FACTORY** du **15 Février au 03 Juillet 2025**.

---

## Chronologie réelle du projet

| Période | Phase | Travaux réalisés |
|---------|-------|-----------------|
| **Fév 2025** — Semaines 1-2 | Phase 1 — Exploration | Prise en main du dataset MVTec 3D-AD (14.2 GB), exploration des 10 catégories, analyse de la structure (train/val/test), visualisation des nuages de points 3D et des masques GT, sélection de la catégorie `bagel` |
| **Mar 2025** — Semaines 3-4 | Phase 2 — Prétraitement | Pipeline de prétraitement complet : normalisation, sous-échantillonnage (2048 pts), augmentation de données, construction des DataLoaders PyTorch (tenseur 16×2048×3 validé) |
| **Avr 2025** — Semaines 5-8 | Phase 3 — Modélisation | Exploration et comparaison de 4 approches (autoencodeur, PatchCore XYZ, fusion globale, BTF). Implémentation finale BTF RGB+Depth — AUC=0.9514, F1=0.9392 |
| **Mai 2025** — Semaines 9-10 | Phase 4 — Évaluation | Génération des cartes d'anomalie, calcul IoU, analyse par type de défaut, 3 tests de validation du modèle, sauvegarde des métriques JSON |
| **Juin 2025** — Semaines 11-12 | Phase 5 — Déploiement | Développement de l'interface Gradio, déploiement avec lien public, test sur image externe, sauvegarde du pipeline complet |
| **Juil 2025** — Semaine 13 | Finalisation stage | Rédaction du rapport de stage, préparation de la soutenance, fin officielle du stage le 03 Juillet 2025 |
| **Mai 2026** | Publication GitHub | Nettoyage et structuration du code, rédaction du README complet, publication du dépôt GitHub public |

---

## Environnement de travail

| Composant | Détail |
|-----------|--------|
| Plateforme | Google Colab (GPU Tesla T4) |
| Stockage | Google Drive — dossier `Detection_Anomalie_3D/` |
| Dataset | MVTec 3D-AD `.tar.xz` — 14.2 GB |
| Framework | PyTorch 2.x + torchvision |
| Interface | Gradio 4.x |

---

## Décisions architecturales clés

| Date | Décision | Justification |
|------|----------|---------------|
| Mar 2025 | Catégorie `bagel` retenue | Géométrie régulière, défauts bien définis, idéal pour validation |
| Avr 2025 | Abandon autoencodeur (AUC=0.657) | Trop bon en reconstruction — incapable de distinguer normaux/défauts |
| Avr 2025 | Adoption BTF RGB+Depth | Fusion multimodale indispensable pour les défauts fins (crack) |
| Mai 2025 | Seuil = 0.4037 | Calibré par maximisation F1 sur le split validation |

---

*Olivier OUEDRAOGO — SUPMTI ISI — 3D SMART FACTORY — 2025*
