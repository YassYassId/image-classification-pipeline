# Rapport – Pipeline CIFAR‑10: MLOps avec DVC, Google Drive et Docker (CPU‑only)

Étudiants: IDRISSI Yassine, FANNAN Noussair  
Master SDIA

---

## 1. Introduction

### 1.1 Contexte
Ce projet implémente un pipeline complet d’apprentissage automatique pour la classification d’images sur le dataset CIFAR‑10. Il met l’accent sur la reproductibilité, la traçabilité des données et la portabilité des exécutions grâce à DVC (Data Version Control) et Docker. Le stockage distant des données et artefacts est géré par un remote DVC sur Google Drive.

### 1.2 Objectifs
- Construire un pipeline reproductible (prepare → train → evaluate) traçable par DVC.
- Séparer le code (Git) des données/modèles (DVC remote GDrive).
- Proposer une containerisation CPU‑only pour exécuter le pipeline de manière isolée.
- Documenter des modes d’authentification GDrive (OAuth et Service Account) adaptés aux environnements headless (Docker) et CI.

---

## 2. Architecture MLOps mise en place

### 2.1 Vue d’ensemble
```
┌──────────────────────────┐          ┌────────────────────────┐
│   Référentiel Git        │          │   Google Drive         │
│   (code + méta DVC)      │  DVC     │   (Remote DVC)         │
│   - src/, dvc.yaml       ├─────────▶│   - cache objets       │
│   - params.yaml          │          │   - artefacts modèles  │
└─────────────┬────────────┘          └──────────────┬─────────┘
              │                                      │
              │ dvc repro / dvc pull                 │
              ▼                                      │
       ┌───────────────┐                             │
       │   Docker      │◀────────────────────────────┘
       │  (CPU only)   │  (auth OAuth / Service Account)
       └───────────────┘
```

### 2.2 Composants
- DVC: orchestration des dépendances/outputs des stages, suivi des artefacts (sans versionner les gros fichiers dans Git).
- Remote DVC Google Drive: stockage des objets (cache) et partage multi‑machines.
- Docker: exécution isolée et reproductible (image CPU‑only basée sur `python:3.10-slim`).

---

## 3. Données, pipeline et paramètres

### 3.1 Données attendues
Le projet n’intègre pas de code de téléchargement. Le dataset CIFAR‑10 doit être présent localement ou récupéré via DVC:
```
data/raw/cifar10/
  train/<classe_0>/ ... <classe_9>/
  test/<classe_0>/  ... <classe_9>/
```

### 3.2 Stages DVC (dvc.yaml)
- prepare: `python -u src/prepare.py`
  - deps: `src/prepare.py`, `src/utils/helpers.py`, `params.yaml`, `data/raw/cifar10`
  - outs: `data/processed`
- train: `python -u src/train.py`
  - deps: `src/train.py`, `src/utils/helpers.py`, `params.yaml`, `data/processed/*.json`
  - outs: `models`, metrics: `metrics/train.json`
- evaluate: `python -u src/evaluate.py`
  - deps: `src/evaluate.py`, `params.yaml`, `models`, `data/processed/test.json`
  - metrics: `metrics/eval.json`, plots: `metrics/plots/`

Ces stages s’exécutent via `dvc repro`, qui ne relance que ce qui a changé.

### 3.3 Paramètres (params.yaml)
- `image_size`: 32 (CNN) ; 96 pour MobileNetV2 (interne si besoin)
- `val_split`: fraction du train pour validation
- `batch_size`, `learning_rate`
- `epochs_cnn`, `epochs_tl`
- `model_choice`: `cnn` ou `mobilenetv2`

---

## 4. Implémentation modèle (aperçu)
Deux approches CPU‑friendly:
- CNN simple: 2 blocs convolutionnels (32, 64), petite tête dense, dropout modéré, entrainé sur 32×32.
- MobileNetV2 transfert: base gelée, petite tête de classification, entrainement réduit, entrées 96×96.

Sorties attendues:
- Modèles: `models/cnn.keras`, `models/mobilenetv2.keras`
- Métriques: `metrics/train.json` (train/val), `metrics/eval.json` (test: accuracy, macro‑precision/recall/F1)
- Visualisations: `metrics/plots/confusion_matrix_<model>.png`, `metrics/plots/misclassified_<model>.png`

---

## 5. Exécution locale (hors Docker)
1) Python 3.10 recommandé, créer/activer un venv.
2) Installer les dépendances:
```
pip install -r requirements.txt
```
3) Vérifier la présence des données sous `data/raw/cifar10/` ou s’assurer que DVC peut les tirer depuis le remote.
4) Lancer le pipeline:
```
dvc repro
```

Reprendre après modification de `params.yaml` ou des données; DVC n’exécutera que les étapes impactées.

---

## 6. Remote DVC Google Drive
Le remote est déclaré dans `.dvc/config` (URL GDrive). Ne pas versionner d’informations sensibles: `.dvc/config.local` contient les identifiants locaux et doit rester ignoré par Git.

Modes d’authentification supportés:
- OAuth interactif (machine avec navigateur) — par défaut via DVC/pydrive2.
- Service Account (non‑interactif) — recommandé pour CI/Docker headless.

Points clés:
- Pour OAuth, DVC peut tenter un flux « navigateur local ». En environnements headless, privilégier le device code ou l’usage d’un Service Account.
- Pour Service Account, partager le dossier GDrive du remote avec l’email du SA et fournir le JSON de clés.

---

## 7. Exécution avec Docker (CPU‑only)

### 7.1 Construction de l’image
```
docker build -t cifar10-dvc .
```

### 7.2 Chemins et montages (Windows PowerShell)
- Partagez votre disque (Docker Desktop → Settings → Resources → File Sharing) pour permettre les montages depuis `D:`.
- Utilisez des backslashes: `-v ${PWD}\data:/app/data`.

### 7.3 Scénarios d’exécution

A) Utiliser les données locales (plus simple, sans remote)
```
$env:SKIP_DVC_PULL = "1"
docker run --rm -it ^
  -e SKIP_DVC_PULL=$env:SKIP_DVC_PULL ^
  -v ${PWD}\data:/app/data ^
  -v ${PWD}\models:/app/models ^
  -v ${PWD}\metrics:/app/metrics ^
  cifar10-dvc dvc repro
```

B) OAuth dans le conteneur (interactif)
```
docker run --rm -it cifar10-dvc dvc pull
# Suivre l’URL/code affichés (device code si disponible), puis:
docker run --rm -it cifar10-dvc dvc repro
```
Astuce: pour accélérer les ré‑exécutions, montez un cache persistant:
```
mkdir ${PWD}\_dvc_cache 2>$null

docker run --rm -it ^
  -v ${PWD}\_dvc_cache:/root/.cache ^
  cifar10-dvc dvc pull
```

C) Service Account (non‑interactif, recommandé pour CI/headless)
1) Partager le dossier GDrive du remote avec l’email du SA.  
2) Utiliser un fichier d’environnement pour injecter la clé JSON:
```
# Créer un fichier env.list contenant:
# GDRIVE_CREDENTIALS_DATA={...json en une seule ligne...}

docker run --rm -it --env-file env.list cifar10-dvc dvc pull

docker run --rm -it ^
  --env-file env.list ^
  -v ${PWD}\models:/app/models ^
  -v ${PWD}\metrics:/app/metrics ^
  cifar10-dvc dvc repro
```

Le Dockerfile inclut un entrypoint qui:
- bascule automatiquement en mode Service Account si `GDRIVE_CREDENTIALS_DATA` est présent;
- tente `dvc pull` au démarrage sauf si `SKIP_DVC_PULL=1`.

---

## 8. Bonnes pratiques & sécurité
- Ne pas committer `.dvc/config.local`, clés JSON ou tokens (déjà ignorés).
- Préférer `--env-file` ou le montage d’un fichier JSON au passage d’un long JSON en ligne de commande.
- Ajouter un `.dockerignore` pour réduire le contexte de build et éviter l’inclusion accidentelle de données/sources volumineuses: 
```
.dvc/config.local
.dvc/gdrive-creds.json
.dvc/cache/
.dvc/tmp/
.git/
__pycache__/
*.pyc
models/
metrics/
data/
```

---

## 9. Dépannage (FAQ)
- Le navigateur s’ouvre sur `localhost:8080` en conteneur: c’est un flux OAuth « web local ». Utiliser device code ou Service Account.
- Erreur d’accès GDrive (403/404): vérifiez le partage du dossier avec votre compte ou le SA.
- Montages Windows ne fonctionnent pas: partagez le disque dans Docker Desktop et utilisez `-v ${PWD}\\...`.
- OOM CPU: réduire `batch_size` et les époques dans `params.yaml`.

---

## 10. Arborescence du projet
```
image-classification-pipeline/
├── data/
│   ├── raw/cifar10/ (attendu)
│   └── processed/
├── src/
│   ├── prepare.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils/helpers.py
├── models/
├── metrics/
│   ├── train.json
│   ├── eval.json
│   └── plots/
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── requirements.txt
├── Dockerfile
├── README.md
└── report.md
```

---

## 11. Recettes rapides (cheat‑sheet)
- Installer (local): `pip install -r requirements.txt`
- Vérifier état vs remote: `dvc status -c`
- Exécuter une seule étape: `dvc repro prepare`
- Graphe des dépendances: `dvc dag`
- Build Docker: `docker build -t cifar10-dvc .`
- Run (local data): `SKIP_DVC_PULL=1` + monter `data/` et lancer `dvc repro`
- Run (Service Account): `--env-file env.list` + `dvc pull` puis `dvc repro`

---

## 12. Limites & pistes d’amélioration
- Pas de workflow CI/CD fourni par défaut (intégration possible avec GitHub Actions + Service Account).
- Téléchargement automatique du dataset non inclus (à ajouter si besoin).
- Optimisations CPU supplémentaires possibles (quantization, early‑stopping, etc.).

---

## 13. Références
- DVC: https://dvc.org/
- PyDrive2 (GDrive backend pour DVC): https://github.com/iterative/PyDrive2
- Docker: https://docs.docker.com/
- CIFAR‑10: https://www.cs.toronto.edu/~kriz/cifar.html


---

## Annexe – Guide A→Z (exécution pas à pas, sans aide externe)

Ce guide unique vous mène du clone du dépôt jusqu’aux métriques et modèles, en couvrant tous les cas d’usage (local, DVC + Google Drive, Docker, Service Account). Suivez-le dans l’ordre.

### A. Prérequis (Windows, CPU‑only)
1) Installer:
   - Git: https://git-scm.com/downloads
   - Python 3.10 (64‑bit): https://www.python.org/downloads/
   - Docker Desktop (activer le partage du disque D: si votre repo est sur D:)
2) Vérifier Docker Desktop → Settings → Resources → File Sharing → cocher le disque où se trouve le dépôt.
3) Cloner le dépôt (ou se placer dans son dossier si déjà cloné):
```
cd D:\Yassine\SDIA\S3\DevOps_MLOps\image-classification-pipeline
```
4) (Facultatif) Créer un environnement virtuel Python:
```
python -m venv venv
venv\Scripts\Activate.ps1
```
5) Installer les dépendances:
```
pip install -r requirements.txt
```

### B. Choisir votre source de données (une seule option)
- Option B1 — Données locales déjà présentes (plus simple):
  - Placez le dataset CIFAR‑10 sous `data/raw/cifar10/` avec la structure:
  ```
  data/raw/cifar10/
    train/<classe_0>/ ... <classe_9>/
    test/<classe_0>/  ... <classe_9>/
  ```
- Option B2 — Tirer les données depuis le remote DVC (Google Drive) avec OAuth sur l’hôte:
  1) S’assurer que `.dvc/config` pointe vers votre dossier GDrive (c’est déjà le cas).
  2) Lancer l’auth une fois sur l’hôte (navigateur):
  ```
  $env:DVC_NO_BROWSER = "0"
  dvc pull
  ```
  3) Les crédentials seront enregistrés en local (réutilisables en Docker via montage de `.dvc/` si besoin).
- Option B3 — Tirer les données via Service Account (non‑interactif, recommandé en CI/headless):
  1) Avoir un fichier `service-account.json` (type `service_account`).
  2) Partager le dossier GDrive du remote avec l’email du SA (Éditeur).
  3) Créer un fichier `env.list` à la racine contenant une ligne:
  ```
  GDRIVE_CREDENTIALS_DATA={...JSON sur une seule ligne...}
  ```

### C. Exécution locale (hors Docker)
1) Si Option B1 (local) ou B2/B3 (après pull), lancer le pipeline complet:
```
dvc repro
```
2) Ce que vous verrez/obtiendrez:
   - Manifests: `data/processed/*.json`
   - Modèles: `models/` (ex: `cnn.keras`)
   - Métriques: `metrics/train.json`, `metrics/eval.json`
   - Graphiques: `metrics/plots/*.png`
3) Rejouer après modification de `params.yaml`:
```
dvc repro
```
4) Exécuter un seul stage (ex: préparation):
```
dvc repro prepare
```
5) Outils utiles:
```
dvc status -c    # Compare cache local vs remote
dvc dag          # Affiche le graphe du pipeline
```

### D. Exécution avec Docker (CPU‑only)
1) Construire l’image:
```
docker build -t cifar10-dvc .
```
2) Cas D1 — Utiliser les données locales (aucune auth requise):
```
$env:SKIP_DVC_PULL = "1"
docker run --rm -it ^
  -e SKIP_DVC_PULL=$env:SKIP_DVC_PULL ^
  -v ${PWD}\data:/app/data ^
  -v ${PWD}\models:/app/models ^
  -v ${PWD}\metrics:/app/metrics ^
  cifar10-dvc dvc repro
```
3) Cas D2 — Réutiliser l’auth OAuth faite sur l’hôte:
```
# Après avoir fait `dvc pull` sur l’hôte (section B2), montez le dossier .dvc
$env:SKIP_DVC_PULL = "1"
docker run --rm -it ^
  -e SKIP_DVC_PULL=$env:SKIP_DVC_PULL ^
  -v ${PWD}\.dvc:/app/.dvc ^
  -v ${PWD}\models:/app/models ^
  -v ${PWD}\metrics:/app/metrics ^
  cifar10-dvc dvc repro
```
4) Cas D3 — Service Account (non‑interactif):
```
# Assurez-vous d’avoir `env.list` (voir section B3)
docker run --rm -it --env-file env.list cifar10-dvc dvc pull

docker run --rm -it ^
  --env-file env.list ^
  -v ${PWD}\models:/app/models ^
  -v ${PWD}\metrics:/app/metrics ^
  cifar10-dvc dvc repro
```
5) (Optionnel) Accélérer les exécutions répétées en persistant le cache DVC:
```
mkdir ${PWD}\_dvc_cache 2>$null

docker run --rm -it ^
  -v ${PWD}\_dvc_cache:/root/.cache ^
  cifar10-dvc dvc pull
```

### E. Paramétrage & re‑lancements
1) Ouvrir `params.yaml` et modifier:
   - `model_choice` (`cnn`|`mobilenetv2`), `epochs_cnn`, `epochs_tl`, `batch_size`, `image_size`, etc.
2) Rejouer uniquement ce qui a changé:
```
dvc repro
```
3) Exécuter une étape précise:
```
dvc repro train
```

### F. Vérifier les résultats
- Modèles: `models/` (poids entraînés)
- Métriques:
```
# PowerShell
type .\metrics\train.json
type .\metrics\eval.json
```
- Graphiques: `metrics/plots/*.png` (matrice de confusion, erreurs typiques)

### G. Utiliser le remote Google Drive
- Voir l’état vs remote:
```
dvc status -c
```
- Télécharger (pull) ou envoyer (push) les objets:
```
dvc pull
dvc push
```
- Notes:
  - `.dvc/config` contient l’URL du remote (OK à versionner).
  - `.dvc/config.local` contient vos secrets (NE PAS versionner; déjà ignoré).

### H. Sécurité & hygiène
- Ne jamais committer des secrets (`.dvc/config.local`, `service-account.json`).
- Préférer les fichiers `--env-file` pour injecter des JSON de clés en Docker.
- Utiliser `.dockerignore` (fourni) pour réduire le contexte de build et éviter d’embarquer données/artefacts.
- Si une clé fuite, la révoquer/rotater depuis Google Cloud Console.

### I. Dépannage rapide
- OAuth ouvre une page `localhost:8080` en conteneur → privilégier Service Account (D3) ou faites l’OAuth sur l’hôte puis montez `.dvc/` (D2).
- "Cette application est bloquée" (Google) → utiliser Service Account (D3).
- 403/404 Google Drive → vérifier partage du dossier GDrive du remote avec votre compte ou le SA (Éditeur).
- Montages Windows ne fonctionnent pas → partager le disque dans Docker Desktop; utiliser des chemins `-v ${PWD}\...`.
- Mémoire CPU saturée → réduire `batch_size`, `epochs_*` dans `params.yaml`.

### J. Check‑list finale (résumé 60 secondes)
1) Installer dépendances (`pip install -r requirements.txt`).
2) Choisir la source de données: B1 (local) OU B2 (pull OAuth sur hôte) OU B3 (Service Account).
3) Lancer localement: `dvc repro` (ou passer par Docker D1/D2/D3).
4) Vérifier: `models/`, `metrics/train.json`, `metrics/eval.json`, `metrics/plots/*`.
5) Synchroniser (optionnel): `dvc push`.
