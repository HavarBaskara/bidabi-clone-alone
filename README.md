# bidabi-clone-adapt-create
# BIDABI : Clone → Adapt → Create

Dépôt pédagogique du cours **Big Data and Business Intelligence (BIDABI)**.  
Ce projet a pour objectif d’initier les étudiants au travail avec du code open‑source, à l’adaptation de projets existants et à la création de leur propre jeu de données d’images.

## 🎯 Objectif du dépôt
Ce dépôt sert de **plateforme d’apprentissage** où les étudiants réalisent un cycle complet de travail en data et en machine learning :

- cloner un projet open‑source depuis GitHub
- analyser sa structure, ses dépendances et son fonctionnement
- adapter le code à un nouveau contexte
- créer un jeu de données d’images personnalisé
- intégrer ce jeu de données dans un pipeline ML existant

L’objectif est de reproduire des situations réelles rencontrées par les ingénieurs data et ML lorsqu’ils doivent réutiliser et modifier du code provenant d’autres développeurs.

## 🎓 Public visé
Ce projet est destiné aux étudiants du cours **BIDABI**, notamment ceux qui s’intéressent à :

- l’apprentissage automatique
- l’ingénierie des données
- la reproductibilité des expériences
- l’utilisation de GitHub et des projets open‑source

## 🧩 Contenu du dépôt
Le dépôt inclura :

- des exemples de code à analyser et adapter
- un modèle de structure pour le jeu de données
- des consignes pour les travaux pratiques
- des instructions pour exécuter et modifier le projet

## 🛠️ Compétences développées
Les étudiants apprendront à :

- lire et comprendre du code écrit par d’autres
- manipuler des dépôts GitHub
- concevoir et organiser un jeu de données d’images
- intégrer des données dans un pipeline ML
- documenter leur travail de manière claire et reproductible

## 📄 Licence et usage
Ce dépôt est destiné **exclusivement à des fins pédagogiques** dans le cadre du cours BIDABI.  
Le code et les ressources peuvent être simplifiés ou modifiés pour faciliter l’apprentissage.

## Description
Projet de classification d’images de produits alimentaires avec Deep Learning.

Le pipeline comprend :
•⁠  ⁠collecte des données (scraping)
•⁠  ⁠stockage dataset RAW
•⁠  ⁠séparation train / val / test
•⁠  ⁠entraînement d’un modèle ResNet-18
•⁠  ⁠sauvegarde du meilleur modèle
•⁠  ⁠versioning avec Git et DVC

## Dataset utilisé

Catégories :
•⁠  ⁠bread
•⁠  ⁠milk
•⁠  ⁠sugar
•⁠  ⁠champagnes

Structure :

data/raw/images/<categorie>/

Nombre total d’images : (mettre ton nombre)

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
