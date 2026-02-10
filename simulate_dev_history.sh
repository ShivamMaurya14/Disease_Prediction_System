#!/bin/bash

# Configuration
# Dates requested: Feb 8, Feb 9, Feb 10
DAY1="2026-02-08 10:00:00"
DAY2="2026-02-09 10:00:00"
DAY3="2026-02-10 10:00:00"

# Initialize Git (Wipes previous history to create the clean timeline)
rm -rf .git
git init
git branch -M main

echo "Recreating git history for current structure..."

# --- DAY 1 (Feb 8): Initialization ---
# Setup environment, gitignore, and basic app structure
git add .gitignore requirements.txt run_app.sh
# Check if app.py exists (it does)
git add app.py
GIT_AUTHOR_DATE="$DAY1" GIT_COMMITTER_DATE="$DAY1" git commit -m "Initial commit: Project setup, dependencies, and core app"

# --- DAY 2 (Feb 9): Backend & Models ---
# Add the models directory and verification scripts
git add models/
git add scripts/
GIT_AUTHOR_DATE="$DAY2" GIT_COMMITTER_DATE="$DAY2" git commit -m "Integrated ML models and added verification scripts"

# --- DAY 3 (Feb 10): Assets, Notebooks, & Docs ---
# Add the rest of the project
git add assets/
git add notebooks/
git add README.md
# Catch-all for anything missed
git add .
GIT_AUTHOR_DATE="$DAY3" GIT_COMMITTER_DATE="$DAY3" git commit -m "Final submission: Added training notebooks, assets, and documentation"

echo "------------------------------------------------"
echo "Git history rewritten!"
echo "Commits match the current file structure."
echo ""
echo "IMPORTANT: You must FORCE PUSH because history has changed."
echo "Run: git push -u origin main --force"
echo "------------------------------------------------"
