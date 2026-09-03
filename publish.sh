#!/usr/bin/env bash
# Push this folder to an existing, empty GitHub repository.
#
#   ./publish.sh git@github.com:<user>/<repo>.git      (SSH key)
#   ./publish.sh https://github.com/<user>/<repo>.git  (asks for a token)
set -euo pipefail

REMOTE="${1:?Usage: ./publish.sh <repository-url>}"
BRANCH=main

cd "$(dirname "$0")"

if [ ! -d .git ]; then
    git init -q
    git branch -M "$BRANCH"
fi

git add -A
git diff --cached --quiet || \
    git commit -qm "Spectro-Kinetic Transformer: model, preprocessing and training loop"

if git remote get-url origin >/dev/null 2>&1; then
    git remote set-url origin "$REMOTE"
else
    git remote add origin "$REMOTE"
fi

# If the repository was initialised with a README or a licence on GitHub,
# bring those commits in first; a plain push would be rejected.
git fetch -q origin || true
if git rev-parse --verify --quiet "origin/$BRANCH" >/dev/null; then
    git pull --rebase origin "$BRANCH"
fi

git push -u origin "$BRANCH"
