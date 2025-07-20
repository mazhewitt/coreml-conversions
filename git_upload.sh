#!/bin/bash

# Alternative upload method using git
cd flan-t5-base-coreml-repo

# Initialize git repo
git init
git lfs install

# Track large files with LFS
git lfs track "*.mlpackage/**"
git add .gitattributes

# Add all files
git add .

# Commit
git commit -m "Add FLAN-T5 base CoreML models"

# Add remote (you may need to set up authentication)
git remote add origin https://huggingface.co/mazhewitt/flan-t5-base-coreml

# Push
git push -u origin main