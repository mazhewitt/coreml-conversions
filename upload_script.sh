#!/bin/bash

echo "Please run: huggingface-cli login"
echo "Then run this script to upload the models"
echo ""

# Upload files individually to avoid large file issues
cd flan-t5-base-coreml-repo

echo "Uploading README.md..."
huggingface-cli upload mazhewitt/flan-t5-base-coreml README.md --repo-type model

echo "Uploading config.json..."
huggingface-cli upload mazhewitt/flan-t5-base-coreml config.json --repo-type model

echo "Uploading encoder model..."
huggingface-cli upload mazhewitt/flan-t5-base-coreml flan_t5_base_encoder.mlpackage --repo-type model

echo "Uploading decoder model..."
huggingface-cli upload mazhewitt/flan-t5-base-coreml flan_t5_base_decoder.mlpackage --repo-type model

echo "Upload complete!"
echo "Your models are now available at: https://huggingface.co/mazhewitt/flan-t5-base-coreml"