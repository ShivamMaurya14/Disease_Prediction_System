#!/bin/bash
# Quick setup script for brain tumor model

echo "================================"
echo "Brain Tumor Model Setup"
echo "================================"
echo ""

# Create models directory if it doesn't exist
mkdir -p models

echo "📥 Downloading brain tumor model from Google Drive..."
echo ""

# Download using gdown
python -m gdown "https://drive.google.com/uc?id=12oBWm5zYq7az62TPq7w68iFz5IOTygrG" \
    -O models/brain_tumor_model.keras --quiet

# Check if download was successful
if [ -f "models/brain_tumor_model.keras" ]; then
    echo "✓ Model downloaded successfully"
    echo ""
    echo "📊 File information:"
    ls -lh models/brain_tumor_model.keras
    echo ""
    echo "✓ Setup complete!"
    echo ""
    echo "Next steps:"
    echo "1. Install dependencies: pip install -r requirements.txt"
    echo "2. Run the app: streamlit run app.py"
    echo "3. Navigate to 'Brain Tumor Detection' tab"
    exit 0
else
    echo "✗ Download failed"
    echo ""
    echo "Try manual download:"
    echo "Visit: https://drive.google.com/file/d/12oBWm5zYq7az62TPq7w68iFz5IOTygrG/view"
    echo "Save to: models/brain_tumor_model.keras"
    exit 1
fi
