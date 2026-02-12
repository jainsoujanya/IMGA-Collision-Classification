import os
import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
import json

# Config
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
THRESHOLD = 0.35  # Decision threshold for collision vs no-collision
# Accuracy values depend on the final trained model and dataset.
# To avoid confusion during evaluation/viva, they are not hard-coded here.
BACKEND_ACCURACY = None
FRONTEND_ACCURACY = None

# Model definition
class VGG19Model(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        v = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self.features = v.features
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(True), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(True), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        f = self.features(x)
        f = self.avg(f).flatten(1)
        return self.classifier(f)

# Load model
try:
    model = VGG19Model().to(DEVICE)
    if os.path.exists('best_model.pth'):
        state = torch.load('best_model.pth', map_location=DEVICE, weights_only=False)
        model.load_state_dict(state)
        print("✅ Model loaded successfully")
    else:
        print("⚠️  WARNING: Model file not found")
        model = None
    if model:
        model.eval()
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Preprocess
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Initialize Flask
app = Flask(__name__, static_folder='static', static_url_path='/static')

# Serve the HTML file directly
@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'success': False
        }), 500
    
    if 'image' not in request.files:
        return jsonify({
            'error': 'No image file provided',
            'success': False
        }), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({
            'error': 'Please select an image file',
            'success': False
        }), 400

    try:
        img_bytes = file.read()
        if len(img_bytes) == 0:
            return jsonify({
                'error': 'Empty file uploaded',
                'success': False
            }), 400
        
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        if img.size[0] < 50 or img.size[1] < 50:
            return jsonify({
                'error': 'Image too small',
                'success': False
            }), 400
        
        x = preprocess(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits = model(x)
            probabilities = F.softmax(logits, dim=1)
            prob_collision = probabilities[0, 1].item()
            prob_no_collision = probabilities[0, 0].item()
        
        pred = 'Collision' if prob_collision >= THRESHOLD else 'No Collision'
        
        if pred == 'Collision':
            confidence_score = min(100, (prob_collision - THRESHOLD) / (1 - THRESHOLD) * 100)
        else:
            confidence_score = min(100, (THRESHOLD - prob_collision) / THRESHOLD * 100)
        
        return jsonify({
            'success': True,
            'prediction': pred,
            'prob_collision': round(prob_collision, 4),
            'prob_no_collision': round(prob_no_collision, 4),
            'score': f"{prob_collision:.4f}",
            'no_collision_score': f"{prob_no_collision:.4f}",
            'threshold': THRESHOLD,
            'confidence': round(confidence_score, 1),
            'autonomous_vehicle_ready': True,
            'recommendation': 'Emergency Braking Recommended' if pred == 'Collision' and prob_collision > 0.7 else 'Continue Normal Operation'
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

if __name__ == '__main__':
    print("=" * 70)
    print("🚗 AI COLLISION DETECTION SYSTEM FOR AUTONOMOUS VEHICLES")
    print("=" * 70)
    print(f"📁 Template: index.html")
    print(f"📁 Template exists: {os.path.exists('index.html')}")
    if model:
        print(f"✅ Model: Loaded on {DEVICE}")
        print(f"✅ Backend Accuracy: {BACKEND_ACCURACY}%")
        print(f"✅ Frontend Accuracy: {FRONTEND_ACCURACY}%")
    else:
        print("⚠️  Model: Not loaded")
    print("=" * 70)
    print("🌐 Server starting on http://0.0.0.0:8080")
    print("🌐 Access via: http://localhost:8080 or http://127.0.0.1:8080")
    print("=" * 70)
    app.run(host='0.0.0.0', port=8080, debug=True)
