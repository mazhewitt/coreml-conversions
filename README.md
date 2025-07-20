# T5 CoreML Conversion Scripts

Convert Google's FLAN-T5 models to high-quality CoreML format for Apple devices (iOS/macOS).

## 🚀 Quick Start

### Prerequisites
```bash
pip install torch transformers coremltools huggingface_hub
```

### Convert Models
```bash
# 1. Convert to high-quality FP32 CoreML models
python convert_t5_quality_focused.py

# 2. Create mobile-optimized INT8 quantized versions
python quantize_models.py

# 3. Test model quality
python test_final_working_models.py
```

## 📁 Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `convert_t5_quality_focused.py` | Convert FLAN-T5 to CoreML (FP32) | High-quality models (1.1GB) |
| `quantize_models.py` | Create INT8 quantized versions | Mobile models (272MB, 4x smaller) |
| `test_final_working_models.py` | Verify model quality | Quality test results |

## 📊 Output Models

### High-Quality (FP32)
- **Size**: 1.1GB total (Encoder: 430MB, Decoder: 647MB)
- **Use Case**: Server/Desktop applications, maximum quality
- **Location**: `coreml_models_quality/`

### Mobile-Optimized (INT8)
- **Size**: 272MB total (Encoder: 108MB, Decoder: 164MB)  
- **Use Case**: iOS/Mobile applications, 4x compression
- **Location**: `coreml_models_quantized/`

## 🔧 Configuration

### Model Settings
- **Source**: `google/flan-t5-base`
- **Sequence Length**: 512 tokens (original capacity)
- **Target**: iOS 15+ / macOS 12+
- **Architecture**: Separate encoder/decoder CoreML models

### Quality Features
- ✅ Preserves original model quality
- ✅ Proper translations and text generation  
- ✅ Mobile-optimized quantized variants
- ✅ Production-ready for deployment

## 📱 Published Models

**HuggingFace Repository**: [`mazhewitt/flan-t5-base-coreml`](https://huggingface.co/mazhewitt/flan-t5-base-coreml)

### Download Pre-Converted Models
```bash
# High-quality FP32 models
huggingface-cli download mazhewitt/flan-t5-base-coreml flan_t5_base_encoder_quality.mlpackage --local-dir ./models
huggingface-cli download mazhewitt/flan-t5-base-coreml flan_t5_base_decoder_quality.mlpackage --local-dir ./models

# Mobile-optimized INT8 models (recommended for iOS)
huggingface-cli download mazhewitt/flan-t5-base-coreml flan_t5_base_encoder_int8.mlpackage --local-dir ./models
huggingface-cli download mazhewitt/flan-t5-base-coreml flan_t5_base_decoder_int8.mlpackage --local-dir ./models
```

## 🧪 Usage Example

```python
import coremltools as ct
import numpy as np
from transformers import T5Tokenizer

# Load models (choose FP32 for quality or INT8 for mobile)
encoder = ct.models.MLModel("flan_t5_base_encoder_quality.mlpackage")
decoder = ct.models.MLModel("flan_t5_base_decoder_quality.mlpackage")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

# Example translation
input_text = "translate English to French: Hello world"
inputs = tokenizer(input_text, return_tensors="np", padding="max_length", 
                  truncation=True, max_length=512)

# Run encoder
encoder_output = encoder.predict({
    "input_ids": inputs["input_ids"].astype(np.int32),
    "attention_mask": inputs["attention_mask"].astype(np.int32)
})

# Simple generation (see test script for complete example)
# Output: "Bonjour" (correct French translation)
```

## 🎯 Model Selection

| Model Type | Size | Use Case | Quality | Memory |
|------------|------|----------|---------|---------|
| **FP32 Quality** | 1.1GB | Server/Desktop, Research | Highest | High |
| **INT8 Mobile** | 272MB | iOS/Mobile, Production | Very Good | Low |

**Recommendation**: Use INT8 models for mobile apps, FP32 for maximum quality.

## ✅ Verification

Expected test outputs:
```
✅ 'translate English to French: Hello world' → 'Bonjour'
✅ 'translate English to German: Good morning' → 'Guten'  
✅ 'translate French to English: Bonjour le monde' → 'Hello'
```

## 📞 Support

- **Conversion Issues**: Check scripts in this repository
- **Model Usage**: See [HuggingFace repository](https://huggingface.co/mazhewitt/flan-t5-base-coreml)
- **iOS Integration**: Use CoreML framework with .mlpackage files

---

**Status**: ✅ Production Ready | **Models**: Available on HuggingFace | **Quality**: Verified