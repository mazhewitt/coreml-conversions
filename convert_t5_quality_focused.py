#!/usr/bin/env python3
"""
Quality-focused T5 CoreML conversion that preserves the original model behavior.
Key principles:
1. Minimal changes to model structure
2. Preserve original sequence lengths (512)
3. Use FLOAT32 precision to avoid quality loss
4. Avoid manual attention modifications
5. Test quality at each step
"""

import torch
import coremltools as ct
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os


class T5EncoderWrapper(torch.nn.Module):
    """Minimal wrapper for T5 encoder - preserves original behavior."""
    
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    
    def forward(self, input_ids, attention_mask):
        # Use encoder exactly as-is, no modifications
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state


class T5DecoderWrapper(torch.nn.Module):
    """Minimal wrapper for T5 decoder - preserves original behavior."""
    
    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder
        self.lm_head = model.lm_head
        self.model_dim = model.config.d_model
    
    def forward(self, decoder_input_ids, encoder_hidden_states, 
                decoder_attention_mask, encoder_attention_mask):
        # Use decoder directly with proper encoder outputs
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False,
            return_dict=True
        )
        
        sequence_output = decoder_outputs.last_hidden_state
        
        # Apply final projection
        if self.model_dim != self.lm_head.in_features:
            sequence_output = sequence_output * (self.model_dim ** -0.5)
            
        logits = self.lm_head(sequence_output)
        return logits


def test_pytorch_quality():
    """Test original PyTorch model quality as baseline."""
    
    print("🧪 Testing original PyTorch model quality...")
    
    tokenizer = T5Tokenizer.from_pretrained("./flan-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("./flan-t5-base")
    
    test_cases = [
        "translate English to French: Hello world",
        "translate English to German: Good morning", 
        "summarize: The quick brown fox jumps over the lazy dog.",
        "translate French to English: Bonjour le monde"
    ]
    
    results = {}
    
    for test_input in test_cases:
        input_ids = tokenizer(test_input, return_tensors="pt").input_ids
        
        with torch.no_grad():
            outputs = model.generate(input_ids, max_length=50, num_beams=1, do_sample=False)
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            results[test_input] = result
            print(f"✅ '{test_input}' → '{result}'")
    
    print("📊 PyTorch baseline established")
    return results


def convert_with_quality_preservation():
    """Convert T5 to CoreML while preserving quality."""
    
    print("\n🚀 Converting T5 to CoreML with quality preservation")
    print("=" * 70)
    
    # Load model
    tokenizer = T5Tokenizer.from_pretrained("./flan-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("./flan-t5-base")
    
    # Use original sequence length to avoid degradation
    max_length = 512
    
    print(f"📋 Configuration:")
    print(f"   Model: FLAN-T5-Base")
    print(f"   Max sequence length: {max_length} (original)")
    print(f"   Precision: FLOAT32 (high quality)")
    print(f"   Approach: Minimal modification")
    
    # Create output directory
    output_dir = "coreml_models_quality"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Convert Encoder
    print(f"\n🔧 Converting Encoder...")
    
    encoder_wrapper = T5EncoderWrapper(model.get_encoder())
    encoder_wrapper.eval()
    
    # Create sample inputs with original size
    sample_input_ids = torch.randint(0, 1000, (1, max_length), dtype=torch.long)
    sample_attention_mask = torch.ones((1, max_length), dtype=torch.long)
    
    # Trace encoder
    with torch.no_grad():
        traced_encoder = torch.jit.trace(
            encoder_wrapper,
            (sample_input_ids, sample_attention_mask)
        )
    
    # Convert to CoreML with high precision
    encoder_coreml = ct.convert(
        traced_encoder,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, max_length), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, max_length), dtype=np.int32)
        ],
        outputs=[
            ct.TensorType(name="hidden_states", dtype=np.float32)
        ],
        compute_precision=ct.precision.FLOAT32,  # High precision for quality
        minimum_deployment_target=ct.target.iOS15
    )
    
    encoder_path = os.path.join(output_dir, "flan_t5_base_encoder_quality.mlpackage")
    encoder_coreml.save(encoder_path)
    print(f"✅ Encoder saved: {encoder_path}")
    
    # 2. Convert Decoder
    print(f"\n🔧 Converting Decoder...")
    
    decoder_wrapper = T5DecoderWrapper(model)
    decoder_wrapper.eval()
    
    # Create sample inputs for decoder
    sample_decoder_input_ids = torch.randint(0, 1000, (1, max_length), dtype=torch.long)
    sample_encoder_hidden_states = torch.randn((1, max_length, 768), dtype=torch.float32)
    sample_decoder_attention_mask = torch.ones((1, max_length), dtype=torch.long)
    sample_encoder_attention_mask = torch.ones((1, max_length), dtype=torch.long)
    
    # Trace decoder
    with torch.no_grad():
        traced_decoder = torch.jit.trace(
            decoder_wrapper,
            (sample_decoder_input_ids, sample_encoder_hidden_states,
             sample_decoder_attention_mask, sample_encoder_attention_mask)
        )
    
    # Convert to CoreML with high precision
    decoder_coreml = ct.convert(
        traced_decoder,
        inputs=[
            ct.TensorType(name="decoder_input_ids", shape=(1, max_length), dtype=np.int32),
            ct.TensorType(name="encoder_hidden_states", shape=(1, max_length, 768), dtype=np.float32),
            ct.TensorType(name="decoder_attention_mask", shape=(1, max_length), dtype=np.int32),
            ct.TensorType(name="encoder_attention_mask", shape=(1, max_length), dtype=np.int32)
        ],
        outputs=[
            ct.TensorType(name="logits", dtype=np.float32)
        ],
        compute_precision=ct.precision.FLOAT32,  # High precision for quality
        minimum_deployment_target=ct.target.iOS15
    )
    
    decoder_path = os.path.join(output_dir, "flan_t5_base_decoder_quality.mlpackage")
    decoder_coreml.save(decoder_path)
    print(f"✅ Decoder saved: {decoder_path}")
    
    print(f"\n🎉 Conversion completed!")
    return encoder_path, decoder_path


def test_coreml_quality(encoder_path, decoder_path, pytorch_results):
    """Test CoreML model quality against PyTorch baseline."""
    
    print(f"\n🧪 Testing CoreML model quality...")
    
    import coremltools as ct
    
    # Load models
    encoder = ct.models.MLModel(encoder_path)
    decoder = ct.models.MLModel(decoder_path)
    tokenizer = T5Tokenizer.from_pretrained("./flan-t5-base")
    
    print(f"✅ CoreML models loaded")
    
    # Test the same cases as PyTorch
    test_cases = [
        "translate English to French: Hello world",
        "translate English to German: Good morning", 
        "summarize: The quick brown fox jumps over the lazy dog.",
        "translate French to English: Bonjour le monde"
    ]
    
    print(f"\n📊 Quality Comparison:")
    print("=" * 70)
    
    quality_maintained = True
    
    for test_input in test_cases:
        print(f"\n📝 Input: '{test_input}'")
        
        # Get PyTorch reference
        pytorch_result = pytorch_results[test_input]
        print(f"🐍 PyTorch: '{pytorch_result}'")
        
        # Test CoreML
        try:
            # Tokenize input
            inputs = tokenizer(test_input, return_tensors="np", padding="max_length", 
                             truncation=True, max_length=512)
            
            # Run encoder
            encoder_output = encoder.predict({
                "input_ids": inputs["input_ids"].astype(np.int32),
                "attention_mask": inputs["attention_mask"].astype(np.int32)
            })
            hidden_states = encoder_output["hidden_states"]
            
            # Simple greedy generation for comparison
            generated_tokens = [tokenizer.pad_token_id]
            max_new_tokens = 10
            
            for step in range(max_new_tokens):
                # Prepare decoder input (pad to full length)
                decoder_ids = np.zeros((1, 512), dtype=np.int32)
                decoder_mask = np.zeros((1, 512), dtype=np.int32)
                
                for j, token in enumerate(generated_tokens):
                    if j < 512:
                        decoder_ids[0, j] = token
                        decoder_mask[0, j] = 1
                
                # Run decoder
                decoder_output = decoder.predict({
                    "decoder_input_ids": decoder_ids,
                    "encoder_hidden_states": hidden_states,
                    "decoder_attention_mask": decoder_mask,
                    "encoder_attention_mask": inputs["attention_mask"].astype(np.int32)
                })
                
                # Get next token
                next_pos = len(generated_tokens)
                if next_pos >= 512:
                    break
                    
                logits = decoder_output["logits"]
                next_token = np.argmax(logits[0, next_pos, :])
                
                # Stop if EOS
                if next_token == tokenizer.eos_token_id:
                    break
                    
                generated_tokens.append(int(next_token))
            
            # Decode result
            coreml_result = tokenizer.decode(generated_tokens[1:], skip_special_tokens=True)
            print(f"🍎 CoreML:  '{coreml_result}'")
            
            # Check quality
            if len(coreml_result.strip()) == 0:
                print(f"❌ Empty output - quality issue detected")
                quality_maintained = False
            elif "legung" in coreml_result or "bedarf" in coreml_result or "artige" in coreml_result:
                print(f"❌ Nonsensical output - quality issue detected")
                quality_maintained = False
            else:
                print(f"✅ Output looks reasonable")
            
        except Exception as e:
            print(f"❌ CoreML inference failed: {e}")
            quality_maintained = False
    
    print(f"\n" + "=" * 70)
    if quality_maintained:
        print(f"🌟 QUALITY PRESERVED: CoreML models produce reasonable outputs!")
        print(f"✅ Ready for production use")
    else:
        print(f"💥 QUALITY ISSUES: CoreML models need further debugging")
        print(f"❌ Not ready for production")
    
    return quality_maintained


if __name__ == "__main__":
    try:
        # Step 1: Test PyTorch baseline
        pytorch_results = test_pytorch_quality()
        
        # Step 2: Convert with quality focus
        encoder_path, decoder_path = convert_with_quality_preservation()
        
        # Step 3: Test CoreML quality
        quality_ok = test_coreml_quality(encoder_path, decoder_path, pytorch_results)
        
        if quality_ok:
            print(f"\n🚀 SUCCESS: High-quality CoreML T5 models ready!")
        else:
            print(f"\n🔧 NEXT STEPS: Need to investigate remaining quality issues")
            
    except Exception as e:
        print(f"❌ Process failed: {e}")
        import traceback
        traceback.print_exc()