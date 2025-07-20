#!/usr/bin/env python3
"""
Test the downloaded CoreML models to verify they work correctly.
"""

import coremltools as ct
import numpy as np
import os

def test_downloaded_models():
    """Test the downloaded CoreML models."""
    
    encoder_path = "downloaded_models/flan_t5_base_encoder.mlpackage"
    decoder_path = "downloaded_models/flan_t5_base_decoder.mlpackage"
    
    print("Testing downloaded FLAN-T5 CoreML models...")
    print("=" * 50)
    
    # Check if files exist
    if not os.path.exists(encoder_path):
        print(f"❌ Encoder model not found at: {encoder_path}")
        return False
    
    if not os.path.exists(decoder_path):
        print(f"❌ Decoder model not found at: {decoder_path}")
        return False
    
    print(f"✅ Encoder model found: {encoder_path}")
    print(f"✅ Decoder model found: {decoder_path}")
    
    try:
        # Load models
        print("\nLoading models...")
        encoder = ct.models.MLModel(encoder_path)
        decoder = ct.models.MLModel(decoder_path)
        print("✅ Models loaded successfully")
        
        # Print model specifications
        print("\n📋 ENCODER SPECIFICATIONS:")
        print(f"   Inputs: {list(encoder.input_description.keys())}")
        print(f"   Outputs: {list(encoder.output_description.keys())}")
        
        print("\n📋 DECODER SPECIFICATIONS:")
        print(f"   Inputs: {list(decoder.input_description.keys())}")
        print(f"   Outputs: {list(decoder.output_description.keys())}")
        
        # Test encoder
        print("\n🧪 Testing encoder...")
        max_seq_length = 512
        vocab_size = 32128
        
        # Create test inputs
        input_ids = np.random.randint(0, vocab_size, (1, max_seq_length)).astype(np.int32)
        attention_mask = np.ones((1, max_seq_length), dtype=np.int32)
        
        encoder_output = encoder.predict({
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })
        
        hidden_states = encoder_output["hidden_states"]
        print(f"✅ Encoder output shape: {hidden_states.shape}")
        print(f"   Expected: (1, 512, 768)")
        
        # Test decoder
        print("\n🧪 Testing decoder...")
        decoder_input_ids = np.random.randint(0, vocab_size, (1, max_seq_length)).astype(np.int32)
        decoder_attention_mask = np.ones((1, max_seq_length), dtype=np.int32)
        encoder_attention_mask = np.ones((1, max_seq_length), dtype=np.int32)
        
        decoder_output = decoder.predict({
            "decoder_input_ids": decoder_input_ids,
            "encoder_hidden_states": hidden_states,
            "decoder_attention_mask": decoder_attention_mask,
            "encoder_attention_mask": encoder_attention_mask
        })
        
        logits = decoder_output["logits"]
        print(f"✅ Decoder output shape: {logits.shape}")
        print(f"   Expected: (1, 512, 32128)")
        
        # Verify shapes
        if hidden_states.shape == (1, 512, 768) and logits.shape == (1, 512, 32128):
            print("\n🎉 SUCCESS: All tests passed!")
            print("   • Models downloaded correctly")
            print("   • Models load without errors")
            print("   • Input/output shapes are correct")
            print("   • End-to-end inference works")
            return True
        else:
            print("\n❌ FAILURE: Output shapes are incorrect")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_downloaded_models()
    if success:
        print("\n✅ Models are ready for use in your projects!")
        print("\nUsage in other projects:")
        print("huggingface-cli download mazhewitt/flan-t5-base-coreml --local-dir ./models")
    else:
        print("\n❌ Testing failed. Please check the models.")