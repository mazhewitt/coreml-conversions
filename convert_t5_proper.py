#!/usr/bin/env python3
"""
Proper T5 CoreML conversion using HuggingFace Optimum exporters.
This avoids manual tracing and preserves model quality.
"""

from optimum.exporters.coreml import export, T5CoreMLConfig
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from collections import OrderedDict
from optimum.exporters.coreml.config import InputDescription
import os

def convert_t5_to_coreml():
    """Convert T5 to CoreML using proper exporters."""
    
    print("🚀 Converting T5 to CoreML using HuggingFace Optimum")
    print("=" * 60)
    
    # Model configuration
    model_id = "google/flan-t5-base"
    max_length = 128
    
    print(f"📥 Loading model: {model_id}")
    tokenizer = T5TokenizerFast.from_pretrained(model_id)
    model = T5ForConditionalGeneration.from_pretrained(model_id)
    
    print(f"✅ Model loaded successfully")
    print(f"   Vocab size: {model.config.vocab_size}")
    print(f"   Hidden size: {model.config.d_model}")
    print(f"   Max length: {max_length}")
    
    # Create output directory
    output_dir = "coreml_models_proper"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Export Encoder
    print(f"\n🔧 Exporting Encoder...")
    
    class EncoderConfig(T5CoreMLConfig):
        @property
        def inputs(self) -> OrderedDict[str, InputDescription]:
            d = super().inputs
            # Set sequence length for encoder
            d["input_ids"].sequence_length = max_length
            if "attention_mask" in d:
                d["attention_mask"].sequence_length = max_length
            return d
    
    encoder_config = EncoderConfig(
        model.config, 
        task="text2text-generation", 
        seq2seq="encoder"
    )
    
    # Export encoder submodel
    encoder_mlpackage = export(
        tokenizer, 
        model.get_encoder(), 
        encoder_config
    )
    
    encoder_path = os.path.join(output_dir, "flan_t5_base_encoder_proper.mlpackage")
    encoder_mlpackage.save(encoder_path)
    print(f"✅ Encoder saved: {encoder_path}")
    
    # 2. Export Decoder
    print(f"\n🔧 Exporting Decoder...")
    
    class DecoderConfig(T5CoreMLConfig):
        @property
        def inputs(self) -> OrderedDict[str, InputDescription]:
            d = super().inputs
            # Set sequence length for decoder inputs
            if "decoder_input_ids" in d:
                d["decoder_input_ids"].sequence_length = max_length
            if "decoder_attention_mask" in d:
                d["decoder_attention_mask"].sequence_length = max_length
            if "encoder_hidden_states" in d:
                d["encoder_hidden_states"].sequence_length = max_length
            if "encoder_attention_mask" in d:
                d["encoder_attention_mask"].sequence_length = max_length
            return d
    
    decoder_config = DecoderConfig(
        model.config, 
        task="text2text-generation", 
        seq2seq="decoder"
    )
    
    # Export full model for decoder (includes cross-attention)
    decoder_mlpackage = export(
        tokenizer, 
        model, 
        decoder_config
    )
    
    decoder_path = os.path.join(output_dir, "flan_t5_base_decoder_proper.mlpackage")
    decoder_mlpackage.save(decoder_path)
    print(f"✅ Decoder saved: {decoder_path}")
    
    print(f"\n🎉 Conversion completed successfully!")
    print(f"📁 Models saved in: {output_dir}/")
    print(f"   Encoder: flan_t5_base_encoder_proper.mlpackage")
    print(f"   Decoder: flan_t5_base_decoder_proper.mlpackage")
    
    return encoder_path, decoder_path

def test_converted_models(encoder_path, decoder_path):
    """Quick test to verify the models work."""
    
    print(f"\n🧪 Testing converted models...")
    
    import coremltools as ct
    import numpy as np
    
    # Load models
    encoder = ct.models.MLModel(encoder_path)
    decoder = ct.models.MLModel(decoder_path)
    
    print(f"✅ Models loaded successfully")
    
    # Print model specs
    print(f"\n📋 Encoder specs:")
    for input_desc in encoder.input_description:
        print(f"   {input_desc.name}: {input_desc.type}")
    for output_desc in encoder.output_description:
        print(f"   → {output_desc.name}: {output_desc.type}")
    
    print(f"\n📋 Decoder specs:")
    for input_desc in decoder.input_description:
        print(f"   {input_desc.name}: {input_desc.type}")
    for output_desc in decoder.output_description:
        print(f"   → {output_desc.name}: {output_desc.type}")
    
    # Basic inference test
    print(f"\n🔍 Running basic inference test...")
    
    # Create sample inputs
    batch_size = 1
    seq_len = 128
    hidden_size = 768  # T5-base hidden size
    
    # Test encoder
    sample_input_ids = np.random.randint(0, 1000, (batch_size, seq_len), dtype=np.int32)
    sample_attention_mask = np.ones((batch_size, seq_len), dtype=np.int32)
    
    encoder_inputs = {
        "input_ids": sample_input_ids,
        "attention_mask": sample_attention_mask
    }
    
    try:
        encoder_output = encoder.predict(encoder_inputs)
        print(f"✅ Encoder inference successful")
        
        # Get encoder hidden states for decoder
        encoder_hidden_states = encoder_output["last_hidden_state"]
        print(f"   Encoder output shape: {encoder_hidden_states.shape}")
        
        # Test decoder
        sample_decoder_input_ids = np.random.randint(0, 1000, (batch_size, seq_len), dtype=np.int32)
        sample_decoder_attention_mask = np.ones((batch_size, seq_len), dtype=np.int32)
        
        decoder_inputs = {
            "decoder_input_ids": sample_decoder_input_ids,
            "decoder_attention_mask": sample_decoder_attention_mask,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": sample_attention_mask
        }
        
        decoder_output = decoder.predict(decoder_inputs)
        print(f"✅ Decoder inference successful")
        
        logits = decoder_output["logits"]
        print(f"   Decoder output shape: {logits.shape}")
        print(f"   Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        return False

if __name__ == "__main__":
    try:
        encoder_path, decoder_path = convert_t5_to_coreml()
        success = test_converted_models(encoder_path, decoder_path)
        
        if success:
            print(f"\n🌟 Models are ready for quality testing!")
        else:
            print(f"\n💥 Models need debugging")
            
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()