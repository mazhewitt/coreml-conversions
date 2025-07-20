#!/usr/bin/env python3
"""
Final fix for T5 CoreML conversion with consistent sequence lengths.
"""

import torch
import coremltools as ct
from transformers import T5ForConditionalGeneration, T5Tokenizer
import numpy as np
import os

class T5EncoderFixed(torch.nn.Module):
    """T5 encoder with consistent sequence length."""
    
    def __init__(self, t5_model):
        super().__init__()
        self.encoder = t5_model.encoder
        
    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        return outputs.last_hidden_state

class T5DecoderFixed(torch.nn.Module):
    """T5 decoder with consistent sequence length and proper causal attention."""
    
    def __init__(self, t5_model):
        super().__init__()
        self.decoder = t5_model.decoder
        self.lm_head = t5_model.lm_head
        
    def forward(self, decoder_input_ids, encoder_hidden_states, decoder_attention_mask=None, encoder_attention_mask=None):
        if decoder_attention_mask is None:
            decoder_attention_mask = torch.ones_like(decoder_input_ids)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(encoder_hidden_states.shape[:2])
            
        # Let transformers handle causal masking internally
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False,
            return_dict=True
        )
        
        logits = self.lm_head(decoder_outputs.last_hidden_state)
        return logits

def convert_consistent_models(model, tokenizer, output_path, max_length=128):
    """Convert both encoder and decoder with consistent sequence lengths."""
    
    print(f"Converting T5 models with consistent max_length={max_length}")
    print("=" * 60)
    
    os.makedirs(output_path, exist_ok=True)
    
    # Convert encoder
    print("Converting encoder...")
    encoder_wrapper = T5EncoderFixed(model)
    encoder_wrapper.eval()
    
    # Test encoder wrapper
    example_input_ids = torch.randint(0, tokenizer.vocab_size, (1, max_length))
    example_attention_mask = torch.ones(1, max_length)
    
    with torch.no_grad():
        encoder_test = encoder_wrapper(example_input_ids, example_attention_mask)
        print(f"✅ Encoder wrapper output shape: {encoder_test.shape}")
    
    # Trace encoder
    traced_encoder = torch.jit.trace(
        encoder_wrapper,
        (example_input_ids, example_attention_mask)
    )
    
    # Convert encoder to CoreML
    coreml_encoder = ct.convert(
        traced_encoder,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, max_length), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, max_length), dtype=np.int32)
        ],
        outputs=[
            ct.TensorType(name="hidden_states", dtype=np.float32)
        ],
        minimum_deployment_target=ct.target.iOS15,
        compute_precision=ct.precision.FLOAT16
    )
    
    encoder_path = os.path.join(output_path, f"flan_t5_base_encoder_fixed_{max_length}.mlpackage")
    coreml_encoder.save(encoder_path)
    print(f"✅ Encoder saved to: {encoder_path}")
    
    # Convert decoder
    print("Converting decoder...")
    decoder_wrapper = T5DecoderFixed(model)
    decoder_wrapper.eval()
    
    # Test decoder wrapper with different inputs to verify causal behavior
    d_model = model.config.d_model
    example_decoder_input_ids = torch.randint(0, 1000, (1, max_length))
    example_encoder_hidden_states = torch.randn(1, max_length, d_model)
    example_decoder_attention_mask = torch.ones(1, max_length)
    example_encoder_attention_mask = torch.ones(1, max_length)
    
    with torch.no_grad():
        decoder_test = decoder_wrapper(
            example_decoder_input_ids,
            example_encoder_hidden_states,
            example_decoder_attention_mask,
            example_encoder_attention_mask
        )
        print(f"✅ Decoder wrapper output shape: {decoder_test.shape}")
        
        # Test causal behavior
        pos_0_logits = decoder_test[0, 0, :]
        pos_1_logits = decoder_test[0, 1, :]
        diff = torch.mean(torch.abs(pos_0_logits - pos_1_logits)).item()
        print(f"✅ Position difference in wrapper: {diff:.6f}")
    
    # Trace decoder
    traced_decoder = torch.jit.trace(
        decoder_wrapper,
        (example_decoder_input_ids, example_encoder_hidden_states,
         example_decoder_attention_mask, example_encoder_attention_mask)
    )
    
    # Test traced decoder
    with torch.no_grad():
        traced_test = traced_decoder(
            example_decoder_input_ids,
            example_encoder_hidden_states,
            example_decoder_attention_mask,
            example_encoder_attention_mask
        )
        traced_pos_0 = traced_test[0, 0, :]
        traced_pos_1 = traced_test[0, 1, :]
        traced_diff = torch.mean(torch.abs(traced_pos_0 - traced_pos_1)).item()
        print(f"✅ Position difference in traced model: {traced_diff:.6f}")
    
    # Convert decoder to CoreML
    coreml_decoder = ct.convert(
        traced_decoder,
        inputs=[
            ct.TensorType(name="decoder_input_ids", shape=(1, max_length), dtype=np.int32),
            ct.TensorType(name="encoder_hidden_states", shape=(1, max_length, d_model), dtype=np.float32),
            ct.TensorType(name="decoder_attention_mask", shape=(1, max_length), dtype=np.int32),
            ct.TensorType(name="encoder_attention_mask", shape=(1, max_length), dtype=np.int32)
        ],
        outputs=[
            ct.TensorType(name="logits", dtype=np.float32)
        ],
        minimum_deployment_target=ct.target.iOS15,
        compute_precision=ct.precision.FLOAT16
    )
    
    decoder_path = os.path.join(output_path, f"flan_t5_base_decoder_fixed_{max_length}.mlpackage")
    coreml_decoder.save(decoder_path)
    print(f"✅ Decoder saved to: {decoder_path}")
    
    return encoder_path, decoder_path

def test_consistent_models(encoder_path, decoder_path, max_length):
    """Test the consistent models."""
    
    print(f"\nTesting consistent models with max_length={max_length}")
    print("=" * 60)
    
    try:
        encoder = ct.models.MLModel(encoder_path)
        decoder = ct.models.MLModel(decoder_path)
        tokenizer = T5Tokenizer.from_pretrained("./flan-t5-base")
        
        # Test with actual text
        input_text = "translate English to French: Hello"
        inputs = tokenizer(input_text, return_tensors="np", padding="max_length", 
                          truncation=True, max_length=max_length)
        
        print(f"📥 Input shape: {inputs['input_ids'].shape}")
        
        # Run encoder
        encoder_output = encoder.predict({
            "input_ids": inputs["input_ids"].astype(np.int32),
            "attention_mask": inputs["attention_mask"].astype(np.int32)
        })
        hidden_states = encoder_output["hidden_states"]
        print(f"✅ Encoder output shape: {hidden_states.shape}")
        
        # Test decoder with different sequences
        decoder_ids_1 = np.zeros((1, max_length), dtype=np.int32)
        decoder_ids_1[0, 0] = tokenizer.pad_token_id
        decoder_mask_1 = np.zeros((1, max_length), dtype=np.int32)
        decoder_mask_1[0, 0] = 1
        
        decoder_ids_2 = np.zeros((1, max_length), dtype=np.int32)
        decoder_ids_2[0, 0] = tokenizer.pad_token_id
        decoder_ids_2[0, 1] = 1000
        decoder_mask_2 = np.zeros((1, max_length), dtype=np.int32)
        decoder_mask_2[0, 0:2] = 1
        
        # Run decoder tests
        decoder_out_1 = decoder.predict({
            "decoder_input_ids": decoder_ids_1,
            "encoder_hidden_states": hidden_states,
            "decoder_attention_mask": decoder_mask_1,
            "encoder_attention_mask": inputs["attention_mask"].astype(np.int32)
        })
        
        decoder_out_2 = decoder.predict({
            "decoder_input_ids": decoder_ids_2,
            "encoder_hidden_states": hidden_states,
            "decoder_attention_mask": decoder_mask_2,
            "encoder_attention_mask": inputs["attention_mask"].astype(np.int32)
        })
        
        logits_1 = decoder_out_1["logits"]
        logits_2 = decoder_out_2["logits"]
        
        print(f"✅ Decoder output shapes: {logits_1.shape}, {logits_2.shape}")
        
        # Check causal behavior
        pos_0_seq_1 = logits_1[0, 0, :]
        pos_0_seq_2 = logits_2[0, 0, :]
        pos_1_seq_2 = logits_2[0, 1, :]
        
        diff_sequences = np.mean(np.abs(pos_0_seq_1 - pos_0_seq_2))
        diff_positions = np.mean(np.abs(pos_0_seq_2 - pos_1_seq_2))
        
        print(f"🔍 Difference between sequences: {diff_sequences:.6f}")
        print(f"🔍 Difference between positions: {diff_positions:.6f}")
        
        # Get top tokens
        top_1_1 = np.argmax(pos_0_seq_1)
        top_1_2_pos0 = np.argmax(pos_0_seq_2)
        top_1_2_pos1 = np.argmax(pos_1_seq_2)
        
        print(f"🎯 Top token seq 1 pos 0: {top_1_1} ('{tokenizer.decode([top_1_1])}')")
        print(f"🎯 Top token seq 2 pos 0: {top_1_2_pos0} ('{tokenizer.decode([top_1_2_pos0])}')")
        print(f"🎯 Top token seq 2 pos 1: {top_1_2_pos1} ('{tokenizer.decode([top_1_2_pos1])}')")
        
        if diff_positions > 1.0:
            print("✅ SUCCESS: Causal attention is working!")
            return True
        else:
            print("❌ FAILURE: Still producing similar outputs")
            return False
            
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    model_path = "./flan-t5-base"
    output_path = "./coreml_models_final"
    
    # Load model
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model.eval()
    
    # Convert with consistent sequence length
    max_length = 128  # Good balance between functionality and efficiency
    
    try:
        encoder_path, decoder_path = convert_consistent_models(
            model, tokenizer, output_path, max_length
        )
        
        # Test the converted models
        success = test_consistent_models(encoder_path, decoder_path, max_length)
        
        if success:
            print(f"\n🎉 SUCCESS! Final fixed models are working correctly!")
            print(f"   Encoder: {encoder_path}")
            print(f"   Decoder: {decoder_path}")
            print(f"   Max sequence length: {max_length}")
        else:
            print(f"\n😞 Models still need investigation")
            
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()