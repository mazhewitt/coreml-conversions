#!/usr/bin/env python3
"""
Simplified T5 to CoreML conversion with focus on fixing the decoder issue.
"""

import torch
import coremltools as ct
from transformers import T5ForConditionalGeneration, T5Tokenizer
import numpy as np
import os

class T5DecoderSimplified(torch.nn.Module):
    """Simplified T5 decoder that relies on transformers' built-in attention."""
    
    def __init__(self, t5_model):
        super().__init__()
        self.decoder = t5_model.decoder
        self.lm_head = t5_model.lm_head
        
    def forward(self, decoder_input_ids, encoder_hidden_states, decoder_attention_mask=None, encoder_attention_mask=None):
        if decoder_attention_mask is None:
            decoder_attention_mask = torch.ones_like(decoder_input_ids)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(encoder_hidden_states.shape[:2])
            
        # Let the transformers library handle the causal masking internally
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

def test_decoder_behavior():
    """Test to understand why the decoder is producing identical outputs."""
    print("=" * 60)
    print("DEBUGGING DECODER BEHAVIOR")
    print("=" * 60)
    
    model_path = "./flan-t5-base"
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model.eval()
    
    # Create test inputs
    input_text = "translate English to French: Hello"
    inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", max_length=10)
    
    with torch.no_grad():
        # Get encoder outputs
        encoder_outputs = model.encoder(**inputs)
        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        # Test decoder with different input sequences
        print("\nTesting decoder with different sequences:")
        
        # Sequence 1: Just pad token
        decoder_ids_1 = torch.tensor([[tokenizer.pad_token_id] + [tokenizer.pad_token_id] * 9])
        decoder_mask_1 = torch.tensor([[1] + [0] * 9])
        
        # Sequence 2: Pad token + some content
        decoder_ids_2 = torch.tensor([[tokenizer.pad_token_id, 1000] + [tokenizer.pad_token_id] * 8])
        decoder_mask_2 = torch.tensor([[1, 1] + [0] * 8])
        
        # Run decoder
        decoder_out_1 = model.decoder(
            input_ids=decoder_ids_1,
            attention_mask=decoder_mask_1,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=inputs.attention_mask,
            use_cache=False
        )
        
        decoder_out_2 = model.decoder(
            input_ids=decoder_ids_2,
            attention_mask=decoder_mask_2,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=inputs.attention_mask,
            use_cache=False
        )
        
        logits_1 = model.lm_head(decoder_out_1.last_hidden_state)
        logits_2 = model.lm_head(decoder_out_2.last_hidden_state)
        
        # Check first position logits
        pos_0_seq_1 = logits_1[0, 0, :]
        pos_0_seq_2 = logits_2[0, 0, :]
        pos_1_seq_2 = logits_2[0, 1, :]
        
        diff_seq = torch.mean(torch.abs(pos_0_seq_1 - pos_0_seq_2)).item()
        diff_pos = torch.mean(torch.abs(pos_0_seq_2 - pos_1_seq_2)).item()
        
        print(f"Difference between sequences at pos 0: {diff_seq:.6f}")
        print(f"Difference between positions in seq 2: {diff_pos:.6f}")
        
        # Check top tokens
        top_1_seq_1 = torch.topk(pos_0_seq_1, 3)
        top_1_seq_2 = torch.topk(pos_0_seq_2, 3)
        top_1_pos_2 = torch.topk(pos_1_seq_2, 3)
        
        print(f"\nSequence 1, Position 0 top tokens: {top_1_seq_1.indices.tolist()}")
        print(f"Sequence 2, Position 0 top tokens: {top_1_seq_2.indices.tolist()}")
        print(f"Sequence 2, Position 1 top tokens: {top_1_pos_2.indices.tolist()}")
        
        if diff_pos > 0.01:
            print("✅ Original PyTorch model works correctly")
            return True
        else:
            print("❌ Issue detected in original model!")
            return False

def convert_models_simple(model, tokenizer, output_path):
    """Convert with simpler approach to avoid tracing issues."""
    print("\n" + "=" * 60)
    print("CONVERTING WITH SIMPLIFIED APPROACH")
    print("=" * 60)
    
    # Convert encoder (this should work fine)
    from convert_t5_to_coreml import T5EncoderWrapper, convert_encoder_to_coreml
    
    print("Converting encoder...")
    encoder_coreml = convert_encoder_to_coreml(model, tokenizer, output_path)
    
    # Convert decoder with smaller sequence length and simpler approach
    print("Converting decoder...")
    decoder_wrapper = T5DecoderSimplified(model)
    decoder_wrapper.eval()
    
    # Use smaller sequence length to avoid memory issues
    max_seq_length = 64
    d_model = model.config.d_model
    
    example_decoder_input_ids = torch.randint(0, 1000, (1, max_seq_length))  # Smaller vocab range
    example_encoder_hidden_states = torch.randn(1, max_seq_length, d_model)
    example_decoder_attention_mask = torch.ones(1, max_seq_length)
    example_encoder_attention_mask = torch.ones(1, max_seq_length)
    
    # Test the wrapper first
    print("Testing decoder wrapper...")
    with torch.no_grad():
        test_output = decoder_wrapper(
            example_decoder_input_ids,
            example_encoder_hidden_states,
            example_decoder_attention_mask,
            example_encoder_attention_mask
        )
        print(f"Wrapper output shape: {test_output.shape}")
        
        # Check if different positions produce different outputs
        pos_0_logits = test_output[0, 0, :]
        pos_1_logits = test_output[0, 1, :]
        diff = torch.mean(torch.abs(pos_0_logits - pos_1_logits)).item()
        print(f"Difference between positions 0 and 1: {diff:.6f}")
        
        if diff < 0.01:
            print("❌ Wrapper is producing identical outputs!")
            return None
        else:
            print("✅ Wrapper produces different outputs for different positions")
    
    # Trace the model
    print("Tracing decoder...")
    try:
        traced_decoder = torch.jit.trace(
            decoder_wrapper,
            (example_decoder_input_ids, example_encoder_hidden_states,
             example_decoder_attention_mask, example_encoder_attention_mask)
        )
        
        print("✅ Tracing successful")
        
        # Test traced model
        with torch.no_grad():
            traced_output = traced_decoder(
                example_decoder_input_ids,
                example_encoder_hidden_states,
                example_decoder_attention_mask,
                example_encoder_attention_mask
            )
            
            traced_pos_0 = traced_output[0, 0, :]
            traced_pos_1 = traced_output[0, 1, :]
            traced_diff = torch.mean(torch.abs(traced_pos_0 - traced_pos_1)).item()
            print(f"Traced model difference between positions: {traced_diff:.6f}")
            
            if traced_diff < 0.01:
                print("❌ Tracing broke the causal behavior!")
                return None
            else:
                print("✅ Traced model maintains causal behavior")
        
    except Exception as e:
        print(f"❌ Tracing failed: {e}")
        return None
    
    # Convert to CoreML
    print("Converting to CoreML...")
    try:
        coreml_decoder = ct.convert(
            traced_decoder,
            inputs=[
                ct.TensorType(name="decoder_input_ids", shape=(1, max_seq_length), dtype=np.int32),
                ct.TensorType(name="encoder_hidden_states", shape=(1, max_seq_length, d_model), dtype=np.float32),
                ct.TensorType(name="decoder_attention_mask", shape=(1, max_seq_length), dtype=np.int32),
                ct.TensorType(name="encoder_attention_mask", shape=(1, max_seq_length), dtype=np.int32)
            ],
            outputs=[
                ct.TensorType(name="logits", dtype=np.float32)
            ],
            minimum_deployment_target=ct.target.iOS15,
            compute_precision=ct.precision.FLOAT16
        )
        
        decoder_path = os.path.join(output_path, "flan_t5_base_decoder_simplified.mlpackage")
        coreml_decoder.save(decoder_path)
        print(f"✅ Decoder saved to: {decoder_path}")
        
        return coreml_decoder
        
    except Exception as e:
        print(f"❌ CoreML conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    model_path = "./flan-t5-base"
    output_path = "./coreml_models_fixed"
    
    os.makedirs(output_path, exist_ok=True)
    
    # Load model
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model.eval()
    
    # Test original behavior
    if not test_decoder_behavior():
        print("❌ Original model has issues, stopping conversion")
        return
    
    # Convert models
    result = convert_models_simple(model, tokenizer, output_path)
    
    if result:
        print("\n✅ Conversion completed successfully!")
    else:
        print("\n❌ Conversion failed")

if __name__ == "__main__":
    main()