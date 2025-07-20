#!/usr/bin/env python3
"""
Test the fixed CoreML models to verify they produce different outputs for different positions.
"""

import coremltools as ct
import numpy as np
from transformers import T5Tokenizer

def test_fixed_coreml_models():
    """Test the fixed CoreML models."""
    
    print("Testing Fixed CoreML Models")
    print("=" * 50)
    
    # Load models
    encoder_path = "coreml_models_fixed/flan_t5_base_encoder.mlpackage"
    decoder_path = "coreml_models_fixed/flan_t5_base_decoder_simplified.mlpackage"
    
    try:
        encoder = ct.models.MLModel(encoder_path)
        decoder = ct.models.MLModel(decoder_path)
        tokenizer = T5Tokenizer.from_pretrained("./flan-t5-base")
        
        print("✅ Models loaded successfully")
        
        # Test with actual text - encoder expects 512, decoder expects 64
        input_text = "translate English to French: Hello world"
        encoder_inputs = tokenizer(input_text, return_tensors="np", padding="max_length", 
                                 truncation=True, max_length=512)
        decoder_max_length = 64
        
        # Run encoder
        encoder_output = encoder.predict({
            "input_ids": encoder_inputs["input_ids"].astype(np.int32),
            "attention_mask": encoder_inputs["attention_mask"].astype(np.int32)
        })
        hidden_states = encoder_output["hidden_states"]
        
        print(f"✅ Encoder output shape: {hidden_states.shape}")
        
        # Test decoder with different sequences
        print("\nTesting decoder with different input sequences...")
        
        # Sequence 1: Start with pad token
        decoder_ids_1 = np.array([[tokenizer.pad_token_id] + [tokenizer.pad_token_id] * (decoder_max_length-1)], dtype=np.int32)
        decoder_mask_1 = np.array([[1] + [0] * (decoder_max_length-1)], dtype=np.int32)
        
        # Sequence 2: Start with pad token + some content
        decoder_ids_2 = np.array([[tokenizer.pad_token_id, 1000] + [tokenizer.pad_token_id] * (decoder_max_length-2)], dtype=np.int32)
        decoder_mask_2 = np.array([[1, 1] + [0] * (decoder_max_length-2)], dtype=np.int32)
        
        # Run decoder for both sequences
        decoder_out_1 = decoder.predict({
            "decoder_input_ids": decoder_ids_1,
            "encoder_hidden_states": hidden_states,
            "decoder_attention_mask": decoder_mask_1,
            "encoder_attention_mask": encoder_inputs["attention_mask"].astype(np.int32)
        })
        
        decoder_out_2 = decoder.predict({
            "decoder_input_ids": decoder_ids_2,
            "encoder_hidden_states": hidden_states,
            "decoder_attention_mask": decoder_mask_2,
            "encoder_attention_mask": encoder_inputs["attention_mask"].astype(np.int32)
        })
        
        logits_1 = decoder_out_1["logits"]
        logits_2 = decoder_out_2["logits"]
        
        print(f"Decoder output 1 shape: {logits_1.shape}")
        print(f"Decoder output 2 shape: {logits_2.shape}")
        
        # Check position 0 for both sequences
        pos_0_seq_1 = logits_1[0, 0, :]
        pos_0_seq_2 = logits_2[0, 0, :]
        
        # Check position 1 for sequence 2
        pos_1_seq_2 = logits_2[0, 1, :]
        
        # Calculate differences
        diff_between_sequences = np.mean(np.abs(pos_0_seq_1 - pos_0_seq_2))
        diff_between_positions = np.mean(np.abs(pos_0_seq_2 - pos_1_seq_2))
        
        print(f"\nDifference between sequences at position 0: {diff_between_sequences:.6f}")
        print(f"Difference between positions 0 and 1 in seq 2: {diff_between_positions:.6f}")
        
        # Get top tokens for each
        top_k = 5
        top_indices_seq1_pos0 = np.argpartition(pos_0_seq_1, -top_k)[-top_k:]
        top_indices_seq2_pos0 = np.argpartition(pos_0_seq_2, -top_k)[-top_k:]
        top_indices_seq2_pos1 = np.argpartition(pos_1_seq_2, -top_k)[-top_k:]
        
        top_scores_seq1_pos0 = pos_0_seq_1[top_indices_seq1_pos0]
        top_scores_seq2_pos0 = pos_0_seq_2[top_indices_seq2_pos0]
        top_scores_seq2_pos1 = pos_1_seq_2[top_indices_seq2_pos1]
        
        # Sort by score
        sorted_idx_1 = np.argsort(top_scores_seq1_pos0)[::-1]
        sorted_idx_2 = np.argsort(top_scores_seq2_pos0)[::-1]
        sorted_idx_3 = np.argsort(top_scores_seq2_pos1)[::-1]
        
        print(f"\nTop tokens for Sequence 1, Position 0:")
        for i in range(top_k):
            idx = top_indices_seq1_pos0[sorted_idx_1[i]]
            score = top_scores_seq1_pos0[sorted_idx_1[i]]
            token = tokenizer.decode([idx])
            print(f"  Token {idx} ('{token}'): {score:.3f}")
        
        print(f"\nTop tokens for Sequence 2, Position 0:")
        for i in range(top_k):
            idx = top_indices_seq2_pos0[sorted_idx_2[i]]
            score = top_scores_seq2_pos0[sorted_idx_2[i]]
            token = tokenizer.decode([idx])
            print(f"  Token {idx} ('{token}'): {score:.3f}")
        
        print(f"\nTop tokens for Sequence 2, Position 1:")
        for i in range(top_k):
            idx = top_indices_seq2_pos1[sorted_idx_3[i]]
            score = top_scores_seq2_pos1[sorted_idx_3[i]]
            token = tokenizer.decode([idx])
            print(f"  Token {idx} ('{token}'): {score:.3f}")
        
        # Verdict
        print("\n" + "=" * 50)
        if diff_between_positions > 1.0:  # Should be significantly different
            print("✅ SUCCESS: Fixed decoder produces different outputs for different positions!")
            print("   The causal attention mechanism is working correctly.")
            return True
        else:
            print("❌ FAILURE: Decoder still produces similar outputs for different positions")
            print("   The causal attention mechanism may still have issues.")
            return False
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fixed_coreml_models()
    if success:
        print("\n🎉 The fixed CoreML models are working correctly!")
        print("You can now use them for proper T5 text generation.")
    else:
        print("\n😞 The models still need further fixes.")