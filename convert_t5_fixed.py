#!/usr/bin/env python3
"""
Fixed T5 to CoreML conversion with proper causal attention handling.
"""

import torch
import coremltools as ct
from transformers import T5ForConditionalGeneration, T5Tokenizer
import numpy as np
import os

class T5DecoderWrapperFixed(torch.nn.Module):
    """Fixed wrapper for T5 decoder with proper causal attention."""
    
    def __init__(self, t5_model):
        super().__init__()
        self.decoder = t5_model.decoder
        self.lm_head = t5_model.lm_head
        self.config = t5_model.config
        
    def forward(self, decoder_input_ids, encoder_hidden_states, decoder_attention_mask=None, encoder_attention_mask=None):
        batch_size, seq_len = decoder_input_ids.shape
        device = decoder_input_ids.device
        
        # Create proper causal attention mask
        if decoder_attention_mask is None:
            decoder_attention_mask = torch.ones_like(decoder_input_ids)
            
        # Create causal mask (lower triangular)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        # Expand to batch size and add head dimension
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        causal_mask = causal_mask.expand(batch_size, 1, seq_len, seq_len)
        
        # Apply decoder attention mask to causal mask
        decoder_mask_expanded = decoder_attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
        causal_mask = causal_mask & decoder_mask_expanded
        
        # Convert to attention scores format (0 for attend, -inf for mask)
        causal_attention_mask = torch.zeros_like(causal_mask, dtype=torch.float)
        causal_attention_mask.masked_fill_(~causal_mask, float('-inf'))
        
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(encoder_hidden_states.shape[:2], device=device)
            
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False,  # Disable caching for consistent conversion
            return_dict=True
        )
        
        logits = self.lm_head(decoder_outputs.last_hidden_state)
        return logits

class T5DecoderSingleStep(torch.nn.Module):
    """T5 decoder for single step generation (more suitable for CoreML)."""
    
    def __init__(self, t5_model):
        super().__init__()
        self.decoder = t5_model.decoder
        self.lm_head = t5_model.lm_head
        
    def forward(self, decoder_input_ids, encoder_hidden_states, position_ids, encoder_attention_mask=None):
        """
        Single step decoder that processes one token at a time.
        position_ids: [batch_size, 1] - current position in sequence
        """
        batch_size = decoder_input_ids.shape[0]
        device = decoder_input_ids.device
        
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(encoder_hidden_states.shape[:2], device=device)
        
        # Create attention mask that allows attention to all previous positions + current
        max_pos = position_ids.max().item() + 1
        decoder_attention_mask = torch.zeros(batch_size, max_pos, device=device)
        for i in range(batch_size):
            pos = position_ids[i, 0].item()
            decoder_attention_mask[i, :pos+1] = 1
            
        # Pad decoder input to match attention mask size
        if decoder_input_ids.shape[1] < max_pos:
            padding = torch.zeros(batch_size, max_pos - decoder_input_ids.shape[1], 
                                device=device, dtype=decoder_input_ids.dtype)
            decoder_input_ids = torch.cat([decoder_input_ids, padding], dim=1)
        
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False,
            return_dict=True
        )
        
        # Get logits for the current position
        current_logits = self.lm_head(decoder_outputs.last_hidden_state)
        
        # Return only the logits for the current position
        batch_indices = torch.arange(batch_size, device=device)
        pos_indices = position_ids.squeeze(-1)
        current_token_logits = current_logits[batch_indices, pos_indices, :]
        
        return current_token_logits.unsqueeze(1)  # [batch_size, 1, vocab_size]

def test_original_model():
    """Test the original PyTorch model to verify it works correctly."""
    print("Testing original PyTorch model...")
    
    model_path = "./flan-t5-base"
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model.eval()
    
    # Test with a simple translation task
    input_text = "translate English to French: Hello world"
    inputs = tokenizer(input_text, return_tensors="pt")
    
    with torch.no_grad():
        # Test encoder
        encoder_outputs = model.encoder(**inputs)
        print(f"✅ Encoder output shape: {encoder_outputs.last_hidden_state.shape}")
        
        # Test decoder with greedy generation
        output_ids = model.generate(
            inputs.input_ids,
            max_length=20,
            num_beams=1,
            do_sample=False
        )
        
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"✅ Generated text: {output_text}")
        
    return model, tokenizer

def convert_decoder_fixed(model, tokenizer, output_path, use_single_step=True):
    """Convert T5 decoder with proper causal attention handling."""
    print(f"Converting decoder (single_step={use_single_step}) to CoreML...")
    
    if use_single_step:
        decoder_wrapper = T5DecoderSingleStep(model)
        max_seq_length = 128  # Smaller for single step
        
        # Example inputs for single step
        example_decoder_input_ids = torch.randint(0, tokenizer.vocab_size, (1, 10))
        example_encoder_hidden_states = torch.randn(1, 128, model.config.d_model)
        example_position_ids = torch.tensor([[9]])  # Current position
        example_encoder_attention_mask = torch.ones(1, 128)
        
        traced_decoder = torch.jit.trace(
            decoder_wrapper,
            (example_decoder_input_ids, example_encoder_hidden_states, 
             example_position_ids, example_encoder_attention_mask)
        )
        
        coreml_decoder = ct.convert(
            traced_decoder,
            inputs=[
                ct.TensorType(name="decoder_input_ids", shape=(1, ct.RangeDim(1, max_seq_length)), dtype=np.int32),
                ct.TensorType(name="encoder_hidden_states", shape=(1, 128, model.config.d_model), dtype=np.float32),
                ct.TensorType(name="position_ids", shape=(1, 1), dtype=np.int32),
                ct.TensorType(name="encoder_attention_mask", shape=(1, 128), dtype=np.int32)
            ],
            outputs=[
                ct.TensorType(name="logits", dtype=np.float32)
            ],
            minimum_deployment_target=ct.target.iOS15,
            compute_precision=ct.precision.FLOAT16
        )
        
        decoder_path = os.path.join(output_path, "flan_t5_base_decoder_single_step.mlpackage")
        
    else:
        decoder_wrapper = T5DecoderWrapperFixed(model)
        max_seq_length = 128
        d_model = model.config.d_model
        
        example_decoder_input_ids = torch.randint(0, tokenizer.vocab_size, (1, max_seq_length))
        example_encoder_hidden_states = torch.randn(1, max_seq_length, d_model)
        example_decoder_attention_mask = torch.ones(1, max_seq_length)
        example_encoder_attention_mask = torch.ones(1, max_seq_length)
        
        traced_decoder = torch.jit.trace(
            decoder_wrapper,
            (example_decoder_input_ids, example_encoder_hidden_states,
             example_decoder_attention_mask, example_encoder_attention_mask)
        )
        
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
        
        decoder_path = os.path.join(output_path, "flan_t5_base_decoder_fixed.mlpackage")
    
    coreml_decoder.save(decoder_path)
    print(f"Fixed decoder saved to: {decoder_path}")
    
    return coreml_decoder

def test_decoder_outputs(model, tokenizer):
    """Test different positions produce different outputs."""
    print("\nTesting decoder output variation...")
    
    model.eval()
    encoder_input = tokenizer("translate English to French: Hello", return_tensors="pt")
    
    with torch.no_grad():
        encoder_outputs = model.encoder(**encoder_input)
        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        # Test decoder with different positions
        decoder_input_1 = torch.tensor([[tokenizer.pad_token_id]])  # Start token
        decoder_input_2 = torch.tensor([[tokenizer.pad_token_id, 1234]])  # Start + one token
        
        decoder_out_1 = model.decoder(
            input_ids=decoder_input_1,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_input.attention_mask
        )
        
        decoder_out_2 = model.decoder(
            input_ids=decoder_input_2,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_input.attention_mask
        )
        
        logits_1 = model.lm_head(decoder_out_1.last_hidden_state)
        logits_2 = model.lm_head(decoder_out_2.last_hidden_state)
        
        # Check if outputs are different
        pos_0_logits_1 = logits_1[0, 0, :].numpy()
        pos_0_logits_2 = logits_2[0, 0, :].numpy()
        pos_1_logits_2 = logits_2[0, 1, :].numpy()
        
        diff_across_inputs = np.mean(np.abs(pos_0_logits_1 - pos_0_logits_2))
        diff_across_positions = np.mean(np.abs(pos_0_logits_2 - pos_1_logits_2))
        
        print(f"Difference across inputs: {diff_across_inputs:.4f}")
        print(f"Difference across positions: {diff_across_positions:.4f}")
        
        if diff_across_positions > 0.01:
            print("✅ Original model produces different outputs for different positions")
        else:
            print("❌ Original model issue detected!")

def main():
    model_path = "./flan-t5-base"
    output_path = "./coreml_models_fixed"
    
    os.makedirs(output_path, exist_ok=True)
    
    # Test original model first
    model, tokenizer = test_original_model()
    test_decoder_outputs(model, tokenizer)
    
    try:
        # Convert with both approaches
        print("\n" + "="*60)
        print("CONVERTING FIXED DECODER (Full Sequence)")
        print("="*60)
        decoder_fixed = convert_decoder_fixed(model, tokenizer, output_path, use_single_step=False)
        
        print("\n" + "="*60)
        print("CONVERTING SINGLE STEP DECODER")
        print("="*60)
        decoder_single_step = convert_decoder_fixed(model, tokenizer, output_path, use_single_step=True)
        
        print("\n✅ Both decoder variants converted successfully!")
        print(f"Fixed models saved in: {output_path}")
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()