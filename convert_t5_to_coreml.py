#!/usr/bin/env python3
"""
Convert FLAN-T5 model to CoreML format.
Separates encoder and decoder components for individual conversion.
"""

import torch
import coremltools as ct
from transformers import T5ForConditionalGeneration, T5Tokenizer
import numpy as np
import os

def load_model_and_tokenizer(model_path):
    """Load the T5 model and tokenizer."""
    print("Loading model and tokenizer...")
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    return model, tokenizer

class T5EncoderWrapper(torch.nn.Module):
    """Wrapper for T5 encoder to make it standalone."""
    
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

class T5DecoderWrapper(torch.nn.Module):
    """Wrapper for T5 decoder to make it standalone."""
    
    def __init__(self, t5_model):
        super().__init__()
        self.decoder = t5_model.decoder
        self.lm_head = t5_model.lm_head
        
    def forward(self, decoder_input_ids, encoder_hidden_states, decoder_attention_mask=None, encoder_attention_mask=None):
        if decoder_attention_mask is None:
            decoder_attention_mask = torch.ones_like(decoder_input_ids)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(encoder_hidden_states.shape[:2])
            
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True
        )
        
        logits = self.lm_head(decoder_outputs.last_hidden_state)
        return logits

def convert_encoder_to_coreml(model, tokenizer, output_path):
    """Convert T5 encoder to CoreML."""
    print("Converting encoder to CoreML...")
    
    # Wrap the encoder
    encoder_wrapper = T5EncoderWrapper(model)
    encoder_wrapper.eval()
    
    # Create example inputs
    max_seq_length = 512  # T5 base max sequence length
    example_input_ids = torch.randint(0, tokenizer.vocab_size, (1, max_seq_length))
    example_attention_mask = torch.ones(1, max_seq_length)
    
    # Trace the model
    traced_encoder = torch.jit.trace(
        encoder_wrapper,
        (example_input_ids, example_attention_mask)
    )
    
    # Convert to CoreML
    coreml_encoder = ct.convert(
        traced_encoder,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, max_seq_length), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, max_seq_length), dtype=np.int32)
        ],
        outputs=[
            ct.TensorType(name="hidden_states", dtype=np.float32)
        ],
        minimum_deployment_target=ct.target.iOS15,
        compute_precision=ct.precision.FLOAT16
    )
    
    # Save the model
    encoder_path = os.path.join(output_path, "flan_t5_base_encoder.mlpackage")
    coreml_encoder.save(encoder_path)
    print(f"Encoder saved to: {encoder_path}")
    
    return coreml_encoder

def convert_decoder_to_coreml(model, tokenizer, output_path):
    """Convert T5 decoder to CoreML."""
    print("Converting decoder to CoreML...")
    
    # Wrap the decoder
    decoder_wrapper = T5DecoderWrapper(model)
    decoder_wrapper.eval()
    
    # Create example inputs
    max_seq_length = 512
    d_model = model.config.d_model  # 768 for T5-base
    
    example_decoder_input_ids = torch.randint(0, tokenizer.vocab_size, (1, max_seq_length))
    example_encoder_hidden_states = torch.randn(1, max_seq_length, d_model)
    example_decoder_attention_mask = torch.ones(1, max_seq_length)
    example_encoder_attention_mask = torch.ones(1, max_seq_length)
    
    # Trace the model
    traced_decoder = torch.jit.trace(
        decoder_wrapper,
        (example_decoder_input_ids, example_encoder_hidden_states, 
         example_decoder_attention_mask, example_encoder_attention_mask)
    )
    
    # Convert to CoreML
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
    
    # Save the model
    decoder_path = os.path.join(output_path, "flan_t5_base_decoder.mlpackage")
    coreml_decoder.save(decoder_path)
    print(f"Decoder saved to: {decoder_path}")
    
    return coreml_decoder

def main():
    model_path = "./flan-t5-base"
    output_path = "./coreml_models"
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    try:
        # Convert encoder
        print("\n" + "="*50)
        print("CONVERTING ENCODER")
        print("="*50)
        encoder_coreml = convert_encoder_to_coreml(model, tokenizer, output_path)
        
        # Convert decoder
        print("\n" + "="*50)
        print("CONVERTING DECODER")
        print("="*50)
        decoder_coreml = convert_decoder_to_coreml(model, tokenizer, output_path)
        
        print("\n" + "="*50)
        print("CONVERSION COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"Encoder: {output_path}/flan_t5_base_encoder.mlpackage")
        print(f"Decoder: {output_path}/flan_t5_base_decoder.mlpackage")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()