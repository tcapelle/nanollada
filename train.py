import os
import math
import argparse
import logging
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from config_llada import ModelConfig
from modelling_llada import LLaDAModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Simple text dataset for training
class TextDataset(Dataset):
    def __init__(self, tokenizer, max_length=512, data_size=1000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_size = data_size
        
        # For demo purposes, we'll just generate random tokens within vocab size range
        # In real training, you'd load actual text data
        # Make sure token IDs are within a safe range (avoiding special tokens)
        vocab_size = min(tokenizer.vocab_size, 1000)  # Use smaller vocab for testing
        self.data = [torch.randint(1, vocab_size-1, (max_length,)) for _ in range(data_size)]
        
    def __len__(self):
        return self.data_size
    
    def __getitem__(self, idx):
        return {"input_ids": self.data[idx]}

def forward_process(input_ids, mask_token_id, eps=1e-3):
    """
    Apply the forward diffusion process to the input_ids as described in the LLADA paper.
    This adds noise to the input_ids by masking tokens with a probability determined by t.
    
    Args:
        input_ids: tensor of token ids, shape (batch_size, seq_len)
        mask_token_id: id for the mask token
        eps: minimum masking probability
        
    Returns:
        noisy_batch: input_ids with some tokens masked
        masked_indices: boolean tensor indicating which tokens were masked
        p_mask: probability of masking at each position
    """
    b, l = input_ids.shape
    
    # Sample noise level t uniformly between 0 and 1
    # This determines how much noise to add (how many tokens to mask)
    t = torch.rand(b, device=input_ids.device)
    
    # Linear noise schedule as defined in the paper
    # p_mask ranges from eps to 1, controlled by t
    p_mask = (1 - eps) * t + eps
    
    # Expand p_mask to have the same shape as input_ids
    p_mask = p_mask[:, None].repeat(1, l)

    # Create a mask by sampling from a Bernoulli distribution
    # Each token has probability p_mask of being masked
    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    
    # Replace masked tokens with mask_token_id
    noisy_batch = torch.where(masked_indices, mask_token_id, input_ids)
    
    return noisy_batch, masked_indices, p_mask

def train_epoch(model, dataloader, optimizer, scheduler, device, mask_token_id, epoch):
    """Train the LLADA model for one epoch following the paper guidelines"""
    model.train()
    total_loss = 0
    
    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        
        # Randomly set 1% of the data to a random length (as mentioned in guidelines)
        # "We set 1% of the pre-training data to a random length that is uniformly sampled 
        # from the range [1, 4096]."
        if torch.rand(1) < 0.01:
            random_length = torch.randint(1, input_ids.shape[1] + 1, (1,))
            input_ids = input_ids[:, :random_length]
        
        # Apply the forward process (add noise via masking)
        noisy_batch, masked_indices, p_mask = forward_process(input_ids, mask_token_id)
        
        # Forward pass through the LLaDA model
        # "logits = model(input_ids=noisy_batch).logits"
        outputs = model(input_ids=noisy_batch)
        logits = outputs.logits
        
        # Compute loss only on masked tokens
        # Weight loss by inverse of masking probability
        # "token_loss = F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]"
        token_loss = F.cross_entropy(
            logits[masked_indices], 
            input_ids[masked_indices], 
            reduction='none'
        ) / p_mask[masked_indices]
        
        # "loss = token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])"
        # Normalize by batch size * sequence length
        loss = token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # Log progress
        if step % 10 == 0:
            logger.info(f"Epoch {epoch} | Step {step} | Loss {loss.item():.4f}")
            
            # Log masking statistics
            mask_ratio = masked_indices.float().mean().item()
            logger.info(f"  Avg mask ratio: {mask_ratio:.4f}")
    
    return total_loss / len(dataloader)

def validate(model, dataloader, device, mask_token_id):
    """Validate the LLADA model on the validation set"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            
            # Apply the forward process (add noise via masking)
            noisy_batch, masked_indices, p_mask = forward_process(input_ids, mask_token_id)
            
            # Forward pass
            outputs = model(input_ids=noisy_batch)
            logits = outputs.logits
            
            # Compute loss only on masked tokens
            token_loss = F.cross_entropy(
                logits[masked_indices], 
                input_ids[masked_indices], 
                reduction='none'
            ) / p_mask[masked_indices]
            
            loss = token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def sample_tokens(model, input_ids, mask_token_id, steps=10, temperature=0.0):
    """
    Sample from the model using the fixed-length sampling procedure described in the paper.
    
    Args:
        model: The trained LLADA model
        input_ids: Starting point for generation (prompt)
        mask_token_id: ID of the mask token
        steps: Number of sampling steps
        temperature: Temperature for sampling (0 = greedy)
        
    Returns:
        Generated token sequence
    """
    model.eval()
    
    # Create initial sequence with masks
    b, prompt_len = input_ids.shape
    gen_len = 32  # For demonstration, use a small generation length
    x = torch.full((b, prompt_len + gen_len), mask_token_id, device=input_ids.device)
    x[:, :prompt_len] = input_ids  # Copy the prompt
    
    # Run the reverse diffusion process
    for step in range(steps):
        # Get mask positions
        mask_indices = (x == mask_token_id)
        
        if not mask_indices.any():
            break  # All tokens have been generated
            
        # Forward pass to get token predictions
        with torch.no_grad():
            logits = model(x).logits
        
        # Get predictions for all positions
        if temperature == 0:
            # Greedy sampling
            predicted_tokens = torch.argmax(logits, dim=-1)
        else:
            # Sample from the distribution
            probs = F.softmax(logits / temperature, dim=-1)
            predicted_tokens = torch.multinomial(
                probs.view(-1, probs.size(-1)),
                num_samples=1
            ).view(probs.size(0), -1).squeeze(-1)
        
        # Calculate confidence scores
        probs = F.softmax(logits, dim=-1)
        confidence = torch.gather(
            probs, 
            dim=-1, 
            index=predicted_tokens.unsqueeze(-1)
        ).squeeze(-1)
        
        # Only consider masked positions for confidence
        masked_confidence = torch.where(mask_indices, confidence, torch.tensor(-float('inf'), device=x.device))
        
        # Determine how many tokens to unmask in this step
        n_masked = mask_indices.sum().item()
        tokens_per_step = max(1, n_masked // max(1, steps - step))
        
        # Create a mask for tokens to unmask in this step
        to_unmask = torch.zeros_like(mask_indices)
        for i in range(b):
            # Count masks in this batch item
            row_masks = mask_indices[i].sum().item()
            if row_masks > 0:
                # Find positions with highest confidence
                positions = masked_confidence[i].topk(min(tokens_per_step, row_masks)).indices
                to_unmask[i, positions] = True
        
        # Update only the tokens we want to unmask
        x = torch.where(to_unmask, predicted_tokens, x)
    
    return x

def create_tiny_llada_model():
    """Create a tiny LLADA model for training"""
    # Based on the guidelines, LLADA uses a Transformer Encoder
    # (which is just a Transformer Decoder without the causal mask)
    vocab_size = 1000  # Use a small vocab size for testing
    
    config = ModelConfig(
        d_model=128,               # Small hidden size
        n_heads=4,                 # Small number of attention heads
        n_layers=4,                # Small number of layers
        vocab_size=vocab_size,     # Small vocabulary size
        max_sequence_length=512,   # Shorter sequence length
        embedding_size=vocab_size, # Match vocab size
        mask_token_id=vocab_size-1, # Use the last token as mask token
        include_bias=True,         # Including bias terms for stability
        layer_norm_type="default", # Default layer norm
        activation_type="gelu",    # GELU activation
        block_type="sequential",   # Sequential block type
        attention_dropout=0.1,     # Default dropout
        residual_dropout=0.1,      # Default dropout
        embedding_dropout=0.1,     # Default dropout
        rope=True,                 # Use rotary positional embeddings
        weight_tying=True,         # Use weight tying
        init_fn="normal",          # Normal initialization
        init_std=0.02,             # Standard init_std
    )
    
    model = LLaDAModel(config)
    return model, config

def main():
    parser = argparse.ArgumentParser(description="Train a tiny LLADA model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--data_size", type=int, default=1000, help="Size of synthetic dataset")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    parser.add_argument("--save_path", type=str, default="./tiny_llada_model", help="Path to save model")
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model
    model, config = create_tiny_llada_model()
    model = model.to(device)
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")
    
    # Create tokenizer (for a tiny model, we'll use a simple one)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset and dataloader
    dataset = TextDataset(tokenizer, max_length=args.seq_len, data_size=args.data_size)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            mask_token_id=config.mask_token_id,
            epoch=epoch+1,
        )
        
        # Validate
        val_loss = validate(
            model=model,
            dataloader=val_dataloader,
            device=device,
            mask_token_id=config.mask_token_id,
        )
        
        logger.info(f"Epoch {epoch+1} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
    
    # Save model
    os.makedirs(args.save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.save_path, "model.pt"))
    logger.info(f"Model saved to {args.save_path}")
    
    # Demo generation
    logger.info("Generating sample text...")
    # Make sure the sample prompt is within vocabulary range
    sample_prompt = torch.randint(1, config.vocab_size-2, (1, 5), device=device)
    logger.info(f"Sample prompt: {sample_prompt}")
    
    try:
        generated = sample_tokens(
            model=model,
            input_ids=sample_prompt,
            mask_token_id=config.mask_token_id,
            steps=8,
            temperature=0.8
        )
        logger.info(f"Generated output shape: {generated.shape}")
    except Exception as e:
        logger.info(f"Generation error (for demonstration only): {e}")
        logger.info("This is expected in this toy example with random weights")

if __name__ == "__main__":
    main()