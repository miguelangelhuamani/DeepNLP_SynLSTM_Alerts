import os
import pickle
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from utils import load_umt_data

def sentiment_name(sentiment_id):
    """Convert sentiment ID to readable name"""
    if sentiment_id == 0:
        return "Negative"
    elif sentiment_id == 1:
        return "Neutral"
    else:
        return "Positive"

def analyze_dataset():
    # Load the data
    print("Loading dataset...")
    train_sentences, train_labels, train_sa, val_sentences, val_labels, val_sa, test_sentences, test_labels, test_sa = load_umt_data()
    
    # Count tweets in each set
    train_count = len(train_sentences)
    val_count = len(val_sentences)
    test_count = len(test_sentences)
    total_count = train_count + val_count + test_count
    
    # Count sentiments in each set
    train_sentiments = Counter(train_sa)
    val_sentiments = Counter(val_sa)
    test_sentiments = Counter(test_sa)
    
    # Calculate average tweet length (in tokens)
    train_lengths = [len(s) for s in train_sentences]
    val_lengths = [len(s) for s in val_sentences]
    test_lengths = [len(s) for s in test_sentences]
    
    avg_train_length = np.mean(train_lengths)
    avg_val_length = np.mean(val_lengths)
    avg_test_length = np.mean(test_lengths)
    avg_total_length = np.mean(train_lengths + val_lengths + test_lengths)
    
    # Print results
    print("\n===== DATASET STATISTICS =====")
    print(f"\nTotal tweets: {total_count}")
    print(f"  - Training:   {train_count} ({train_count/total_count*100:.1f}%)")
    print(f"  - Validation: {val_count} ({val_count/total_count*100:.1f}%)")
    print(f"  - Testing:    {test_count} ({test_count/total_count*100:.1f}%)")
    
    print("\nSentiment distribution:")
    print("  TRAINING SET:")
    for sentiment_id, count in sorted(train_sentiments.items()):
        print(f"    - {sentiment_name(sentiment_id)}: {count} ({count/train_count*100:.1f}%)")
    
    print("  VALIDATION SET:")
    for sentiment_id, count in sorted(val_sentiments.items()):
        print(f"    - {sentiment_name(sentiment_id)}: {count} ({count/val_count*100:.1f}%)")
        
    print("  TEST SET:")
    for sentiment_id, count in sorted(test_sentiments.items()):
        print(f"    - {sentiment_name(sentiment_id)}: {count} ({count/test_count*100:.1f}%)")
    
    print("\nAverage tweet length (tokens):")
    print(f"  - Training:   {avg_train_length:.2f}")
    print(f"  - Validation: {avg_val_length:.2f}")
    print(f"  - Testing:    {avg_test_length:.2f}")
    print(f"  - Overall:    {avg_total_length:.2f}")
    
    # Create length distribution histogram
    plt.figure(figsize=(10, 6))
    plt.hist(train_lengths, alpha=0.5, label='Training', bins=30)
    plt.hist(val_lengths, alpha=0.5, label='Validation', bins=30)
    plt.hist(test_lengths, alpha=0.5, label='Testing', bins=30)
    plt.xlabel('Tweet Length (tokens)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.title('Distribution of Tweet Lengths')
    
    # Save the plot
    os.makedirs('stats', exist_ok=True)
    plt.savefig('stats/tweet_length_distribution.png')
    print("\nLength distribution histogram saved to 'stats/tweet_length_distribution.png'")

if __name__ == "__main__":
    analyze_dataset()