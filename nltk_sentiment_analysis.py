"""
NLTK Sentiment Analysis Program
This script demonstrates sentiment analysis using NLTK with sample documents.
Uses VADER sentiment analyzer, which is effective for social media and general text.
"""

import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from collections import Counter

# Download required NLTK data (comment out after first run)
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# Sample documents for sentiment analysis
SAMPLE_DOCUMENTS = {
    "Movie Review 1": "This movie was absolutely fantastic! I loved every minute of it. The acting was superb and the plot kept me engaged throughout. A masterpiece!",
    
    "Movie Review 2": "Terrible film. Couldn't even finish watching it. Boring storyline, awful acting, and a complete waste of time and money.",
    
    "Product Review 1": "Amazing product! It works exactly as advertised. Great quality, excellent customer service. Highly recommended!",
    
    "Product Review 2": "Don't waste your money. Poor quality, broke after a week. Customer service was unhelpful. Very disappointed.",
    
    "Tech Review": "The new smartphone is quite good. Battery life is impressive, camera quality is decent. Some minor issues but overall satisfactory.",
    
    "Restaurant Review 1": "Best restaurant experience ever! Food was delicious, service was impeccable, atmosphere was perfect. Will definitely return!",
    
    "Restaurant Review 2": "Disappointing experience. Cold food, slow service, rude staff. Not worth the high prices.",
    
    "Book Review 1": "Couldn't put this book down! Compelling story with great character development. An absolute page-turner!",
    
    "Book Review 2": "Dull and uninspiring. Lost interest halfway through. The narrative was confusing and poorly structured.",
    
    "Neutral Statement": "The weather today is cloudy. Temperature is around 15 degrees Celsius. There might be rain later.",
    
    "Mixed Opinion": "The software has some great features, but the user interface could be better. Worth trying, though not perfect.",
    
    "Service Experience": "Had an okay experience. Nothing special but nothing terrible either. Average pricing for average service.",
}


def initialize_sentiment_analyzer():
    """Initialize VADER sentiment analyzer"""
    print("\n" + "=" * 70)
    print("INITIALIZING NLTK SENTIMENT ANALYSIS")
    print("=" * 70)
    
    sia = SentimentIntensityAnalyzer()
    print("✓ VADER Sentiment Analyzer initialized successfully")
    print("  (VADER: Valence Aware Dictionary and sEntiment Reasoner)")
    
    return sia


def analyze_sentiment(text, sia):
    """
    Analyze sentiment of a single text
    Returns: compound score (-1 to 1), positive, negative, neutral scores
    """
    scores = sia.polarity_scores(text)
    return scores


def classify_sentiment(compound_score):
    """Classify sentiment based on compound score"""
    if compound_score >= 0.05:
        return "POSITIVE"
    elif compound_score <= -0.05:
        return "NEGATIVE"
    else:
        return "NEUTRAL"


def analyze_all_documents(documents, sia):
    """Analyze sentiment for all sample documents"""
    print("\n" + "=" * 70)
    print("STEP 1: SENTIMENT ANALYSIS OF SAMPLE DOCUMENTS")
    print("=" * 70)
    
    results = []
    
    for doc_name, text in documents.items():
        scores = analyze_sentiment(text, sia)
        sentiment_class = classify_sentiment(scores['compound'])
        
        results.append({
            'Document': doc_name,
            'Text': text,
            'Positive': scores['pos'],
            'Negative': scores['neg'],
            'Neutral': scores['neu'],
            'Compound': scores['compound'],
            'Sentiment': sentiment_class
        })
        
        print(f"\n{doc_name}:")
        print(f"  Text: {text[:70]}..." if len(text) > 70 else f"  Text: {text}")
        print(f"  Sentiment: {sentiment_class}")
        print(f"  Scores - Positive: {scores['pos']:.4f}, Negative: {scores['neg']:.4f}, Neutral: {scores['neu']:.4f}")
        print(f"  Compound Score: {scores['compound']:.4f}")
    
    return results


def visualize_sentiment_scores(results):
    """Visualize sentiment analysis results"""
    print("\n" + "=" * 70)
    print("STEP 2: VISUALIZE SENTIMENT SCORES")
    print("=" * 70)
    
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Compound scores for all documents
    colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in df['Compound']]
    axes[0, 0].barh(range(len(df)), df['Compound'], color=colors, alpha=0.7, edgecolor='black')
    axes[0, 0].set_yticks(range(len(df)))
    axes[0, 0].set_yticklabels(df['Document'])
    axes[0, 0].set_xlabel('Sentiment Score (Compound)', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Overall Sentiment Scores', fontsize=12, fontweight='bold')
    axes[0, 0].axvline(x=0, color='black', linestyle='--', linewidth=1)
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    for i, v in enumerate(df['Compound']):
        axes[0, 0].text(v + 0.01 if v > 0 else v - 0.01, i, f'{v:.3f}', 
                       va='center', ha='left' if v > 0 else 'right', fontweight='bold')
    
    # Plot 2: Stacked bar chart of pos/neg/neu scores
    x_pos = np.arange(len(df))
    width = 0.6
    
    axes[0, 1].bar(x_pos, df['Positive'], width, label='Positive', color='green', alpha=0.7)
    axes[0, 1].bar(x_pos, df['Negative'], width, bottom=df['Positive'], 
                   label='Negative', color='red', alpha=0.7)
    axes[0, 1].bar(x_pos, df['Neutral'], width, 
                   bottom=df['Positive'] + df['Negative'], label='Neutral', color='gray', alpha=0.7)
    
    axes[0, 1].set_ylabel('Score', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Sentiment Score Composition', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(df['Document'], rotation=45, ha='right')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Sentiment distribution pie chart
    sentiment_counts = df['Sentiment'].value_counts()
    colors_pie = {'POSITIVE': 'green', 'NEGATIVE': 'red', 'NEUTRAL': 'gray'}
    pie_colors = [colors_pie.get(x, 'blue') for x in sentiment_counts.index]
    
    axes[1, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
                   colors=pie_colors, startangle=90, textprops={'fontsize': 11, 'weight': 'bold'})
    axes[1, 0].set_title('Sentiment Distribution', fontsize=12, fontweight='bold')
    
    # Plot 4: Scatter plot - Positive vs Negative scores
    scatter = axes[1, 1].scatter(df['Positive'], df['Negative'], 
                                 s=300, c=df['Compound'], cmap='RdYlGn', 
                                 alpha=0.6, edgecolors='black', linewidth=2)
    axes[1, 1].set_xlabel('Positive Score', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Negative Score', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Positive vs Negative Score Matrix', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label('Compound Score', fontsize=10, fontweight='bold')
    
    # Annotate with document names (first 5 chars for clarity)
    for i, row in df.iterrows():
        axes[1, 1].annotate(row['Document'][:10], (row['Positive'], row['Negative']), 
                           fontsize=8, alpha=0.7, ha='center')
    
    plt.tight_layout()
    plt.show()
    print("✓ Sentiment visualization displayed")
    
    return df


def analyze_sentence_level_sentiment(text, sia):
    """Analyze sentiment at sentence level"""
    print("\n" + "=" * 70)
    print("STEP 3: SENTENCE-LEVEL SENTIMENT ANALYSIS")
    print("=" * 70)
    
    # Select a document with multiple sentences for analysis
    sample_text = text
    sentences = sent_tokenize(sample_text)
    
    print(f"\nAnalyzing text: {sample_text[:100]}...")
    print(f"Number of sentences: {len(sentences)}\n")
    
    sentence_sentiments = []
    for i, sentence in enumerate(sentences, 1):
        scores = analyze_sentiment(sentence, sia)
        sentiment = classify_sentiment(scores['compound'])
        
        sentence_sentiments.append({
            'Sentence': sentence,
            'Sentiment': sentiment,
            'Compound': scores['compound'],
            'Positive': scores['pos'],
            'Negative': scores['neg'],
            'Neutral': scores['neu']
        })
        
        print(f"Sentence {i}: {sentence}")
        print(f"  Sentiment: {sentiment} | Compound: {scores['compound']:.4f}\n")
    
    return sentence_sentiments


def visualize_sentence_sentiments(sentence_sentiments):
    """Visualize sentence-level sentiment"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    df_sent = pd.DataFrame(sentence_sentiments)
    
    # Plot 1: Sentence sentiment scores
    colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in df_sent['Compound']]
    sentence_labels = [s[:40] + '...' if len(s) > 40 else s for s in df_sent['Sentence']]
    
    axes[0].barh(range(len(df_sent)), df_sent['Compound'], color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_yticks(range(len(df_sent)))
    axes[0].set_yticklabels(sentence_labels, fontsize=9)
    axes[0].set_xlabel('Sentiment Score (Compound)', fontsize=11, fontweight='bold')
    axes[0].set_title('Sentence-Level Sentiment Analysis', fontsize=12, fontweight='bold')
    axes[0].axvline(x=0, color='black', linestyle='--', linewidth=1)
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Sentiment composition for each sentence
    x_pos = np.arange(len(df_sent))
    width = 0.6
    
    axes[1].bar(x_pos, df_sent['Positive'], width, label='Positive', color='green', alpha=0.7)
    axes[1].bar(x_pos, df_sent['Negative'], width, bottom=df_sent['Positive'], 
               label='Negative', color='red', alpha=0.7)
    axes[1].bar(x_pos, df_sent['Neutral'], width, 
               bottom=df_sent['Positive'] + df_sent['Negative'], label='Neutral', color='gray', alpha=0.7)
    
    axes[1].set_ylabel('Score', fontsize=11, fontweight='bold')
    axes[1].set_title('Sentiment Score Composition by Sentence', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(range(1, len(df_sent) + 1))
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    print("✓ Sentence-level visualization displayed")


def generate_summary_statistics(df):
    """Generate summary statistics of sentiment analysis"""
    print("\n" + "=" * 70)
    print("STEP 4: SUMMARY STATISTICS")
    print("=" * 70)
    
    print(f"\nTotal documents analyzed: {len(df)}")
    print(f"\nSentiment Distribution:")
    for sentiment, count in df['Sentiment'].value_counts().items():
        percentage = (count / len(df)) * 100
        print(f"  {sentiment}: {count} ({percentage:.1f}%)")
    
    print(f"\nCompound Score Statistics:")
    print(f"  Mean: {df['Compound'].mean():.4f}")
    print(f"  Median: {df['Compound'].median():.4f}")
    print(f"  Std Dev: {df['Compound'].std():.4f}")
    print(f"  Min: {df['Compound'].min():.4f}")
    print(f"  Max: {df['Compound'].max():.4f}")
    
    print(f"\nPositive Score Statistics:")
    print(f"  Mean: {df['Positive'].mean():.4f}")
    print(f"  Max: {df['Positive'].max():.4f}")
    
    print(f"\nNegative Score Statistics:")
    print(f"  Mean: {df['Negative'].mean():.4f}")
    print(f"  Max: {df['Negative'].max():.4f}")
    
    # Top and bottom documents
    print(f"\nMost Positive Document:")
    top_idx = df['Compound'].idxmax()
    print(f"  {df.loc[top_idx, 'Document']}: {df.loc[top_idx, 'Compound']:.4f}")
    
    print(f"\nMost Negative Document:")
    bottom_idx = df['Compound'].idxmin()
    print(f"  {df.loc[bottom_idx, 'Document']}: {df.loc[bottom_idx, 'Compound']:.4f}")


def analyze_custom_text(sia):
    """Allow user to analyze custom text"""
    print("\n" + "=" * 70)
    print("STEP 5: ANALYZE CUSTOM TEXT")
    print("=" * 70)
    
    custom_examples = [
        "I absolutely love this! This is the best day ever!",
        "This is terrible and I hate it.",
        "The weather is nice today.",
        "It's okay, nothing special.",
    ]
    
    print("\nAnalyzing custom text examples:\n")
    for text in custom_examples:
        scores = analyze_sentiment(text, sia)
        sentiment = classify_sentiment(scores['compound'])
        
        print(f"Text: {text}")
        print(f"Sentiment: {sentiment}")
        print(f"Scores: Positive={scores['pos']:.3f}, Negative={scores['neg']:.3f}, Neutral={scores['neu']:.3f}")
        print(f"Compound: {scores['compound']:.4f}\n")


def print_final_summary():
    """Print final summary"""
    print("\n" + "=" * 70)
    print("FINAL SUMMARY - NLTK SENTIMENT ANALYSIS")
    print("=" * 70)
    
    print("\nSentiment Analysis Complete!")
    print("\nKey Insights:")
    print("  ✓ VADER is effective for social media and general text sentiment")
    print("  ✓ Compound score ranges from -1 (most negative) to +1 (most positive)")
    print("  ✓ Scores: -1 to -0.05 = Negative, -0.05 to +0.05 = Neutral, +0.05 to +1 = Positive")
    print("  ✓ Individual scores (pos/neg/neu) show sentiment distribution")
    print("  ✓ Sentence-level analysis provides detailed insight into document sentiment")
    
    print("\nUse Cases:")
    print("  • Social media monitoring")
    print("  • Customer review analysis")
    print("  • Brand sentiment tracking")
    print("  • Opinion mining")
    print("  • Emotion detection in text")
    
    print("\n" + "=" * 70)


def main():
    """Main function to run sentiment analysis"""
    print("\n" + "=" * 70)
    print("NLTK SENTIMENT ANALYSIS PROGRAM")
    print("=" * 70)
    
    # Initialize sentiment analyzer
    sia = initialize_sentiment_analyzer()
    
    # Step 1: Analyze all documents
    results = analyze_all_documents(SAMPLE_DOCUMENTS, sia)
    
    # Step 2: Visualize sentiment scores
    df = visualize_sentiment_scores(results)
    
    # Step 3: Sentence-level analysis (pick first positive review)
    sample_doc_text = SAMPLE_DOCUMENTS["Movie Review 1"]
    sentence_sentiments = analyze_sentence_level_sentiment(sample_doc_text, sia)
    visualize_sentence_sentiments(sentence_sentiments)
    
    # Step 4: Summary statistics
    generate_summary_statistics(df)
    
    # Step 5: Custom text analysis
    analyze_custom_text(sia)
    
    # Final summary
    print_final_summary()


if __name__ == "__main__":
    main()
