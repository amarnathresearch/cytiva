"""
Item-Item Collaborative Filtering Recommender System
This script implements an item-item collaborative filtering recommender system
using cosine similarity to find similar items and recommend movies/items.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def create_sample_dataset():
    """Create a sample movie rating dataset"""
    print("\n" + "=" * 70)
    print("STEP 1: CREATE SAMPLE MOVIE RATING DATASET")
    print("=" * 70)
    
    # Sample data: Users rating movies (0-5 scale, 0 means not watched/rated)
    data = {
        'User1': [5, 4, 0, 1, 0, 5, 4, 0],
        'User2': [4, 0, 4, 2, 0, 5, 0, 3],
        'User3': [5, 5, 0, 0, 1, 5, 4, 0],
        'User4': [0, 0, 5, 5, 5, 0, 1, 4],
        'User5': [1, 0, 4, 5, 4, 0, 2, 5],
        'User6': [5, 4, 0, 2, 0, 5, 3, 0],
        'User7': [0, 0, 5, 4, 5, 1, 0, 4],
        'User8': [4, 5, 0, 1, 0, 5, 4, 0],
    }
    
    movies = ['Inception', 'Interstellar', 'The Dark Knight', 'Avengers', 
              'Avatar', 'Titanic', 'Gladiator', 'The Matrix']
    
    # Create DataFrame
    df = pd.DataFrame(data, index=movies)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Number of users: {df.shape[1]}")
    print(f"Number of movies: {df.shape[0]}")
    print(f"\nMovie Rating Matrix:")
    print(df)
    print(f"\nSparsity: {(df == 0).sum().sum() / (df.shape[0] * df.shape[1]) * 100:.2f}%")
    
    return df, movies


def calculate_item_similarity_matrix(df):
    """Calculate cosine similarity between items (movies)"""
    print("\n" + "=" * 70)
    print("STEP 2: CALCULATE ITEM-ITEM SIMILARITY MATRIX")
    print("=" * 70)
    
    # Use the movie-user matrix (items as rows)
    # Each row represents a movie's rating vector across all users
    movie_rating_matrix = df
    
    # Calculate cosine similarity between movies
    similarity_matrix = cosine_similarity(movie_rating_matrix)
    
    # Create similarity DataFrame
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=movie_rating_matrix.index,
        columns=movie_rating_matrix.index
    )
    
    print(f"\nItem-Item Similarity Matrix (Cosine Similarity):")
    print(similarity_df)
    print(f"\nSimilarity value range: [{similarity_df.values.min():.4f}, {similarity_df.values.max():.4f}]")
    
    return similarity_df, movie_rating_matrix


def visualize_item_similarity_matrix(similarity_df):
    """Visualize item similarity matrix as heatmap"""
    print("\n" + "=" * 70)
    print("STEP 3: VISUALIZE ITEM-ITEM SIMILARITY MATRIX")
    print("=" * 70)
    
    plt.figure(figsize=(11, 9))
    sns.heatmap(similarity_df, annot=True, fmt='.2f', cmap='RdYlGn', 
                cbar_kws={'label': 'Cosine Similarity'}, 
                annot_kws={'size': 9, 'weight': 'bold'})
    plt.title('Item-Item Similarity Matrix\n(Movie-Movie Similarity)', fontsize=14, fontweight='bold')
    plt.xlabel('Movie', fontsize=12, fontweight='bold')
    plt.ylabel('Movie', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    print("✓ Item similarity matrix visualization displayed")


def get_similar_items(target_item, similarity_df, n_similar=3):
    """Get most similar items to target item"""
    print("\n" + "=" * 70)
    print(f"STEP 4: FIND MOVIES SIMILAR TO '{target_item}'")
    print("=" * 70)
    
    # Get similarity scores for target item
    similarities = similarity_df[target_item].sort_values(ascending=False)
    
    # Exclude the target item itself (similarity = 1.0)
    similar_items = similarities[1:n_similar+1]
    
    print(f"\nTop {n_similar} movies similar to '{target_item}':")
    for movie, score in similar_items.items():
        print(f"  {movie}: {score:.4f}")
    
    return similar_items.index.tolist(), similar_items.values


def recommend_items_for_user(target_user, df, similarity_df, n_recommendations=3):
    """Recommend items to user based on similar items they rated"""
    print("\n" + "=" * 70)
    print(f"STEP 5: RECOMMEND MOVIES TO {target_user} (Item-Item based)")
    print("=" * 70)
    
    # Get movies rated by target user
    user_ratings = df[target_user]
    rated_movies = user_ratings[user_ratings > 0]
    unrated_movies = user_ratings[user_ratings == 0].index.tolist()
    
    print(f"\nMovies already rated by {target_user}:")
    for movie, rating in rated_movies.items():
        print(f"  {movie}: {rating}/5")
    
    print(f"\nMovies not yet rated by {target_user}:")
    for movie in unrated_movies:
        print(f"  - {movie}")
    
    # Calculate recommendation scores based on similar items
    recommendations = {}
    
    for unrated_movie in unrated_movies:
        recommendation_scores = []
        weights = []
        
        # For each rated movie, find similarity to unrated movie
        for rated_movie, user_rating in rated_movies.items():
            similarity_score = similarity_df.loc[rated_movie, unrated_movie]
            
            if similarity_score > 0:  # Only consider positive similarities
                recommendation_scores.append(user_rating)
                weights.append(similarity_score)
        
        # Calculate weighted average
        if recommendation_scores:
            weighted_score = np.average(recommendation_scores, weights=weights)
            recommendations[unrated_movie] = weighted_score
    
    # Sort by predicted rating
    recommendations = dict(sorted(recommendations.items(), key=lambda x: x[1], reverse=True))
    
    print(f"\n{'Recommended Movies for ' + target_user}:")
    print("=" * 70)
    if recommendations:
        for i, (movie, score) in enumerate(list(recommendations.items())[:n_recommendations], 1):
            print(f"{i}. {movie}: {score:.2f}/5.0")
    else:
        print("No recommendations available")
    
    return recommendations, rated_movies


def find_movies_similar_to_rated(target_user, df, similarity_df, n_similar=2):
    """Find similar movies to items the user already rated"""
    print("\n" + "=" * 70)
    print(f"STEP 6: SIMILAR MOVIES FOR {target_user}'s RATED MOVIES")
    print("=" * 70)
    
    user_ratings = df[target_user]
    rated_movies = user_ratings[user_ratings > 0]
    
    similar_to_rated = {}
    
    for rated_movie in rated_movies.index:
        similarities = similarity_df[rated_movie].sort_values(ascending=False)
        # Get similar movies (excluding the movie itself)
        similar_movies = similarities[1:n_similar+1]
        similar_to_rated[rated_movie] = {
            'rating': user_ratings[rated_movie],
            'similar_movies': dict(similar_movies)
        }
    
    print(f"\nMovies similar to {target_user}'s rated movies:")
    for rated_movie, data in similar_to_rated.items():
        print(f"\nBased on '{rated_movie}' (rated {data['rating']}/5):")
        for similar_movie, similarity_score in data['similar_movies'].items():
            print(f"  - {similar_movie} (similarity: {similarity_score:.2f})")
    
    return similar_to_rated


def visualize_item_recommendations(df, target_user, recommendations, rated_movies):
    """Visualize item-based recommendations"""
    print("\n" + "=" * 70)
    print(f"STEP 7: VISUALIZE RECOMMENDATIONS FOR {target_user}")
    print("=" * 70)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Target user's rated movies
    rated_indices = df[target_user][df[target_user] > 0].index.tolist()
    rated_values = df[target_user][df[target_user] > 0].values
    
    axes[0].barh(range(len(rated_indices)), rated_values, color='steelblue')
    axes[0].set_yticks(range(len(rated_indices)))
    axes[0].set_yticklabels(rated_indices)
    axes[0].set_xlabel('Rating', fontsize=12, fontweight='bold')
    axes[0].set_title(f'{target_user}\'s Rated Movies', fontsize=13, fontweight='bold')
    axes[0].set_xlim(0, 5)
    axes[0].grid(True, alpha=0.3, axis='x')
    
    for i, v in enumerate(rated_values):
        axes[0].text(v + 0.1, i, f'{v:.1f}', va='center', fontweight='bold')
    
    # Plot 2: Recommended movies with similarity-based scores
    if recommendations:
        rec_movies = list(recommendations.keys())[:5]
        rec_scores = list(recommendations.values())[:5]
        colors = ['red' if x < 2.5 else 'orange' if x < 3.5 else 'yellow' if x < 4.0 else 'lightgreen' 
                 for x in rec_scores]
        
        axes[1].barh(range(len(rec_movies)), rec_scores, color=colors)
        axes[1].set_yticks(range(len(rec_movies)))
        axes[1].set_yticklabels(rec_movies)
        axes[1].set_xlabel('Predicted Rating (Item-Item)', fontsize=12, fontweight='bold')
        axes[1].set_title(f'Recommended Movies for {target_user}\n(Based on Similar Items)', 
                         fontsize=13, fontweight='bold')
        axes[1].set_xlim(0, 5)
        axes[1].grid(True, alpha=0.3, axis='x')
        
        for i, v in enumerate(rec_scores):
            axes[1].text(v + 0.1, i, f'{v:.2f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    print("✓ Recommendation visualization displayed")


def visualize_movies_heatmap(df):
    """Visualize movie ratings heatmap"""
    print("\n" + "=" * 70)
    print("STEP 8: VISUALIZE MOVIE RATINGS HEATMAP")
    print("=" * 70)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(df, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Rating'},
                annot_kws={'size': 11, 'weight': 'bold'})
    plt.title('Movie Ratings Heatmap\n(Movies × Users)', fontsize=14, fontweight='bold')
    plt.xlabel('User', fontsize=12, fontweight='bold')
    plt.ylabel('Movie', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()
    print("✓ Movie ratings heatmap displayed")


def generate_all_item_recommendations(df, similarity_df):
    """Generate item-based recommendations for all users"""
    print("\n" + "=" * 70)
    print("STEP 9: GENERATE ITEM-BASED RECOMMENDATIONS FOR ALL USERS")
    print("=" * 70)
    
    all_recommendations = {}
    
    for user in df.columns:
        recommendations, rated_movies = recommend_items_for_user(user, df, similarity_df, n_recommendations=3)
        all_recommendations[user] = recommendations
    
    return all_recommendations


def print_summary(df, all_recommendations):
    """Print summary statistics"""
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\nDataset Statistics:")
    print(f"  Total users: {df.shape[1]}")
    print(f"  Total movies: {df.shape[0]}")
    print(f"  Total ratings: {(df > 0).sum().sum()}")
    print(f"  Average rating per user: {(df > 0).sum().mean():.2f} movies")
    print(f"  Sparsity: {(df == 0).sum().sum() / (df.shape[0] * df.shape[1]) * 100:.2f}%")
    
    print(f"\nRecommendation Statistics (Item-Item Based):")
    total_recommendations = sum(len(recs) for recs in all_recommendations.values())
    print(f"  Total recommendations generated: {total_recommendations}")
    print(f"  Users with recommendations: {len([u for u, r in all_recommendations.items() if r])}")
    
    print("\n✓ Item-Item Collaborative Filtering Complete!")
    print("=" * 70)


def main():
    """Main function to run the item-item recommender system"""
    print("\n" + "=" * 70)
    print("ITEM-ITEM COLLABORATIVE FILTERING RECOMMENDER SYSTEM")
    print("=" * 70)
    
    # Step 1: Create sample dataset
    df, movies = create_sample_dataset()
    
    # Step 2: Calculate item similarity matrix
    similarity_df, movie_rating_matrix = calculate_item_similarity_matrix(df)
    
    # Step 3: Visualize item similarity matrix
    visualize_item_similarity_matrix(similarity_df)
    
    # Step 4: Find similar movies
    target_movie = 'Inception'
    similar_movies, sim_scores = get_similar_items(target_movie, similarity_df, n_similar=3)
    
    # Step 5 & 6: Recommend items for specific user
    target_user = 'User1'
    recommendations, rated_movies = recommend_items_for_user(target_user, df, similarity_df, n_recommendations=3)
    similar_to_rated = find_movies_similar_to_rated(target_user, df, similarity_df, n_similar=2)
    
    # Step 7: Visualize recommendations
    visualize_item_recommendations(df, target_user, recommendations, rated_movies)
    
    # Step 8: Visualize movies heatmap
    visualize_movies_heatmap(df)
    
    # Step 9: Generate recommendations for all users
    all_recommendations = generate_all_item_recommendations(df, similarity_df)
    
    # Step 10: Print summary
    print_summary(df, all_recommendations)
    
    # Additional: Detailed recommendation for each user
    print("\n" + "=" * 70)
    print("DETAILED ITEM-BASED RECOMMENDATIONS FOR EACH USER")
    print("=" * 70)
    for user, recs in all_recommendations.items():
        print(f"\n{user}:")
        if recs:
            for i, (movie, score) in enumerate(list(recs.items())[:3], 1):
                print(f"  {i}. {movie}: {score:.2f}/5.0")
        else:
            print("  No recommendations available")


if __name__ == "__main__":
    main()
