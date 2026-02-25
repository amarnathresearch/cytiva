"""
User-User Collaborative Filtering Recommender System
This script implements a user-user collaborative filtering recommender system
using cosine similarity to find similar users and recommend movies/items.
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


def calculate_similarity_matrix(df):
    """Calculate cosine similarity between users"""
    print("\n" + "=" * 70)
    print("STEP 2: CALCULATE USER SIMILARITY MATRIX")
    print("=" * 70)
    
    # Transpose to get user-movie matrix
    user_movie_matrix = df.T
    
    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(user_movie_matrix)
    
    # Create similarity DataFrame
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=user_movie_matrix.index,
        columns=user_movie_matrix.index
    )
    
    print(f"\nUser Similarity Matrix (Cosine Similarity):")
    print(similarity_df)
    print(f"\nSimilarity value range: [{similarity_df.values.min():.4f}, {similarity_df.values.max():.4f}]")
    
    return similarity_df, user_movie_matrix


def visualize_similarity_matrix(similarity_df):
    """Visualize user similarity matrix as heatmap"""
    print("\n" + "=" * 70)
    print("STEP 3: VISUALIZE USER SIMILARITY MATRIX")
    print("=" * 70)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_df, annot=True, fmt='.2f', cmap='RdYlGn', 
                cbar_kws={'label': 'Cosine Similarity'}, 
                annot_kws={'size': 9, 'weight': 'bold'})
    plt.title('User-User Similarity Matrix\n(Cosine Similarity)', fontsize=14, fontweight='bold')
    plt.xlabel('User', fontsize=12, fontweight='bold')
    plt.ylabel('User', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()
    print("✓ Similarity matrix visualization displayed")


def get_similar_users(target_user, similarity_df, n_similar=3):
    """Get most similar users to target user"""
    print("\n" + "=" * 70)
    print(f"STEP 4: FIND SIMILAR USERS TO {target_user}")
    print("=" * 70)
    
    # Get similarity scores for target user
    similarities = similarity_df[target_user].sort_values(ascending=False)
    
    # Exclude the target user itself (similarity = 1.0)
    similar_users = similarities[1:n_similar+1]
    
    print(f"\nTop {n_similar} users most similar to {target_user}:")
    for user, score in similar_users.items():
        print(f"  {user}: {score:.4f}")
    
    return similar_users.index.tolist(), similar_users.values


def recommend_movies(target_user, df, similarity_df, n_recommendations=3, n_similar=3):
    """Recommend movies to target user based on similar users"""
    print("\n" + "=" * 70)
    print(f"STEP 5: RECOMMEND MOVIES TO {target_user}")
    print("=" * 70)
    
    # Get similar users
    similar_users, sim_scores = get_similar_users(target_user, similarity_df, n_similar)
    
    # Get movies not rated by target user
    target_user_ratings = df[target_user]
    unrated_movies = target_user_ratings[target_user_ratings == 0].index.tolist()
    
    print(f"\nMovies not yet rated by {target_user}:")
    for movie in unrated_movies:
        print(f"  - {movie}")
    
    # Calculate recommendation scores
    recommendations = {}
    
    for movie in unrated_movies:
        # Get ratings from similar users for this movie
        similar_users_ratings = []
        similarities = []
        
        for similar_user in similar_users:
            rating = df.loc[movie, similar_user]
            if rating > 0:  # Only consider rated movies
                similar_users_ratings.append(rating)
                similarity_score = similarity_df.loc[target_user, similar_user]
                similarities.append(similarity_score)
        
        # Calculate weighted average rating
        if similar_users_ratings:
            weighted_rating = np.average(similar_users_ratings, weights=similarities)
            recommendations[movie] = weighted_rating
    
    # Sort by predicted rating
    recommendations = dict(sorted(recommendations.items(), key=lambda x: x[1], reverse=True))
    
    print(f"\n{'Recommended Movies for ' + target_user}:")
    print("=" * 70)
    if recommendations:
        for i, (movie, score) in enumerate(list(recommendations.items())[:n_recommendations], 1):
            print(f"{i}. {movie}: {score:.2f}/5.0")
    else:
        print("No recommendations available")
    
    return recommendations


def visualize_ratings_comparison(df, target_user, recommendations):
    """Visualize target user ratings vs recommended movies"""
    print("\n" + "=" * 70)
    print(f"STEP 6: VISUALIZE {target_user} RATINGS")
    print("=" * 70)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Target user's current ratings
    target_ratings = df[target_user].sort_values(ascending=False)
    colors1 = ['green' if x > 0 else 'lightgray' for x in target_ratings.values]
    
    axes[0].barh(range(len(target_ratings)), target_ratings.values, color=colors1)
    axes[0].set_yticks(range(len(target_ratings)))
    axes[0].set_yticklabels(target_ratings.index)
    axes[0].set_xlabel('Rating', fontsize=12, fontweight='bold')
    axes[0].set_title(f'{target_user}\'s Ratings\n(Green=Rated, Gray=Unrated)', 
                     fontsize=13, fontweight='bold')
    axes[0].set_xlim(0, 5)
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, v in enumerate(target_ratings.values):
        if v > 0:
            axes[0].text(v + 0.1, i, f'{v:.1f}', va='center', fontweight='bold')
    
    # Plot 2: Recommended movies
    if recommendations:
        rec_movies = list(recommendations.keys())[:5]
        rec_scores = list(recommendations.values())[:5]
        colors2 = ['red' if x < 2.5 else 'orange' if x < 3.5 else 'yellow' if x < 4.0 else 'lightgreen' 
                  for x in rec_scores]
        
        axes[1].barh(range(len(rec_movies)), rec_scores, color=colors2)
        axes[1].set_yticks(range(len(rec_movies)))
        axes[1].set_yticklabels(rec_movies)
        axes[1].set_xlabel('Predicted Rating', fontsize=12, fontweight='bold')
        axes[1].set_title(f'Top Recommended Movies for {target_user}', fontsize=13, fontweight='bold')
        axes[1].set_xlim(0, 5)
        axes[1].grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, v in enumerate(rec_scores):
            axes[1].text(v + 0.1, i, f'{v:.2f}', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    print("✓ Rating comparison visualization displayed")


def visualize_movies_heatmap(df):
    """Visualize movie ratings heatmap"""
    print("\n" + "=" * 70)
    print("STEP 7: VISUALIZE MOVIE RATINGS HEATMAP")
    print("=" * 70)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(df, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Rating'},
                annot_kws={'size': 11, 'weight': 'bold'})
    plt.title('Movie Ratings Heatmap\n(Users × Movies)', fontsize=14, fontweight='bold')
    plt.xlabel('User', fontsize=12, fontweight='bold')
    plt.ylabel('Movie', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()
    print("✓ Movie ratings heatmap displayed")


def generate_all_recommendations(df, similarity_df):
    """Generate recommendations for all users"""
    print("\n" + "=" * 70)
    print("STEP 8: GENERATE RECOMMENDATIONS FOR ALL USERS")
    print("=" * 70)
    
    all_recommendations = {}
    
    for user in df.columns:
        recommendations = recommend_movies(user, df, similarity_df, n_recommendations=3, n_similar=3)
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
    
    print(f"\nRecommendation Statistics:")
    total_recommendations = sum(len(recs) for recs in all_recommendations.values())
    print(f"  Total recommendations generated: {total_recommendations}")
    print(f"  Users with recommendations: {len([u for u, r in all_recommendations.items() if r])}")
    
    print("\n✓ User-User Collaborative Filtering Complete!")
    print("=" * 70)


def main():
    """Main function to run the recommender system"""
    print("\n" + "=" * 70)
    print("USER-USER COLLABORATIVE FILTERING RECOMMENDER SYSTEM")
    print("=" * 70)
    
    # Step 1: Create sample dataset
    df, movies = create_sample_dataset()
    
    # Step 2: Calculate similarity matrix
    similarity_df, user_movie_matrix = calculate_similarity_matrix(df)
    
    # Step 3: Visualize similarity matrix
    visualize_similarity_matrix(similarity_df)
    
    # Step 4 & 5: Recommend movies for specific user
    target_user = 'User1'
    recommendations = recommend_movies(target_user, df, similarity_df, n_recommendations=3, n_similar=3)
    
    # Step 6: Visualize ratings
    visualize_ratings_comparison(df, target_user, recommendations)
    
    # Step 7: Visualize movies heatmap
    visualize_movies_heatmap(df)
    
    # Step 8: Generate recommendations for all users
    all_recommendations = generate_all_recommendations(df, similarity_df)
    
    # Step 9: Print summary
    print_summary(df, all_recommendations)
    
    # Additional: Detailed recommendation for each user
    print("\n" + "=" * 70)
    print("DETAILED RECOMMENDATIONS FOR EACH USER")
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
