"""
SVD (Singular Value Decomposition) Recommender System
This script implements a matrix factorization recommender system using SVD
to decompose the user-item rating matrix and generate recommendations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
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


def handle_sparsity(df):
    """Handle sparse matrix by filling zeros with global mean"""
    print("\n" + "=" * 70)
    print("STEP 2: HANDLE SPARSITY - FILL WITH GLOBAL MEAN")
    print("=" * 70)
    
    # Convert to float to avoid type issues
    df_numeric = df.astype(float)
    
    # Replace any existing NaN with 0
    df_numeric = df_numeric.fillna(0)
    
    # Calculate global mean of non-zero rated items only
    global_mean = df_numeric[df_numeric > 0].values.mean()
    
    # Create a copy and fill zeros with global mean
    df_filled = df_numeric.copy()
    df_filled[df_filled == 0] = global_mean
    
    # Ensure no NaN values remain (double-check)
    df_filled = df_filled.fillna(global_mean)
    
    # Verify no NaN values
    if df_filled.isna().any().any():
        print("Warning: NaN values still present, filling with mean")
        df_filled = df_filled.fillna(global_mean)
    
    print(f"\nGlobal mean of ratings: {global_mean:.4f}")
    print(f"\nFilled Rating Matrix:")
    print(df_filled)
    
    return df_filled, global_mean


def apply_svd(df_filled, n_components=3):
    """Apply SVD decomposition to rating matrix"""
    print("\n" + "=" * 70)
    print(f"STEP 3: APPLY SVD DECOMPOSITION (n_components={n_components})")
    print("=" * 70)
    
    # Verify no NaN values before SVD
    if df_filled.isna().any().any():
        print("Warning: NaN values detected, removing them")
        df_filled = df_filled.fillna(df_filled.mean().mean())
    
    # Transpose to get (users, movies) matrix
    # SVD will decompose R = U * Σ * V^T
    R = df_filled.T.values  # (8 users, 8 movies)
    
    # Verify no NaN in the matrix
    if np.isnan(R).any():
        print("Error: NaN values in matrix R")
        R = np.nan_to_num(R, nan=0.0)
    
    # Apply Truncated SVD
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    U = svd.fit_transform(R)  # (n_users, n_components)
    V = svd.components_.T  # (n_movies, n_components)
    singular_values = svd.singular_values_
    
    print(f"\nMatrix dimensions:")
    print(f"  Original matrix R: {R.shape}")
    print(f"  U matrix (User factors): {U.shape}")
    print(f"  V matrix (Item factors): {V.shape}")
    print(f"  Singular values: {singular_values.shape}")
    
    print(f"\nSingular Values:")
    for i, sv in enumerate(singular_values):
        print(f"  σ{i+1}: {sv:.4f}")
    
    print(f"\nExplained variance ratio:")
    explained_variance_ratio = svd.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    for i, (var, cum_var) in enumerate(zip(explained_variance_ratio, cumulative_variance)):
        print(f"  Component {i+1}: {var:.4f} ({var*100:.2f}%) | Cumulative: {cum_var:.4f} ({cum_var*100:.2f}%)")
    
    return svd, U, V, singular_values, R


def reconstruct_matrix(U, V, singular_values):
    """Reconstruct the rating matrix using SVD components"""
    print("\n" + "=" * 70)
    print("STEP 4: RECONSTRUCT MATRIX FROM SVD COMPONENTS")
    print("=" * 70)
    
    # Σ is a diagonal matrix, so we multiply singular values with V
    Sigma_V = V * singular_values
    R_reconstructed = U @ Sigma_V.T
    
    print(f"\nReconstructed matrix shape: {R_reconstructed.shape}")
    print(f"Reconstructed matrix (first 4 users, all movies):")
    print(R_reconstructed[:4, :])
    
    return R_reconstructed


def visualize_singular_values(singular_values, explained_variance_ratio):
    """Visualize singular values and explained variance"""
    print("\n" + "=" * 70)
    print("STEP 5: VISUALIZE SINGULAR VALUES AND VARIANCE")
    print("=" * 70)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot singular values
    axes[0].plot(range(1, len(singular_values)+1), singular_values, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Component', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Singular Value', fontsize=12, fontweight='bold')
    axes[0].set_title('Singular Values from SVD', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    for i, sv in enumerate(singular_values):
        axes[0].text(i+1, sv+0.05, f'{sv:.2f}', ha='center', fontweight='bold')
    
    # Plot explained variance
    cumulative_variance = np.cumsum(explained_variance_ratio)
    axes[1].bar(range(1, len(explained_variance_ratio)+1), explained_variance_ratio*100, 
                alpha=0.7, label='Individual', color='steelblue')
    axes[1].plot(range(1, len(cumulative_variance)+1), cumulative_variance*100, 
                'ro-', linewidth=2, markersize=8, label='Cumulative')
    axes[1].set_xlabel('Component', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Explained Variance (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Explained Variance by Component', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    print("✓ Singular values visualization displayed")


def visualize_factor_matrices(U, V, singular_values):
    """Visualize U and V factor matrices"""
    print("\n" + "=" * 70)
    print("STEP 6: VISUALIZE FACTOR MATRICES (U AND V)")
    print("=" * 70)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # U matrix (User factors)
    user_labels = [f'User{i+1}' for i in range(U.shape[0])]
    sns.heatmap(U, annot=True, fmt='.2f', cmap='RdBu_r', cbar_kws={'label': 'Factor Value'},
                xticklabels=[f'Factor{i+1}' for i in range(U.shape[1])],
                yticklabels=user_labels, ax=axes[0],
                annot_kws={'size': 10, 'weight': 'bold'})
    axes[0].set_title('U Matrix (User Latent Factors)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Latent Factors', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Users', fontsize=11, fontweight='bold')
    
    # V matrix (Item factors)
    movie_labels = ['Inception', 'Interstellar', 'Dark Knight', 'Avengers', 
                   'Avatar', 'Titanic', 'Gladiator', 'Matrix']
    sns.heatmap(V, annot=True, fmt='.2f', cmap='RdBu_r', cbar_kws={'label': 'Factor Value'},
                xticklabels=[f'Factor{i+1}' for i in range(V.shape[1])],
                yticklabels=movie_labels, ax=axes[1],
                annot_kws={'size': 10, 'weight': 'bold'})
    axes[1].set_title('V Matrix (Movie Latent Factors)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Latent Factors', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Movies', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    print("✓ Factor matrices visualization displayed")


def get_recommendations_svd(target_user, original_df, R_reconstructed, n_recommendations=3):
    """Get recommendations from reconstructed matrix"""
    print("\n" + "=" * 70)
    print(f"STEP 7: GET RECOMMENDATIONS FOR {target_user} USING SVD")
    print("=" * 70)
    
    # Get user index
    user_columns = list(original_df.columns)
    user_idx = user_columns.index(target_user)
    
    # Get user's reconstructed ratings
    user_reconstructed_ratings = R_reconstructed[user_idx]
    
    # Get movies already rated by user
    user_original_ratings = original_df[target_user].values
    
    # Only recommend movies with 0 rating (not watched)
    unrated_mask = user_original_ratings == 0
    
    # Get predicted ratings for unrated movies
    predicted_ratings = user_reconstructed_ratings[unrated_mask]
    unrated_movies = original_df.index[unrated_mask]
    
    # Create recommendations
    recommendations = dict(zip(unrated_movies, predicted_ratings))
    recommendations = dict(sorted(recommendations.items(), key=lambda x: x[1], reverse=True))
    
    print(f"\nMovies already rated by {target_user}:")
    for movie, rating in original_df[target_user][original_df[target_user] > 0].items():
        print(f"  {movie}: {rating}/5")
    
    print(f"\nTop {n_recommendations} Recommended Movies:")
    for i, (movie, score) in enumerate(list(recommendations.items())[:n_recommendations], 1):
        print(f"{i}. {movie}: {score:.2f}/5.0 (predicted)")
    
    return recommendations


def visualize_svd_recommendations(original_df, target_user, R_reconstructed, recommendations):
    """Visualize SVD-based recommendations"""
    print("\n" + "=" * 70)
    print(f"STEP 8: VISUALIZE SVD RECOMMENDATIONS FOR {target_user}")
    print("=" * 70)
    
    user_columns = list(original_df.columns)
    user_idx = user_columns.index(target_user)
    
    # Get user's original and reconstructed ratings
    original_ratings = original_df[target_user].values
    reconstructed_ratings = R_reconstructed[user_idx]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Original ratings
    movie_list = original_df.index
    rated_mask = original_ratings > 0
    rated_indices = np.where(rated_mask)[0]
    
    axes[0, 0].barh(range(len(rated_indices)), original_ratings[rated_mask], color='steelblue')
    axes[0, 0].set_yticks(range(len(rated_indices)))
    axes[0, 0].set_yticklabels(movie_list[rated_mask])
    axes[0, 0].set_xlabel('Rating', fontsize=11, fontweight='bold')
    axes[0, 0].set_title(f'{target_user}\'s Rated Movies', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlim(0, 5)
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    for i, v in enumerate(original_ratings[rated_mask]):
        axes[0, 0].text(v + 0.1, i, f'{v:.1f}', va='center', fontweight='bold')
    
    # Plot 2: Recommended movies
    if recommendations:
        rec_movies = list(recommendations.keys())[:5]
        rec_scores = list(recommendations.values())[:5]
        colors = ['red' if x < 2.5 else 'orange' if x < 3.5 else 'yellow' if x < 4.0 else 'lightgreen' 
                 for x in rec_scores]
        
        axes[0, 1].barh(range(len(rec_movies)), rec_scores, color=colors)
        axes[0, 1].set_yticks(range(len(rec_movies)))
        axes[0, 1].set_yticklabels(rec_movies)
        axes[0, 1].set_xlabel('Predicted Rating', fontsize=11, fontweight='bold')
        axes[0, 1].set_title(f'SVD Recommended Movies for {target_user}', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlim(0, 5)
        axes[0, 1].grid(True, alpha=0.3, axis='x')
        
        for i, v in enumerate(rec_scores):
            axes[0, 1].text(v + 0.1, i, f'{v:.2f}', va='center', fontweight='bold')
    
    # Plot 3: Comparison of original vs reconstructed for rated movies
    axes[1, 0].scatter(original_ratings[rated_mask], reconstructed_ratings[rated_mask], 
                      s=200, alpha=0.6, color='steelblue', edgecolors='black', linewidth=2)
    axes[1, 0].plot([0, 5], [0, 5], 'r--', linewidth=2, label='Perfect prediction')
    axes[1, 0].set_xlabel('Original Rating', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Reconstructed Rating', fontsize=11, fontweight='bold')
    axes[1, 0].set_title(f'Original vs Reconstructed Ratings for {target_user}', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].set_xlim(0, 5)
    axes[1, 0].set_ylim(0, 5)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: All movies with original and reconstructed ratings
    all_movies = original_df.index
    x_pos = np.arange(len(all_movies))
    width = 0.35
    
    axes[1, 1].bar(x_pos - width/2, original_ratings, width, label='Original', alpha=0.8, color='steelblue')
    axes[1, 1].bar(x_pos + width/2, reconstructed_ratings, width, label='Reconstructed', alpha=0.8, color='coral')
    axes[1, 1].set_ylabel('Rating', fontsize=11, fontweight='bold')
    axes[1, 1].set_title(f'Original vs Reconstructed Ratings - All Movies', fontsize=12, fontweight='bold')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(all_movies, rotation=45, ha='right')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].set_ylim(0, 5)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    print("✓ SVD recommendations visualization displayed")


def calculate_reconstruction_error(original_df, R_reconstructed):
    """Calculate RMSE between original and reconstructed matrix"""
    print("\n" + "=" * 70)
    print("STEP 9: CALCULATE RECONSTRUCTION ERROR")
    print("=" * 70)
    
    # Calculate RMSE only on originally rated items (where original_df > 0)
    user_indices, movie_indices = np.where(original_df.values > 0)
    original_rated_values = original_df.values[user_indices, movie_indices]
    reconstructed_values = R_reconstructed[user_indices, movie_indices]
    
    rmse = np.sqrt(np.mean((original_rated_values - reconstructed_values) ** 2))
    mae = np.mean(np.abs(original_rated_values - reconstructed_values))
    
    print(f"\nReconstruction Error Metrics:")
    print(f"  RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"  MAE (Mean Absolute Error): {mae:.4f}")
    
    return rmse, mae


def print_summary(original_df, singular_values, rmse, mae):
    """Print summary statistics"""
    print("\n" + "=" * 70)
    print("FINAL SUMMARY - SVD RECOMMENDER SYSTEM")
    print("=" * 70)
    print(f"\nDataset Statistics:")
    print(f"  Total users: {original_df.shape[1]}")
    print(f"  Total movies: {original_df.shape[0]}")
    print(f"  Total ratings: {(original_df > 0).sum().sum()}")
    print(f"  Sparsity: {(original_df == 0).sum().sum() / (original_df.shape[0] * original_df.shape[1]) * 100:.2f}%")
    
    print(f"\nSVD Decomposition:")
    print(f"  Number of components: {len(singular_values)}")
    print(f"  Largest singular value: {singular_values[0]:.4f}")
    print(f"  Smallest singular value: {singular_values[-1]:.4f}")
    
    print(f"\nReconstruction Performance:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    
    print("\n✓ SVD Recommender System Complete!")
    print("=" * 70)


def main():
    """Main function to run SVD recommender system"""
    print("\n" + "=" * 70)
    print("SVD (SINGULAR VALUE DECOMPOSITION) RECOMMENDER SYSTEM")
    print("=" * 70)
    
    # Step 1: Create sample dataset
    df, movies = create_sample_dataset()
    
    # Step 2: Handle sparsity
    df_filled, global_mean = handle_sparsity(df)
    
    # Step 3: Apply SVD
    n_components = 3
    svd, U, V, singular_values, R = apply_svd(df_filled, n_components=n_components)
    
    # Step 4: Reconstruct matrix
    R_reconstructed = reconstruct_matrix(U, V, singular_values)
    
    # Step 5: Visualize singular values
    visualize_singular_values(singular_values, svd.explained_variance_ratio_)
    
    # Step 6: Visualize factor matrices
    visualize_factor_matrices(U, V, singular_values)
    
    # Step 7: Get recommendations
    target_user = 'User1'
    recommendations = get_recommendations_svd(target_user, df, R_reconstructed, n_recommendations=3)
    
    # Step 8: Visualize recommendations
    visualize_svd_recommendations(df, target_user, R_reconstructed, recommendations)
    
    # Step 9: Calculate reconstruction error
    rmse, mae = calculate_reconstruction_error(df, R_reconstructed)
    
    # Print summary
    print_summary(df, singular_values, rmse, mae)
    
    # Generate recommendations for all users
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS FOR ALL USERS")
    print("=" * 70)
    for user in df.columns:
        user_recommendations = get_recommendations_svd(user, df, R_reconstructed, n_recommendations=3)
        print()


if __name__ == "__main__":
    main()
