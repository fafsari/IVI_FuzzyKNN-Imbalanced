import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances

# Step 1: SMOTE Implementation
def smote(X, y, minority_class, k=5):
    """
    Synthetic Minority Oversampling Technique (SMOTE)
    Args:
        X: Input features (numpy array)
        y: Labels (numpy array)
        minority_class: Class to oversample
        k: Number of nearest neighbors
    Returns:
        X_resampled, y_resampled: Oversampled dataset
    """
    X_minority = X[y == minority_class]
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_minority)
    synthetic_samples = []
    
    for x in X_minority:
        neighbors = nbrs.kneighbors([x], return_distance=False)[0]
        for neighbor in neighbors:
            diff = X_minority[neighbor] - x
            synthetic = x + np.random.rand() * diff
            synthetic_samples.append(synthetic)

    X_synthetic = np.array(synthetic_samples)
    y_synthetic = np.full(len(synthetic_samples), minority_class)

    return np.vstack([X, X_synthetic]), np.hstack([y, y_synthetic])


# Step 2: Interval-Valued Intuitionistic Fuzzy KNN (IVIF-KNN)
def calculate_ivif_memberships(X, y, k_values, lambda_param, alpha_param):
    """
    Calculate interval-valued memberships and non-memberships for IVIF-KNN.
    Args:
        X: Input features (numpy array)
        y: Labels (numpy array)
        k_values: Range of k values for neighbors
        lambda_param: Lambda parameter for membership computation
        alpha_param: Alpha parameter for non-membership computation
    Returns:
        memberships, non_memberships: Membership and non-membership intervals
    """
    memberships = []
    non_memberships = []
    
    for k in k_values:
        nbrs = NearestNeighbors(n_neighbors=k).fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        for idx, neighbors in enumerate(indices):
            fuzzy_memberships = []
            for neighbor in neighbors:
                class_counts = {label: 0 for label in np.unique(y)}
                for n in neighbors:
                    class_counts[y[n]] += 1
                membership = class_counts[y[neighbor]] / k
                fuzzy_memberships.append(0.51 + (membership * 0.49) if y[neighbor] == y[idx] else membership * 0.49)
            
            # Eq. (4): Interval-valued fuzzy membership
            membership_interval = [
                min(fuzzy_memberships),
                max(fuzzy_memberships)
            ]

            # Eq. (10): Interval-valued membership degrees
            membership_degree = [
                1 - (1 - membership_interval[0])**(lambda_param - 1),
                1 - (1 - membership_interval[1])**(lambda_param - 1)
            ]
            
            # Eq. (12): Interval-valued intuitionistic fuzzy non-membership degrees
            non_membership_degree = [
                (1 - membership_interval[1])**((lambda_param + alpha_param) * (lambda_param - 1)),
                (1 - membership_interval[1])**(lambda_param * (lambda_param - 1))
            ]

            memberships.append(membership_degree)
            non_memberships.append(non_membership_degree)
    
    return np.array(memberships), np.array(non_memberships)

# Step 3: Voting Function
def voting_function(memberships, non_memberships, Urej, Uacc, Nrej, Nacc):
    """
    Voting function based on membership and non-membership intervals.
    Args:
        memberships: Interval-valued memberships
        non_memberships: Interval-valued non-memberships
        Urej, Uacc, Nrej, Nacc: Decision thresholds
    Returns:
        votes: Voting results for each sample
    """
    votes = []
    for m, n in zip(memberships, non_memberships):
        # Lexicographic ordering for vote 1
        if (m[0] > Urej or (m[0] == Urej and m[1] >= Uacc)) and \
           (n[0] < Nacc or (n[0] == Nacc and n[1] <= Nrej)):
            votes.append(1)  # Accept
        # Lexicographic ordering for vote -1
        elif (m[0] < Urej or (m[0] == Urej and m[1] < Uacc)) and \
             (n[0] > Nacc or (n[0] == Nacc and n[1] > Nrej)):
            votes.append(-1)  # Reject
        else:
            votes.append(0)  # Refuse to decide
    return np.array(votes)

# Step 4: Iterative Partitioning Filter (IPF)
def iterative_partitioning_filter(X, y, n_partitions, max_iterations, stop_threshold, Urej, Uacc, Nrej, Nacc):
    """
    Iterative Partitioning Filter for removing noisy and borderline samples.
    Args:
        X, y: Input data and labels
        n_partitions: Number of partitions for filtering
        max_iterations: Maximum iterations for filtering
        stop_threshold: Stop threshold for removal
        Urej, Uacc, Nrej, Nacc: Decision thresholds
    Returns:
        Filtered X and y
    """
    for iteration in range(max_iterations):
        noisy_samples = []
        partitions = np.array_split(range(len(X)), n_partitions)

        for partition in partitions:
            memberships, non_memberships = calculate_ivif_memberships(
                X[partition], y[partition], k_values=5, lambda_param=2, alpha_param=1)

            votes = voting_function(memberships, non_memberships, Urej, Uacc, Nrej, Nacc)
            noisy_samples.extend(partition[votes == -1])

        if len(noisy_samples) / len(X) < stop_threshold:
            break
        X = np.delete(X, noisy_samples, axis=0)
        y = np.delete(y, noisy_samples, axis=0)

    return X, y

# Main Function
def preprocess_data(X, y):
    """
    Preprocess data using SMOTE and IVIF-KNN-based IPF.
    Args:
        X: Input features
        y: Labels
    Returns:
        Filtered X and y
    """
    # Step 1: Oversample data using SMOTE
    X_resampled, y_resampled = smote(X, y, minority_class=1)

    # Step 2: Apply Iterative Partitioning Filter
    X_filtered, y_filtered = iterative_partitioning_filter(
        X_resampled, y_resampled,
        n_partitions=9, max_iterations=3, stop_threshold=0.01,
        Urej=0.4, Uacc=0.7, Nrej=0.7, Nacc=0.4
    )

    return X_filtered, y_filtered
