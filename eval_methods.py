import torch
from skdim.id import TwoNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from scipy.linalg import svd

import numpy as np


# mrc2 loss
class MaximalCodingRateReduction(torch.nn.Module):
    def __init__(self, gam1=1.0, gam2=1.0, eps=0.01):
        super(MaximalCodingRateReduction, self).__init__()
        self.gam1 = gam1
        self.gam2 = gam2
        self.eps = eps

    def compute_discrimn_loss_empirical(self, W):
        """Empirical Discriminative Loss."""
        p, m = W.shape
        I = torch.eye(p, device=W.device, dtype=W.dtype)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + self.gam1 * scalar * W.matmul(W.T))
        return logdet / 2.

    def compute_compress_loss_empirical(self, W, Pi):
        """Empirical Compressive Loss."""
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p, device=W.device, dtype=W.dtype)
        compress_loss = 0.
        for j in range(k):
            trPi = torch.trace(Pi[j]) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W.matmul(Pi[j]).matmul(W.T))
            compress_loss += log_det * trPi / m
        return compress_loss / 2.

    def compute_discrimn_loss_theoretical(self, W):
        """Theoretical Discriminative Loss."""
        p, m = W.shape
        I = torch.eye(p, device=W.device, dtype=W.dtype)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.

    def compute_compress_loss_theoretical(self, W, Pi):
        """Theoretical Compressive Loss."""
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p, device=W.device, dtype=W.dtype)
        compress_loss = 0.
        for j in range(k):
            trPi = torch.trace(Pi[j]) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W.matmul(Pi[j]).matmul(W.T))
            compress_loss += trPi / (2 * m) * log_det
        return compress_loss

    def forward(self, x, y, num_classes=None):
        if num_classes is None:
            num_classes = y.max() + 1
        W = x.T
        Pi = label_to_membership(y.cpu().numpy(), num_classes)
        Pi = torch.tensor(Pi, dtype=torch.float32, device=x.device)

        discrimn_loss_empi = self.compute_discrimn_loss_empirical(W)
        compress_loss_empi = self.compute_compress_loss_empirical(W, Pi)
        discrimn_loss_theo = self.compute_discrimn_loss_theoretical(W)
        compress_loss_theo = self.compute_compress_loss_theoretical(W, Pi)
 
        total_loss_empi = self.gam2 * -discrimn_loss_empi + compress_loss_empi
        return (total_loss_empi,
                [discrimn_loss_empi.item(), compress_loss_empi.item()],
                [discrimn_loss_theo.item(), compress_loss_theo.item()])

# align-uniform loss
class align_unif(torch.nn.Module):
    def __init__(self,alpha=2,t=2):
        super(align_unif, self).__init__()
        self.alpha = alpha
        self.t = t

    def align_loss(self, x, y):
        return (x - y).norm(p=2, dim=1).pow(self.alpha).mean()

    def uniform_loss(self, x):
        return torch.pdist(x, p=2).pow(2).mul(-self.t).exp().mean().log()
    
    def forward(self, x, y):
        align = self.align_loss(x, y)
        unif = self.uniform_loss(x)
        loss = align + unif
        return loss, [align.item(), unif.item()]

#clid loss
class CLID(torch.nn.Module):
    """
    CLID (Cluster Learnability and Intrinsic Dimension) Metric
    
    Implementation based on the original paper methodology:
    "Using Representation Expressiveness and Learnability to Evaluate Self-Supervised Learning Methods"
    
    This implementation strictly follows the paper's procedures:
    1. Intrinsic Dimension (ID): Uses TwoNN estimator as described in Facco et al. (2017)
    2. Cluster Learnability (CL): KNN classifier performance on clustering pseudo-labels
    3. CLID combination: CL + ID_normalized (or weighted version)
    
    Paper References:
    - Main paper: "Using Representation Expressiveness and Learnability to Evaluate Self-Supervised Learning Methods"
    - ID estimation: Facco, E., et al. (2017). "Estimating the intrinsic dimension of datasets by a minimal neighborhood information"
    - TwoNN method: Based on nearest neighbor distance ratios following Pareto distribution
    
    Key implementation details matching the paper:
    - Uses StandardScaler for feature preprocessing
    - K-means clustering for pseudo-label generation  
    - Train/validation split (50/50) with stratification
    - L2 normalization of train/val sets
    - KNN classifier for learnability assessment
    - ID normalization by feature dimension D
    """
    def __init__(self, k=2,
                       n_clusters=10, 
                       k_classifier=5,
                       use_weighted=False):
        
        super(CLID, self).__init__()
        self.k = k # number of nearest neighbors for intrinsic dimension estimation
        self.n_clusters = n_clusters
        self.k_classifier = k_classifier # number of nearest neighbors for clustering loss
        self.use_weighted = use_weighted
        self.id_ = None # intrinsic dimension value
    
    
    def intrinsic_dim(self, x):
        """
        Estimate intrinsic dimension using TwoNN method.
        
        Implementation follows Facco et al. (2017):
        - Uses ratio of distances to first and second nearest neighbors
        - Fits Pareto distribution with parameter μ = ID
        - Employs maximum likelihood estimation
        
        Args:
            x: Feature tensor (N, D)
        Returns:
            float: Estimated intrinsic dimension
        """
        x_np = x.detach().cpu().numpy()
        id_estimator = TwoNN()
        id_estimator.fit(x_np)
        self.id_ = id_estimator.dimension_
        return self.id_
    
    def cluster_loss(self, x):
        """
        Compute Cluster Learnability (CL) following paper methodology.
        
        Paper procedure implementation:
        1. StandardScaler normalization (as specified in paper)
        2. K-means clustering to generate pseudo-labels
        3. Stratified train/validation split (50/50)
        4. L2 normalization of features (per sample)
        5. KNN classifier training and evaluation
        6. Accuracy computation: CL = (1/N) * Σ[y_val == y_pred]
        
        Args:
            X: features matrix (N, D)
        Returns:
            float: clustering learnability (between 0 and 1, closer to 1 is better)
        """
        x_np = x.detach().cpu().numpy()
        
        # Step 1: StandardScaler normalization (as per paper)
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_np)

        # Step 2: K-means clustering for pseudo-labels (paper methodology)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        y = kmeans.fit_predict(x_scaled)

        # Step 3: Stratified split (50/50 as in paper)
        x_train, x_val, y_train, y_val = train_test_split(
            x_scaled, y, test_size=0.5, stratify=y, random_state=42
        )
        
        # Step 4: L2 normalization (per sample, as specified)
        x_train = normalize(x_train, norm='l2', axis=1)
        x_val = normalize(x_val, norm='l2', axis=1)        
        
        # Step 5: KNN classifier training (paper's approach)
        knn = KNeighborsClassifier(n_neighbors=self.k_classifier)
        knn.fit(x_train, y_train)

        # Step 6: Validation and accuracy computation (paper formula)
        y_pred = knn.predict(x_val)
        CL = np.mean(y_val == y_pred)  # Paper's CL formula: (1/N) * Σ[correct predictions]

        return CL

    def forward(self, x):
        id_val = self.intrinsic_dim(x)
        cl = self.cluster_loss(x)

        if np.isnan(id_val):
            id_val = 0.0
        
        if normalize:
            # CL already in [0,1] range
            # ID_normalized = min(ID / D, 1.0)
            D = x.shape[1]
            id_normalized = min(id_val / D, 1.0)
        else:
            id_normalized = id_val

        if self.use_weighted :
            clid = fit_weights[0] * cl + fit_weights[1] * id_normalized
        else:
            clid = cl + id_normalized
        
        return {
            'cl': cl,
            'id': id_val,
            'id_normalized': id_normalized,
            'clid': clid
        }
    
class rankme(torch.nn.Module):
    def __init__(self,eps=1e-8):
        super(rankme, self).__init__()
        self.eps = eps

    def forward(self, features):
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()

        features = features - features.mean(axis=0)   
       # 计算奇异值分解
        U, s, Vt = svd(features, full_matrices=False)
    
    # 计算奇异值的平方（对应特征矩阵的特征值）
        singular_values = s ** 2
    
    # 归一化奇异值
        singular_values = singular_values / singular_values.sum()
    
    # 过滤掉很小的奇异值
        singular_values = singular_values[singular_values > self.eps]
    
    # 计算有效秩（基于熵的定义）
        log_singular_values = np.log(singular_values + self.eps)
        rankme = np.exp(-np.sum(singular_values * log_singular_values))

        return rankme

def fit_weights( features_list, performance_list):
        """
        学习加权参数 w
        
        使用线性回归学习最优权重：
        performance = w^T [CL, ID]
        """
        # 计算所有模型的CL和ID
        #compute all models' CL and ID
        cl_values = []
        id_values = []

        for X, labels in features_list:
            result = CLID.compute_clid(X, labels, normalize=False)
            cl_values.append(result['cl'])
            id_values.append(result['id'])

        cl_values = np.array(cl_values).reshape(-1, 1)
        # 计算所有模型的CL和ID
        cl_values = []
        id_values = []
        
        for X, labels in features_list:
            result = CLID.compute_clid(X, labels, normalize=False)
            cl_values.append(result['cl'])
            id_values.append(result['id'])
        
        cl_values = np.array(cl_values).reshape(-1, 1)
        id_values = np.array(id_values).reshape(-1, 1)
        
        # 标准化CL和ID到[0,1]
        scaler_cl = StandardScaler()
        scaler_id = StandardScaler()
        cl_normalized = scaler_cl.fit_transform(cl_values).flatten()
        id_normalized = scaler_id.fit_transform(id_values).flatten()

        # 构建特征矩阵
        X_train = np.column_stack([cl_normalized, id_normalized])
        y_train = np.array(performance_list)
        
        # 线性回归
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        weights = model.coef_
        weights = weights / np.sum(weights)  # 归一化权重

        print(f"学习到的权重: CL={weights[0]:.4f}, ID={weights[1]:.4f}")

        return weights
    
def label_to_membership(targets, num_classes=None):
    """Generate a true membership matrix, and assign value to current Pi.

    Parameters:
        targets (np.ndarray): matrix with one hot labels

    Return:
        Pi: membership matirx, shape (num_classes, num_samples, num_samples)

    """
    targets = one_hot(targets, num_classes)
    num_samples, num_classes = targets.shape
    Pi = np.zeros(shape=(num_classes, num_samples, num_samples))
    for j in range(len(targets)):
        k = np.argmax(targets[j])
        Pi[k, j, j] = 1.
    return Pi

def one_hot(labels_int, n_classes):
    """Turn labels into one hot vector of K classes. """
    labels_onehot = torch.zeros(size=(len(labels_int), n_classes)).float()
    for i, y in enumerate(labels_int):
        labels_onehot[i, y] = 1.
    return labels_onehot


class RankMe(torch.nn.Module):
    """
    RankMe Metric Implementation
    
    Strictly based on the original RankMe paper methodology:
    "RankMe: Assessing the Downstream Performance of Pretrained Self-Supervised Representations by Their Rank"
    
    This implementation follows the paper's exact procedures:
    1. Feature centering: X = X - mean(X, axis=0)
    2. Singular Value Decomposition: U, s, V^T = SVD(X)
    3. Eigenvalue computation: λ_i = s_i^2 (squared singular values)
    4. Probability distribution: p_i = λ_i / Σλ_i (normalized eigenvalues)
    5. Entropy-based effective rank: RankMe = exp(-Σ p_i * log(p_i))
    
    Paper Reference:
    - "RankMe: Assessing the Downstream Performance of Pretrained Self-Supervised Representations by Their Rank"
    - Based on Eckart-Young-Mirsky theorem
    - Uses Shannon entropy of eigenvalue distribution
    - Provides measure of representational diversity
    
    Two variants implemented:
    - Full version: Entropy-based continuous rank (paper's main method)
    - Simple version: Threshold-based discrete rank (paper's alternative)
    
    Mathematical foundation from paper:
    - Effective rank captures "how many dimensions are effectively used"
    - Higher RankMe indicates more diverse, informative representations
    - Range: [1, min(N,D)] where N=samples, D=features
    """
    
    def __init__(self, eps=1e-12, threshold=0.01, use_simple=False):
        """
        Args:
            eps: float, numerical stability parameter for log computation (paper default: 1e-12)
            threshold: float, singular value threshold ratio for simple version (paper: 0.01)
            use_simple: bool, whether to use threshold-based simple version
        """
        super(RankMe, self).__init__()
        self.eps = eps
        self.threshold = threshold
        self.use_simple = use_simple
    
    def forward(self, features):
        """
        Compute RankMe metric following paper methodology.
        
        Args:
            features: torch.Tensor or np.ndarray, shape (N, D)
                     N samples, D-dimensional features
        
        Returns:
            float: RankMe value
                  - Full version: continuous rank in [1, min(N,D)]
                  - Simple version: discrete rank (integer)
        """
        if self.use_simple:
            return self._compute_simple(features)
        else:
            return self._compute_full(features)
    
    def _compute_full(self, features):
        """
        Compute full RankMe using entropy method (paper's main approach).
        
        Paper Algorithm:
        1. X = X - mean(X, axis=0)           # Center features
        2. U, s, V = SVD(X)                  # Singular value decomposition  
        3. λ = s^2                           # Eigenvalues of covariance matrix
        4. p = λ / sum(λ)                    # Probability distribution
        5. RankMe = exp(-sum(p * log(p)))    # Entropy-based effective rank
        """
        if isinstance(features, torch.Tensor):
            X = features.detach().cpu().numpy()
        else:
            X = features.copy()
        
        # Step 1: Center features (paper requirement)
        X = X - X.mean(axis=0, keepdims=True)
        
        # Step 2: Singular Value Decomposition (paper's core method)
        try:
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
        except np.linalg.LinAlgError:
            return float('nan')
        
        # Step 3: Compute eigenvalues (squared singular values, paper formula)
        eigenvals = s ** 2
        
        # Filter small eigenvalues for numerical stability
        eigenvals = eigenvals[eigenvals > self.eps]
        
        if len(eigenvals) == 0:
            return float('nan')
        
        # Step 4: Normalize to probability distribution (paper requirement)
        eigenvals = eigenvals / eigenvals.sum()
        
        # Step 5: Compute entropy-based effective rank (paper's main formula)
        log_eigenvals = np.log(eigenvals + self.eps)
        rankme = np.exp(-np.sum(eigenvals * log_eigenvals))
        
        return float(rankme)
    
    def _compute_simple(self, features):
        """
        Compute simplified RankMe using threshold method (paper's alternative).
        
        Paper's Simple Algorithm:
        1. X = X - mean(X, axis=0)           # Center features
        2. U, s, V = SVD(X)                  # SVD decomposition
        3. s_norm = s / s[0]                 # Normalize by largest singular value
        4. rank = sum(s_norm > threshold)    # Count significant components
        """
        if isinstance(features, torch.Tensor):
            X = features.detach().cpu().numpy()
        else:
            X = features.copy()
        
        # Step 1: Center features
        X = X - X.mean(axis=0, keepdims=True)
        
        # Step 2: SVD
        try:
            _, s, _ = np.linalg.svd(X, full_matrices=False)
        except np.linalg.LinAlgError:
            return 0
        
        if len(s) == 0 or s[0] <= 0:
            return 0
        
        # Step 3: Normalize by largest singular value (paper method)
        s_normalized = s / s[0]
        
        # Step 4: Count significant components (paper's threshold approach)
        effective_rank = int(np.sum(s_normalized > self.threshold))
        
        return effective_rank

