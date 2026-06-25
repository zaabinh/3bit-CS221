"""
Manual TF-IDF Vectorizer — a from-scratch re-implementation of
sklearn.feature_extraction.text.TfidfVectorizer.

Only standard-library + numpy + scipy are used; scikit-learn is NOT required.
"""

import math
from collections import Counter

import numpy as np
import scipy.sparse as sp


class ManualTfidfVectorizer:
    """Manually implemented TF-IDF vectorizer.

    Parameters
    ----------
    max_features : int or None
        Keep only the top ``max_features`` terms ordered by total term
        frequency across the training corpus.  ``None`` keeps all terms.
    ngram_range : tuple (min_n, max_n)
        The range of n-gram sizes to extract.  ``(1, 2)`` produces both
        unigrams and bigrams.
    sublinear_tf : bool
        If ``True``, apply sublinear TF scaling: ``tf <- 1 + log(tf)``
        for ``tf > 0``.
    """

    def __init__(
        self,
        max_features=None,
        ngram_range=(1, 1),
        sublinear_tf=False,
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf

        # Attributes populated after fit()
        self.vocabulary_ = None
        self.idf_ = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenize(self, document):
        """Split a document into whitespace tokens."""
        return document.lower().split()

    def _extract_ngrams(self, tokens):
        """Generate all n-grams in the configured range from *tokens*."""
        ngrams = []
        min_n, max_n = self.ngram_range
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngrams.append(" ".join(tokens[i : i + n]))
        return ngrams

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, raw_documents):
        """Learn vocabulary and IDF from *raw_documents*.

        Parameters
        ----------
        raw_documents : iterable of str
            An iterable yielding one document string per element.

        Returns
        -------
        self
        """
        raw_documents = list(raw_documents)
        n_docs = len(raw_documents)

        # 1. Count total TF and document frequency for every n-gram ----
        total_tf = Counter()
        doc_freq = Counter()

        for doc in raw_documents:
            tokens = self._tokenize(doc)
            ngrams = self._extract_ngrams(tokens)
            total_tf.update(ngrams)
            # Each unique n-gram in the document counts once for DF
            doc_freq.update(set(ngrams))

        # 2. Select top max_features by total TF ----------------------
        if self.max_features is not None:
            selected = total_tf.most_common(self.max_features)
        else:
            selected = total_tf.most_common()

        # Sort alphabetically for deterministic column ordering
        terms = sorted([term for term, _ in selected])

        # 3. Build vocabulary mapping (term -> column index) -----------
        self.vocabulary_ = {term: idx for idx, term in enumerate(terms)}

        # 4. Compute smooth IDF ----------------------------------------
        #    idf(t) = log((1 + N) / (1 + df(t))) + 1
        n_features = len(self.vocabulary_)
        self.idf_ = np.empty(n_features, dtype=np.float64)
        for term, idx in self.vocabulary_.items():
            df = doc_freq.get(term, 0)
            self.idf_[idx] = math.log((1 + n_docs) / (1 + df)) + 1

        return self

    def transform(self, raw_documents):
        """Transform *raw_documents* to a TF-IDF sparse matrix.

        Parameters
        ----------
        raw_documents : iterable of str

        Returns
        -------
        scipy.sparse.csr_matrix of shape (n_documents, n_features)
        """
        if self.vocabulary_ is None or self.idf_ is None:
            raise RuntimeError("Vectorizer has not been fitted yet.")

        raw_documents = list(raw_documents)
        n_docs = len(raw_documents)
        n_features = len(self.vocabulary_)

        # Build the sparse TF-IDF matrix in COO format for efficiency
        rows = []
        cols = []
        data = []

        for doc_idx, doc in enumerate(raw_documents):
            tokens = self._tokenize(doc)
            ngrams = self._extract_ngrams(tokens)
            tf_counts = Counter(ngrams)

            for term, count in tf_counts.items():
                col_idx = self.vocabulary_.get(term)
                if col_idx is None:
                    continue  # term not in vocabulary

                # Sublinear TF
                tf = 1 + math.log(count) if self.sublinear_tf else float(count)

                # TF x IDF
                tfidf = tf * self.idf_[col_idx]

                rows.append(doc_idx)
                cols.append(col_idx)
                data.append(tfidf)

        # Assemble sparse matrix
        matrix = sp.csr_matrix(
            (data, (rows, cols)),
            shape=(n_docs, n_features),
            dtype=np.float64,
        )

        # L2 normalisation (row-wise) ---------------------------------
        # Compute L2 norm per row
        row_norms = sp.linalg.norm(matrix, axis=1)  # shape (n_docs,)
        # Avoid division by zero
        row_norms[row_norms == 0] = 1.0
        # Divide each row by its norm via diagonal matrix
        inv_norms = sp.diags(1.0 / row_norms)
        matrix = inv_norms @ matrix

        return matrix

    def fit_transform(self, raw_documents):
        """Fit and transform in one step."""
        return self.fit(raw_documents).transform(raw_documents)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"ManualTfidfVectorizer("
            f"max_features={self.max_features}, "
            f"ngram_range={self.ngram_range}, "
            f"sublinear_tf={self.sublinear_tf})"
        )
