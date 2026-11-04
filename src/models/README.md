# Models

Seqvex model-family index. The canonical public paths are:

| Family | Algorithm | Canonical path | Documentation |
|---|---|---|---|
| `linear` | OLS | `seqvex::models::linear::ols` | [OLS](linear/ols/README.md) |
| `linear` | RLS | `seqvex::models::linear::rls` | [RLS](linear/rls/README.md) |
| `linear` | SGD | `seqvex::models::linear::sgd` | [SGD](linear/sgd/README.md) |
| `trees` | Decision Tree | `seqvex::models::trees::decision_tree` | [Decision Tree](trees/decision_tree/README.md) |
| `neighbors` | KNN | `seqvex::models::neighbors::knn` | [KNN](neighbors/knn/README.md) |
| `recurrent` | GRU | `seqvex::models::recurrent::gru` | [Recurrent](recurrent/README.md) |

## Namespace migration (#27)

The former `models::classic` and `models::online` namespaces have been removed. There are no deprecated compatibility aliases. Type identities, algorithm behavior, numerical and precision semantics, state/execution ownership, and Cargo target names/paths are unchanged; this migration changes import paths and documentation locations only.

Both former classic OLS import forms and both former online RLS import forms are covered:

| Old import / path | New import / path |
|---|---|
| `seqvex::models::classic::linear_regression::{LinearRegression, RegressionError}` | `seqvex::models::linear::ols::{LinearRegression, RegressionError}` |
| `seqvex::models::classic::{LinearRegression, RegressionError}` | `seqvex::models::linear::ols::{LinearRegression, RegressionError}` |
| `seqvex::models::classic::decision_tree::{DecisionTree, DecisionTreeError, DecisionTreeNode}` | `seqvex::models::trees::decision_tree::{DecisionTree, DecisionTreeError, DecisionTreeNode}` |
| `seqvex::models::classic::{DecisionTree, DecisionTreeError, DecisionTreeNode}` | `seqvex::models::trees::decision_tree::{DecisionTree, DecisionTreeError, DecisionTreeNode}` |
| `seqvex::models::classic::knn::{Knn, KnnError, Reference}` | `seqvex::models::neighbors::knn::{Knn, KnnError, Reference}` |
| `seqvex::models::classic::{Knn, KnnError, Reference}` | `seqvex::models::neighbors::knn::{Knn, KnnError, Reference}` |
| `seqvex::models::online::rls::{Rls, RlsState, RlsSample, RlsError, RlsBatchError, RlsStateAdmissionError}` | `seqvex::models::linear::rls::{Rls, RlsState, RlsSample, RlsError, RlsBatchError, RlsStateAdmissionError}` |
| `seqvex::models::online::{Rls, RlsState, RlsSample, RlsError, RlsBatchError, RlsStateAdmissionError}` | `seqvex::models::linear::rls::{Rls, RlsState, RlsSample, RlsError, RlsBatchError, RlsStateAdmissionError}` |
| `seqvex::models::classic` (module) | Removed; no replacement module |
| `seqvex::models::online` (module) | Removed; no replacement module |

Cargo target identities are unchanged: `linear_regression`, `decision_tree`, `knn`, `rls` and `sgd` keep their existing names and paths.

Historical findings and numbers remain in [`docs/ALGORITHM_CORRECTNESS_AUDIT.md`](../../docs/ALGORITHM_CORRECTNESS_AUDIT.md) and [`docs/STREAMING_ML_DESIGN.md`](../../docs/STREAMING_ML_DESIGN.md). The historical #23 fixed-coefficient prediction evidence referenced from the OLS README is preserved at the revision-pinned [`src/models/classic/README.md`](https://github.com/Skybrique/seqvex/blob/a050a9359e054af17c99fc06e7d1bfb7dc0b6a01/src/models/classic/README.md).

The [mandatory training-to-inference contract](../../docs/DEVELOPMENT.md#model-training-and-valid-inference--mandatory) still applies to every model. Decision Tree training/fitting and KNN classification/weighting/ANN remain separate work.

The former `models::online` namespace is removed. Whether adaptive or online parameter updates belong in a dedicated model family remains a deferred architecture decision; RLS and SGD remain in `models::linear`.
