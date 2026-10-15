# Online ML (compatibility namespace)

Compatibility namespace only. Recursive least squares (RLS) was relocated to the
linear-model family; the canonical documentation is
[`src/models/linear/rls/README.md`](../linear/rls/README.md) and the canonical
import path is `seqvex::models::linear::rls`.

The `models::online` and `models::online::rls` paths re-export the same canonical
`Rls`, `RlsState`, `RlsSample`, `RlsError`, and `RlsBatchError` types and add no
behavior. See the [vertical-slice evidence](../../../docs/ML_VERTICAL_SLICES.md).

This notice is navigation only; the canonical RLS README is authoritative for the
RLS contract.
