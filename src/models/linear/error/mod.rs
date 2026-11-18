//! Shared linear-family error definitions.
//!
//! Algorithm-specific public errors, dimension admission and mappings remain in
//! their RLS/SGD modules. This module owns only family-shared private
//! classifications.

pub(crate) mod strict_serving;

pub(crate) use strict_serving::StrictWeightConversionError;
