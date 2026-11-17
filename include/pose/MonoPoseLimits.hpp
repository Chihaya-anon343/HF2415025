#pragma once

namespace gpnp {

// Translation-norm envelope shared by visual mono PnP and its gravity sidecar.
// Units are millimeters; the interval is open at both ends.
inline constexpr double kMonoMinTranslationNormMm = 10.0;
inline constexpr double kMonoMaxTranslationNormMm = 500000.0;

} // namespace gpnp
