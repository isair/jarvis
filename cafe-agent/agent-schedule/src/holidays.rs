//! Latvian public holidays (fixed + Easter-based approximations for payroll hints).

pub fn is_lv_public_holiday(iso_date: &str) -> bool {
    lv_public_holidays_for_year(iso_date.get(..4).unwrap_or(""))
        .iter()
        .any(|d| *d == iso_date)
}

/// Fixed dates per year; movable feasts omitted in v1 (extend as needed).
pub fn lv_public_holidays_for_year(year: &str) -> &'static [&'static str] {
    match year {
        "2025" => &[
            "2025-01-01", "2025-05-01", "2025-05-04", "2025-06-23", "2025-06-24",
            "2025-11-18", "2025-12-24", "2025-12-25", "2025-12-26", "2025-12-31",
        ],
        "2026" => &[
            "2026-01-01", "2026-05-01", "2026-05-04", "2026-06-23", "2026-06-24",
            "2026-11-18", "2026-12-24", "2026-12-25", "2026-12-26", "2026-12-31",
        ],
        _ => &["2026-01-01"],
    }
}
