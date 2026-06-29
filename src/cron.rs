//! A small UNIX-cron expression parser with optional **second-level precision**.
//!
//! Accepts either a 5-field expression (`minute hour dom month dow`) — the
//! classic Vixie-cron form, minute-granular — or a 6-field expression
//! (`second minute hour dom month dow`) for sub-minute scheduling. This is
//! what lets the agent honor "in 10 seconds" requests: a job like
//! `*/10 * * * * *` fires every 10 seconds.
//!
//! Fields: second (0-59), minute, hour, day-of-month, month, day-of-week
//! (0-7, both 0 and 7 are Sunday). Supports `*`, comma lists, ranges
//! (`a-b`), and step values (`a-b/n` or `*/n`). Named ranges (Jan, Mon) are
//! intentionally not supported — numeric fields keep the parser
//! dependency-free and predictable.
//!
//! This is intentionally a small, self-contained parser rather than a pull of
//! a heavy cron crate: it gives us `next_after` for scheduling and full
//! control over edge-case semantics (DOM/DOW OR-matching as in Vixie cron).

use anyhow::{Result, bail};

/// Minimum / maximum (inclusive) bounds for each of the 5 standard cron fields
/// (minute, hour, day-of-month, month, day-of-week).
const FIELD_BOUNDS: [(u8, u8); 5] = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 7)];

/// Bounds for the optional seconds field.
const SECONDS_BOUNDS: (u8, u8) = (0, 59);

/// A compiled cron expression.
#[derive(Debug, Clone)]
pub struct Cron {
    /// One bitset per field; bit `i` set means "value i is selected".
    /// Index order: [minute, hour, day-of-month, month, day-of-week].
    fields: [u64; 5],
    /// Seconds bitset. `None` for a 5-field (minute-granular) expression, in
    /// which case only whole-minute instants are ever considered to match.
    seconds: Option<u64>,
    raw: String,
}

impl Cron {
    /// Parse a cron expression.
    ///
    /// 5 fields → minute-granular (classic Vixie form). 6 fields →
    /// second-level precision (the first field is seconds). Any other count is
    /// an error.
    pub fn parse(expr: &str) -> Result<Self> {
        let parts: Vec<&str> = expr.split_whitespace().collect();
        if parts.len() != 5 && parts.len() != 6 {
            bail!(
                "cron expression must have 5 or 6 fields, got {}: {expr:?}",
                parts.len()
            );
        }
        let (seconds, std) = match parts.len() {
            5 => (None, parts.as_slice()),
            // 6-field: first is seconds, rest are the standard 5.
            _ => (Some(parse_field(parts[0], SECONDS_BOUNDS)?), &parts[1..]),
        };
        let mut fields = [0u64; 5];
        for (i, part) in std.iter().enumerate() {
            fields[i] = parse_field(part, FIELD_BOUNDS[i])?;
        }
        Ok(Self {
            fields,
            seconds,
            raw: expr.trim().to_string(),
        })
    }

    /// Whether this expression has a seconds field (6-field form).
    pub fn has_seconds(&self) -> bool {
        self.seconds.is_some()
    }

    /// Original expression text.
    pub fn as_str(&self) -> &str {
        &self.raw
    }

    /// Whether the given instant (in UTC fields) matches the expression.
    ///
    /// Day-of-month and day-of-week follow Vixie cron semantics: when both are
    /// restricted (not wildcard), a match on *either* satisfies the day.
    pub fn matches(&self, minute: u8, hour: u8, day: u8, month: u8, weekday: u8) -> bool {
        let m = bit(self.fields[0], minute);
        let h = bit(self.fields[1], hour);
        let dom = bit(self.fields[2], day);
        let mon = bit(self.fields[3], month);
        let dom_wild = self.fields[2] == wildcard_mask(FIELD_BOUNDS[2]);
        let dow_wild = self.fields[4] == wildcard_mask(FIELD_BOUNDS[4]);
        // Normalize Sunday: both 0 and 7 represent Sunday; the dow bitset
        // already has bit 0 set when 7 is parsed (see parse_value).
        let dow = bit(self.fields[4], weekday);

        if !m || !h || !mon {
            return false;
        }

        if dom_wild || dow_wild {
            dom && dow
        } else {
            dom || dow
        }
    }

    /// Internal matcher including the seconds field. For a 5-field expression
    /// (`seconds == None`) the seconds bitset is treated as the wildcard "0"
    /// (only whole-minute instants are ever probed by `next_after`), so this
    /// reduces to [`matches`].
    fn matches_sec(
        &self,
        second: u8,
        minute: u8,
        hour: u8,
        day: u8,
        month: u8,
        weekday: u8,
    ) -> bool {
        if let Some(sec_bits) = self.seconds
            && !bit(sec_bits, second)
        {
            return false;
        }
        self.matches(minute, hour, day, month, weekday)
    }

    /// Find the next instant strictly after `epoch_secs` (UTC) that matches.
    ///
    /// For a 5-field (minute-granular) expression it iterates minute-by-minute;
    /// for a 6-field (second-precision) expression it iterates second-by-second
    /// so a `*/10 * * * * *` job can fire mid-minute. For the tick resolutions
    /// used here this is cheap (the scheduler's poll drives it, and `next_after`
    /// is only called to compute fire times lazily). Returns the matching epoch
    /// seconds.
    pub fn next_after(&self, epoch_secs: i64) -> i64 {
        let step: i64 = if self.seconds.is_some() { 1 } else { 60 };
        // Round up past `epoch_secs` to the next candidate instant on the
        // expression's granularity (whole minute for 5-field, whole second for
        // 6-field).
        let mut t = (epoch_secs / step + 1) * step;
        // Guards against pathological expressions that never match.
        let limit = epoch_secs + 366 * 24 * 60 * 60; // ~1 year
        while t <= limit {
            let (sec, min, hr, dom, mon, wd) = to_fields(t);
            if self.matches_sec(sec, min, hr, dom, mon, wd) {
                return t;
            }
            t += step;
        }
        t
    }
}

fn bit(mask: u64, val: u8) -> bool {
    (mask >> val) & 1 == 1
}

/// Mask with all in-range bits set — used to detect wildcard fields.
fn wildcard_mask((lo, hi): (u8, u8)) -> u64 {
    let mut m = 0u64;
    for v in lo..=hi {
        m |= 1u64 << v;
    }
    m
}

fn parse_field(field: &str, (lo, hi): (u8, u8)) -> Result<u64> {
    let mut mask = 0u64;
    for item in field.split(',') {
        mask |= parse_item(item, (lo, hi))?;
    }
    if mask == 0 {
        bail!("cron field {field:?} selects no values");
    }
    Ok(mask)
}

fn parse_item(item: &str, (lo, hi): (u8, u8)) -> Result<u64> {
    // Split off an optional step `/n`.
    let (range_part, step) = match item.split_once('/') {
        Some((r, s)) => (r, Some(parse_u8(s, (lo, hi))?)),
        None => (item, None),
    };

    let (start, end) = if range_part == "*" {
        (lo, hi)
    } else if let Some((a, b)) = range_part.split_once('-') {
        (parse_value(a, (lo, hi))?, parse_value(b, (lo, hi))?)
    } else {
        let v = parse_value(range_part, (lo, hi))?;
        (v, v)
    };

    if end < start {
        bail!("cron range {item:?} has end < start");
    }
    let step = step.unwrap_or(1).max(1);

    let mut mask = 0u64;
    let mut v = start;
    loop {
        if v > end {
            break;
        }
        mask |= 1u64 << v;
        v = v.saturating_add(step);
    }
    if mask == 0 {
        bail!("cron item {item:?} selects no in-range values");
    }
    Ok(mask)
}

/// Parse a single value, normalizing dow Sunday (7 → 0).
fn parse_value(s: &str, (lo, hi): (u8, u8)) -> Result<u8> {
    let v = parse_u8(s, (lo, hi))?;
    // Day-of-week field uses hi=7 to allow `7` for Sunday; collapse onto 0.
    if hi == 7 && v == 7 { Ok(0) } else { Ok(v) }
}

fn parse_u8(s: &str, (lo, hi): (u8, u8)) -> Result<u8> {
    let n: u32 = s
        .parse()
        .map_err(|_| anyhow::anyhow!("cron value {s:?} is not an integer"))?;
    if n > hi as u32 || n < lo as u32 {
        bail!("cron value {n} out of range [{lo},{hi}]");
    }
    Ok(n as u8)
}

/// Convert epoch seconds (UTC) to cron fields
/// (second, minute, hour, day-of-month, month, weekday).
/// Weekday: 0 = Sunday. Civil date breakdown via the well-known days-from-civil
/// algorithm (Howard Hinnant) — avoids pulling in a date crate.
fn to_fields(epoch_secs: i64) -> (u8, u8, u8, u8, u8, u8) {
    let days = epoch_secs.div_euclid(86400);
    let secs_of_day = epoch_secs.rem_euclid(86400);
    let hour = (secs_of_day / 3600) as u8;
    let minute = ((secs_of_day % 3600) / 60) as u8;
    let second = (secs_of_day % 60) as u8;

    // days-from-civil: converts a (y,m,d) <-> serial day count.
    // Convert days-since-1970-01-01 -> (year, month, day).
    let z = days + 719_468; // shift to 0000-03-01 epoch
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let year = if m <= 2 { y + 1 } else { y };

    // weekday: 1970-01-01 was a Thursday (4). 0=Sunday.
    let weekday = (((days % 7) + 4 + 7) % 7) as u8;

    let _ = year;
    (second, minute, hour, d as u8, m as u8, weekday)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_basic() {
        Cron::parse("*/5 * * * *").unwrap();
        Cron::parse("0 9 * * 1-5").unwrap();
        Cron::parse("30 2 1 1 *").unwrap();
        assert!(Cron::parse("* * * *").is_err());
        assert!(Cron::parse("60 * * * *").is_err());
    }

    #[test]
    fn every_minute_matches() {
        let c = Cron::parse("* * * * *").unwrap();
        assert!(c.matches(0, 0, 1, 1, 4));
        assert!(c.matches(59, 23, 31, 12, 0));
    }

    #[test]
    fn step_field() {
        let c = Cron::parse("*/15 * * * *").unwrap();
        assert!(c.matches(0, 0, 1, 1, 0));
        assert!(c.matches(15, 0, 1, 1, 0));
        assert!(c.matches(30, 0, 1, 1, 0));
        assert!(c.matches(45, 0, 1, 1, 0));
        assert!(!c.matches(7, 0, 1, 1, 0));
    }

    #[test]
    fn weekday_7_is_sunday() {
        let c = Cron::parse("0 0 * * 7").unwrap();
        // Sunday is weekday 0
        assert!(c.matches(0, 0, 1, 1, 0));
        assert!(!c.matches(0, 0, 1, 1, 1));
    }

    #[test]
    fn dom_or_dow_when_both_restricted() {
        // 0 0 13 * 5 → midnight on the 13th OR on Friday.
        let c = Cron::parse("0 0 13 * 5").unwrap();
        assert!(c.matches(0, 0, 13, 6, 3)); // 13th (Wed)
        assert!(c.matches(0, 0, 20, 6, 5)); // Friday the 20th
        assert!(!c.matches(0, 0, 14, 6, 3)); // 14th Tue neither
    }

    #[test]
    fn next_after_advances() {
        let c = Cron::parse("*/5 * * * *").unwrap();
        // epoch 0 = 1970-01-01 00:00:00 UTC (Thu). Next */5 mark after 0 → 00:05.
        assert_eq!(c.next_after(0), 300);
    }

    #[test]
    fn six_field_parses_and_is_second_precise() {
        let c = Cron::parse("*/10 * * * * *").unwrap();
        assert!(c.has_seconds());
        // epoch 0 = 00:00:00. Next */10s mark strictly after 0 → 00:00:10.
        assert_eq!(c.next_after(0), 10);
        assert_eq!(c.next_after(10), 20);
        assert_eq!(c.next_after(25), 30);
    }

    #[test]
    fn six_field_specific_second() {
        // Fires at second 30 of every minute.
        let c = Cron::parse("30 * * * * *").unwrap();
        assert!(c.has_seconds());
        // epoch 0 → next match at 00:00:30.
        assert_eq!(c.next_after(0), 30);
        // After 00:00:30, next is 00:01:30 (90s later).
        assert_eq!(c.next_after(30), 90);
    }

    #[test]
    fn rejects_wrong_field_counts() {
        assert!(Cron::parse("* * * *").is_err());
        assert!(Cron::parse("* * * * * * *").is_err());
        assert!(Cron::parse("60 * * * * *").is_err()); // seconds out of range
    }

    #[test]
    fn five_and_six_field_agree_on_minute_boundary() {
        // A 6-field `0 * * * * *` (second 0 of every minute) must land on the
        // same whole-minute instants as the 5-field `* * * * *`.
        let five = Cron::parse("* * * * *").unwrap();
        let six = Cron::parse("0 * * * * *").unwrap();
        for t in [0_i64, 59, 60, 121, 600] {
            assert_eq!(five.next_after(t), six.next_after(t), "at t={t}");
        }
    }
}
