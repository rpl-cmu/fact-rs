/*
This is mostly just used as a proof of concept to make sure this works
 so that users can do this if it's ever desired.

For example, this could be used for a multi-robot factor graph
*/

use std::fmt::{self, Write};

use factrs::{
    containers::{Key, KeyFormatter, Symbol, Values, ValuesFormatter},
    variables::VectorVar1,
};

// Char is stored in the high CHR_SIZE
// Idx is stored in the low IDX_SIZE
const TOTAL_SIZE: usize = u64::BITS as usize;
const CHR_SIZE: usize = 2 * 8;
const IDX_SIZE: usize = TOTAL_SIZE - 2 * CHR_SIZE;

const CHR1_MASK: u64 = (char::MAX as u64) << IDX_SIZE << CHR_SIZE;
const CHR2_MASK: u64 = (char::MAX as u64) << IDX_SIZE;
const IDX_MASK: u64 = !(CHR1_MASK | CHR2_MASK);

// Make custom formatter for our stuff
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DoubleCharHandler;

impl DoubleCharHandler {
    pub fn sym_to_key(chr1: char, chr2: char, idx: u32) -> Key {
        debug_assert!(chr1.is_ascii());
        debug_assert!(chr2.is_ascii());

        Key(((chr1 as u64) << IDX_SIZE << CHR_SIZE)
            | ((chr2 as u64) << IDX_SIZE)
            | (idx as u64) & IDX_MASK)
    }

    pub fn key_to_sym(k: Key) -> (char, char, u32) {
        let chr1 = ((k.0 & CHR1_MASK) >> IDX_SIZE >> CHR_SIZE) as u8 as char;
        let chr2 = ((k.0 & CHR2_MASK) >> IDX_SIZE) as u8 as char;
        let idx = (k.0 & IDX_MASK) as u32;
        (chr1, chr2, idx)
    }

    pub fn format(f: &mut dyn Write, chr1: char, chr2: char, idx: u32) -> fmt::Result {
        write!(f, "{chr1}{chr2}{idx}")
    }
}

impl KeyFormatter for DoubleCharHandler {
    fn fmt(f: &mut dyn Write, key: Key) -> fmt::Result {
        let (chr1, chr2, idx) = Self::key_to_sym(key);
        Self::format(f, chr1, chr2, idx)
    }
}

// Make a practice symbol to use
pub struct XY(pub u32);

impl From<XY> for Key {
    fn from(xy: XY) -> Key {
        DoubleCharHandler::sym_to_key('X', 'Y', xy.0)
    }
}

impl fmt::Debug for XY {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        DoubleCharHandler::format(f, 'X', 'Y', self.0)
    }
}

impl Symbol for XY {}

#[test]
fn test_round_trip() {
    let key = DoubleCharHandler::sym_to_key('X', 'Y', 101);
    let (chr1, chr2, idx) = DoubleCharHandler::key_to_sym(key);
    assert_eq!('X', chr1, "chr1 is off");
    assert_eq!('Y', chr2, "chr2 is off");
    assert_eq!(101, idx, "idx is off");
}

#[test]
fn test_values() {
    let mut values = Values::new();
    values.insert_unchecked(XY(1), VectorVar1::new(1.0));
    let output = format!("{}", ValuesFormatter::<DoubleCharHandler>::new(&values));
    assert_eq!(
        output, "Values { XY1: VectorVar1(1.000), }",
        "Values formatted wrong"
    );
}
