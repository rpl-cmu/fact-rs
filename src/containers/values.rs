use std::{default::Default, fmt, fmt::Write, iter::IntoIterator, marker::PhantomData};

use foldhash::fast::RandomState;
use indexmap::{map::Entry, IndexMap};
use pad_adapter::PadAdapter;

use super::{
    symbol::{DefaultSymbolHandler, KeyFormatter},
    Key, Symbol, TypedSymbol,
};
use crate::{
    linear::LinearValues,
    variables::{VariableDtype, VariableSafe},
};

// Since we won't be passing dual numbers through any of this,
// we can just use dtype rather than using generics with Numeric

/// Structure to hold the Variables used in the graph.
///
/// Values is essentially a thin wrapper around a Hashmap that maps [Key] ->
/// [VariableSafe]. If you'd like to define a custom variable to be used in
/// Values, it must implement [Variable](crate::variables::Variable), and then
/// will implement [VariableSafe] via a blanket implementation.
/// ```
/// # use factrs::{
/// #    assign_symbols,
/// #    containers::Values,
/// #    variables::SO2,
/// # };
/// # assign_symbols!(X: SO2);
/// let x = SO2::from_theta(0.1);
/// let mut values = Values::new();
/// values.insert(X(0), x);
/// ```
#[derive(Default, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Values {
    values: IndexMap<Key, Box<dyn VariableSafe>, RandomState>,
}

impl Values {
    pub fn new() -> Self {
        Values::default()
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Returns an [std::collections::hash_map::Entry] from the underlying
    /// HashMap.
    pub fn entry(&mut self, key: impl Symbol) -> Entry<'_, Key, Box<dyn VariableSafe>> {
        self.values.entry(key.into())
    }

    pub fn insert<S, V>(&mut self, symbol: S, value: V) -> Option<Box<dyn VariableSafe>>
    where
        S: TypedSymbol<V>,
        V: VariableDtype,
    {
        self.values.insert(symbol.into(), Box::new(value))
    }

    /// Unchecked version of [Values::insert].
    pub fn insert_unchecked<S, V>(&mut self, symbol: S, value: V) -> Option<Box<dyn VariableSafe>>
    where
        S: Symbol,
        V: VariableDtype,
    {
        self.values.insert(symbol.into(), Box::new(value))
    }

    pub(crate) fn get_raw<S>(&self, symbol: S) -> Option<&dyn VariableSafe>
    where
        S: Symbol,
    {
        self.values.get(&symbol.into()).map(|f| f.as_ref())
    }

    /// Returns the underlying variable.
    ///
    /// This will return the value if variable is in the graph. Requires a typed
    /// symbol and as such is guaranteed to return the correct type. Returns
    /// None if key isn't found.
    /// ```
    /// # use factrs::{
    /// #    assign_symbols,
    /// #    containers::Values,
    /// #    variables::SO2,
    /// # };
    /// # assign_symbols!(X: SO2);
    /// # let x = SO2::from_theta(0.1);
    /// # let mut values = Values::new();
    /// # values.insert(X(0), x);
    /// let x_out = values.get(X(0));
    /// ```
    pub fn get<S, V>(&self, symbol: S) -> Option<&V>
    where
        S: TypedSymbol<V>,
        V: VariableDtype,
    {
        self.values
            .get(&symbol.into())
            .and_then(|value| value.downcast_ref::<V>())
    }

    /// Returns the underlying variable, not checking the type.
    pub fn get_unchecked<S, V>(&self, symbol: S) -> Option<&V>
    where
        S: Symbol,
        V: VariableDtype,
    {
        self.values
            .get(&symbol.into())
            .and_then(|value| value.downcast_ref::<V>())
    }

    /// Mutable version of [Values::get].
    pub fn get_mut<S, V>(&mut self, symbol: S) -> Option<&mut V>
    where
        S: TypedSymbol<V>,
        V: VariableDtype,
    {
        self.values
            .get_mut(&symbol.into())
            .and_then(|value| value.downcast_mut::<V>())
    }

    /// Mutable version of [Values::get_unchecked].
    pub fn get_unchecked_mut<S, V>(&mut self, symbol: S) -> Option<&mut V>
    where
        S: Symbol,
        V: VariableDtype,
    {
        self.values
            .get_mut(&symbol.into())
            .and_then(|value| value.downcast_mut::<V>())
    }

    pub fn remove<S, V>(&mut self, symbol: S) -> Option<V>
    where
        S: TypedSymbol<V>,
        V: VariableDtype,
    {
        self.values
            .shift_remove(&symbol.into())
            .and_then(|value| value.downcast::<V>().ok())
            .map(|value| *value)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Key, &Box<dyn VariableSafe>)> {
        self.values.iter()
    }

    /// Returns a iterator of references of all variables of a specific type in
    /// the values.
    ///
    /// ```
    /// # use factrs::{
    /// #    assign_symbols,
    /// #    containers::Values,
    /// #    traits::*,
    /// #    variables::SO2,
    /// # };
    /// # assign_symbols!(X: SO2);
    /// # let mut values = Values::new();
    /// # (0..10).for_each(|i| {values.insert(X(0), SO2::identity());} );
    /// let mine: Vec<&SO2> = values.filter().collect();
    /// ```
    pub fn filter<'a, T: 'a + VariableSafe>(&'a self) -> impl Iterator<Item = &'a T> {
        self.values
            .iter()
            .filter_map(|(_, value)| value.downcast_ref::<T>())
    }

    /// Update variables in place via the
    /// [oplus](crate::variables::Variable::oplus) operation.
    ///
    /// The [LinearValues] need to be setup to have the same keys and each key
    /// must have a variable of the same length.
    pub fn oplus_mut(&mut self, delta: &LinearValues) {
        // TODO: More error checking here
        for (key, value) in delta.iter() {
            if let Some(v) = self.values.get_mut(key) {
                assert!(v.dim() == value.len(), "Dimension mismatch in values oplus",);
                v.oplus_mut(value);
            }
        }
    }
}

impl fmt::Debug for Values {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&ValuesFormatter::<DefaultSymbolHandler>::new(self), f)
    }
}

impl fmt::Display for Values {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&ValuesFormatter::<DefaultSymbolHandler>::new(self), f)
    }
}

/// Formatter for values
///
/// Specifically, this can be used if custom symbols are desired. See
/// [tests/custom_key](https://github.com/rpl-cmu/factrs/blob/dev/tests/custom_key.rs) for examples.
pub struct ValuesFormatter<'v, KF> {
    values: &'v Values,
    kf: PhantomData<KF>,
}

impl<'v, KF> ValuesFormatter<'v, KF> {
    pub fn new(values: &'v Values) -> Self {
        Self {
            values,
            kf: Default::default(),
        }
    }
}

impl<KF: KeyFormatter> fmt::Display for ValuesFormatter<'_, KF> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let precision = f.precision().unwrap_or(3);
        if f.alternate() {
            f.write_str("Values {\n")?;
            let mut pad = PadAdapter::new(f);
            for (key, value) in self.values.iter() {
                KF::fmt(&mut pad, *key)?;
                writeln!(pad, ": {:#.p$},", value, p = precision)?;
            }
        } else {
            f.write_str("Values { ")?;
            for (key, value) in self.values.iter() {
                KF::fmt(f, *key)?;
                write!(f, ": {:.p$}, ", value, p = precision)?;
            }
        }
        f.write_str("}")
    }
}

impl<KF: KeyFormatter> fmt::Debug for ValuesFormatter<'_, KF> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let precision = f.precision().unwrap_or(3);
        if f.alternate() {
            f.write_str("Values {\n")?;
            let mut pad = PadAdapter::new(f);
            for (key, value) in self.values.iter() {
                KF::fmt(&mut pad, *key)?;
                writeln!(pad, ": {:#.p$?},", value, p = precision)?;
            }
        } else {
            f.write_str("Values { ")?;
            for (key, value) in self.values.iter() {
                KF::fmt(f, *key)?;
                write!(f, ": {:.p$?}, ", value, p = precision)?;
            }
        }
        f.write_str("}")
    }
}

impl IntoIterator for Values {
    type Item = (Key, Box<dyn VariableSafe>);
    type IntoIter = indexmap::map::IntoIter<Key, Box<dyn VariableSafe>>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.into_iter()
    }
}
