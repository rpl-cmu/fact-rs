use syn::{parse_macro_input, ItemImpl};

mod fac;
mod noise;
mod residual;
mod robust;
mod variable;

enum BoxedTypes {
    Residual,
    Variable,
    Noise,
    Robust,
}

fn check_type(input: &ItemImpl) -> syn::Result<BoxedTypes> {
    let err = syn::Error::new_spanned(input, "Missing trait");
    let result = &input
        .trait_
        .as_ref()
        .ok_or(err.clone())?
        .1
        .segments
        .last()
        .ok_or(err)?
        .ident
        .to_string();

    if result.contains("Residual") {
        Ok(BoxedTypes::Residual)
    } else if result.contains("Variable") {
        Ok(BoxedTypes::Variable)
    } else if result.contains("Noise") {
        Ok(BoxedTypes::Noise)
    } else if result.contains("Robust") {
        Ok(BoxedTypes::Robust)
    } else {
        Err(syn::Error::new_spanned(input, "Not a valid trait"))
    }
}

#[proc_macro_attribute]
pub fn mark(
    _args: proc_macro::TokenStream,
    input: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let input = syn::parse_macro_input!(input as ItemImpl);

    let trait_type = match check_type(&input) {
        Ok(syntax_tree) => syntax_tree,
        Err(err) => return err.to_compile_error().into(),
    };

    match trait_type {
        BoxedTypes::Residual => residual::mark(input),
        BoxedTypes::Variable => variable::mark(input),
        BoxedTypes::Noise => noise::mark(input),
        BoxedTypes::Robust => robust::mark(input),
    }
    .into()
}

#[proc_macro]
pub fn fac(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let factor = parse_macro_input!(input as fac::Factor);

    fac::fac(factor).into()
}
