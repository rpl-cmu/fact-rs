use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{parse_quote, ItemImpl, Path};

fn parse_residual_trait(item: &ItemImpl) -> syn::Result<(Path, u32)> {
    let err = syn::Error::new_spanned(item, "unable to parse residual number");
    let residual_trait = item.trait_.clone().ok_or(err.clone())?.1;

    let num = residual_trait
        .segments
        .last()
        .ok_or(err.clone())?
        .ident
        .to_string()
        .replace("Residual", "")
        .parse::<u32>();

    match num {
        Result::Err(_) => Err(err),
        Result::Ok(n) => Ok((residual_trait, n)),
    }
}

pub fn mark(mut item: ItemImpl) -> TokenStream2 {
    // Parse what residual number we're using
    let (residual_trait, num) = match parse_residual_trait(&item) {
        Result::Err(e) => return e.to_compile_error(),
        Result::Ok(n) => n,
    };

    // Build all the things we need from it
    let residual_values = format_ident!("residual{}_values", num);
    let residual_jacobian = format_ident!("residual{}_jacobian", num);

    // If we should add typetag::Tagged to the generic bounds
    let typetag = if cfg!(feature = "serde") {
        // Add where clauses to all impl
        let all_type_params: Vec<_> = item.generics.type_params().cloned().collect();
        for type_param in all_type_params {
            let ident = &type_param.ident;
            item.generics.make_where_clause();
            item.generics
                .where_clause
                .as_mut()
                .unwrap()
                .predicates
                .push(parse_quote!(#ident: typetag::Tagged));
        }

        quote!( #[typetag::serde] )
    } else {
        TokenStream2::new()
    };

    // Build the output
    let generics = &item.generics;
    let self_ty = &item.self_ty;
    let where_clause = &generics.where_clause;

    quote! {
        #item

        #typetag
        impl #generics factrs::residuals::Residual for #self_ty #where_clause {
            fn dim_in(&self) -> usize {
                <<Self as #residual_trait>::DimIn as factrs::linalg::DimName>::DIM
            }

            fn dim_out(&self) -> usize {
                <<Self as  #residual_trait>::DimOut as factrs::linalg::DimName>::DIM
            }

            fn residual(&self, values: &factrs::containers::Values, keys: &[factrs::containers::Key]) -> factrs::linalg::VectorX {
                #residual_trait::#residual_values(self, values, keys)
            }

            fn residual_jacobian(&self, values: &factrs::containers::Values, keys: &[factrs::containers::Key]) -> factrs::linalg::DiffResult<factrs::linalg::VectorX, factrs::linalg::MatrixX> {
                #residual_trait::#residual_jacobian(self, values, keys)
            }
        }
    }
}
