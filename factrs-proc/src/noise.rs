use proc_macro2::{Ident, TokenStream as TokenStream2};
use quote::quote;
use syn::{GenericParam, ItemImpl, Type, TypePath};

fn type_name(mut ty: &Type) -> Option<Ident> {
    loop {
        match ty {
            Type::Path(TypePath { qself: None, path }) => {
                return Some(path.segments.last().unwrap().ident.clone());
            }
            Type::Group(group) => {
                ty = &group.elem;
            }
            _ => return None,
        }
    }
}

pub fn mark(item: ItemImpl) -> TokenStream2 {
    if !cfg!(feature = "serde") {
        return quote! { #item };
    }

    let mut expanded = quote! {
        #[typetag::serde]
        #item
    };

    let name: Ident = type_name(&item.self_ty).unwrap();

    // Have to tag for serialization
    match item.generics.params.len() {
        // If no generics, just tag
        0 => {
            expanded.extend(quote!(
                factrs::noise::tag_noise!(#name);
            ));
        }
        // If one generic and it's const, do first 20
        1 => {
            if let GenericParam::Const(_) = item.generics.params.first().unwrap() {
                for i in 1usize..=32 {
                    let name_str = format!("{name}<{i}>");
                    expanded.extend(quote!(
                        typetag::__private::inventory::submit! {
                            <dyn factrs::noise::NoiseModel>::typetag_register(
                                #name_str,
                                (|deserializer| typetag::__private::Result::Ok(
                                    typetag::__private::Box::new(
                                        typetag::__private::erased_serde::deserialize::<#name<#i>>(deserializer)?
                                    ),
                                )) as typetag::__private::DeserializeFn<<dyn factrs::noise::NoiseModel as typetag::__private::Strictest>::Object>,
                            )
                        }
                    ));
                }
            }
        }
        // Anymore and it's up to the user
        _ => {}
    }

    expanded
}
