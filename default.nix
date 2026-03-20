{
  sources ? import ./npins,
  nixpkgs ? sources.nixpkgs,
  pkgs ? import nixpkgs { },
  devShell ? false,
}:
(pkgs.callPackage ./package.nix { }).overrideAttrs (prev: {
  nativeBuildInputs =
    prev.nativeBuildInputs
    ++ (pkgs.lib.optionals devShell [
      pkgs.clippy
      pkgs.rustfmt
      pkgs.rust-analyzer
      pkgs.npins
      pkgs.git
    ]);
})
