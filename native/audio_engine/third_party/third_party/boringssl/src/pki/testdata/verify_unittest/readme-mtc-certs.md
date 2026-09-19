# MTC test cert generation

The following test certs are created according to these instructions.

## Certs

- `mtc-leaf.pem`
- `mtc-leaf-bitflip.pem`
  - a copy of `mtc-leaf.pem`, but with a bitflip in its inclusion proof
- `mtc-leaf-b.pem`
- `mtc-leaf-c.pem`

## Instructions

- Run
  `go run github.com/ietf-plants-wg/merkle-tree-certs/demo@9029a99bcfa4e91b8b8e9ba646ac386a6e1c208f -config=mtc-config.json -out=out`
- copy/move the following output files:
  - `out/cert_9_0.pem` to `mtc-leaf.pem`
  - `out/cert_9_1.pem` to `mtc-leaf-bitflip.pem`
  - `out/cert_9_2.pem` to `mtc-leaf-unused-bit.pem`
  - `out/cert_10_0.pem` to `mtc-leaf-b.pem`
  - `out/cert_19_0.pem` to `mtc-leaf-c.pem`
- edit `VerifyMTCTest::SetUp` to set the trusted subtrees to the ones output by
  the above command.
- remove other artifacts created by the merkle-tree-certs/demo tool (e.g.
  `rm -r out`).
