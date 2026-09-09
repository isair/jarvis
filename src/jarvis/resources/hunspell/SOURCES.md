# Vendored Hunspell dictionaries

Plain `*.aff` + `*.dic` pairs read directly by `spylls` (`Dictionary.from_files`). No
LibreOffice runtime, no system Hunspell, no network access at runtime.

## Upstreams

| Dictionary id | Upstream | Source files | Upstream commit |
|---|---|---|---|
| `en_US` | <https://github.com/LibreOffice/dictionaries> | `en/en_US.aff`, `en/en_US.dic` | `32b006a2c22a4ac7e8ed3f03346f7b3d85a970a4` |
| `cs_CZ` | <https://github.com/LibreOffice/dictionaries> | `cs_CZ/cs_CZ.aff`, `cs_CZ/cs_CZ.dic` | `32b006a2c22a4ac7e8ed3f03346f7b3d85a970a4` |
| `sk_SK` | <https://github.com/LibreOffice/dictionaries> | `sk_SK/sk_SK.aff`, `sk_SK/sk_SK.dic` | `32b006a2c22a4ac7e8ed3f03346f7b3d85a970a4` |
| `vi_VN` | <https://github.com/1ec5/hunspell-vi> | `dictionaries/vi-DauMoi.aff` -> `vi_VN/vi_VN.aff`, `dictionaries/vi-DauMoi.dic` -> `vi_VN/vi_VN.dic` | `39cc647f0cf3bdfc579980a2c3aac3f6f4203fe6` |

Vendor date: 2026-09-09. Cloned with `--depth 1 --filter=blob:none --sparse`, then
`sparse-checkout set en cs_CZ sk_SK` (LibreOffice) and `sparse-checkout set dictionaries`
(hunspell-vi). Only the eight listed files are copied; the upstream `*.mk`, `description.xml`,
`dictionaries.xcu`, hyphenation (`hyph_*`) and thesaurus (`th_*`) files are not shipped.

## File mapping

```
en/en_US.aff             -> hunspell/en_US/en_US.aff
en/en_US.dic             -> hunspell/en_US/en_US.dic
cs_CZ/cs_CZ.aff          -> hunspell/cs_CZ/cs_CZ.aff
cs_CZ/cs_CZ.dic          -> hunspell/cs_CZ/cs_CZ.dic
sk_SK/sk_SK.aff          -> hunspell/sk_SK/sk_SK.aff
sk_SK/sk_SK.dic          -> hunspell/sk_SK/sk_SK.dic
dictionaries/vi-DauMoi.aff -> hunspell/vi_VN/vi_VN.aff
dictionaries/vi-DauMoi.dic -> hunspell/vi_VN/vi_VN.dic
```

The Vietnamese pair is the reformed accent style (`DauMoi`, the style used in Vietnam). The
older/overseas placement variant `vi-DauCu` is deliberately not shipped.

## Licenses

- `en_US`: SCOWL-derived WordList dictionary (`http://wordlist.sourceforge.net`), dictionary
  version 2020.12.07. Bundled upstream under GPL-2.0+ (`en/license.txt`); the SCOWL source
  terms are a permissive public-domain-style grant.
- `cs_CZ`: Czech spellcheck dictionary for LibreOffice, originally based on the Czech ispell
  dictionary by Petr Kolár and contributors. Upstream pack is released under GPL/LGPL/MPL
  terms as stated in `cs_CZ/README_en.txt`.
- `sk_SK`: sk-spell project (`http://sk-spell.sk.cx/`, `ispell-sk`), snapshot 2024-08-29,
  triple-licensed and selectable: GPL-2.0+, LGPL-2.1+, or MPL-1.1 (`sk_SK/LICENSE.txt`).
- `vi_VN`: hunspell-vi package, derived from the GNU Aspell 0.60 Vietnamese dictionary
  (GPL), converted by László Németh; `.aff` rules improved by Ivan Garcia (2007).

## Known upstream limitation for `vi`

The upstream README notes that Hunspell alone cannot decide the internal boundaries of
Vietnamese multi-word compound words, so the spaced tokens of a compound are not joined or
split by the engine. The postprocessor mirrors that: for `vi` the Whisper token segmentation
is preserved exactly and only one-token-to-one-token replacements happen.
